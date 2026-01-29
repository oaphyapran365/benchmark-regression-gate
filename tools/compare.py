#!/usr/bin/env python3
"""Compare pytest-benchmark JSON outputs and fail on performance regression.

Designed for CI gating:
- Reads a baseline JSON (main branch) and a current JSON (PR)
- Compares a chosen robust metric (default: min for CI stability)
- Writes a Markdown report with a clear PASS/FAIL summary
- Exits non-zero if:
  (a) any benchmark regresses beyond the threshold, OR
  (b) any baseline benchmark is missing from the current run
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple, Optional


@dataclass(frozen=True)
class BenchResult:
    name: str
    value_seconds: float


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"ERROR: File not found: {path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"ERROR: Invalid JSON in {path}: {e}")


def _bench_key(entry: dict) -> str:
    # Prefer unique identifiers when available.
    return str(entry.get("fullname") or entry.get("name") or "unknown")


def _extract_stat(entry: dict, metric: str) -> Optional[float]:
    stats = entry.get("stats") or {}
    value = stats.get(metric)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fallback_metrics(metric: str) -> list[str]:
    # If the chosen metric isn't present, fall back in a stable order.
    # (min may be missing in some formats; median tends to be more stable than mean)
    if metric == "min":
        return ["min", "median", "mean"]
    if metric == "median":
        return ["median", "mean"]
    return ["mean"]


def load_stats(pytest_bench_json: Path, metric: str) -> Dict[str, BenchResult]:
    data = _load_json(pytest_bench_json)
    benches = data.get("benchmarks")
    if not isinstance(benches, list):
        raise SystemExit(
            f"ERROR: Expected key 'benchmarks' (list) in {pytest_bench_json}, got: {type(benches)}"
        )

    out: Dict[str, BenchResult] = {}
    for b in benches:
        if not isinstance(b, dict):
            continue
        key = _bench_key(b)

        val = None
        for m in _fallback_metrics(metric):
            val = _extract_stat(b, m)
            if val is not None:
                break

        if val is None:
            continue

        out[key] = BenchResult(name=key, value_seconds=val)

    if not out:
        raise SystemExit(
            f"ERROR: No benchmark stats parsed from {pytest_bench_json}. "
            "Make sure pytest-benchmark ran and produced stats."
        )

    return out


def pct_change(baseline: float, current: float) -> float:
    if baseline <= 0:
        return 0.0
    return (current - baseline) / baseline


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000.0:.3f}"


def write_report(
    report_path: Path,
    threshold: float,
    metric: str,
    rows: Iterable[Tuple[str, float, float, float, bool]],
    missing: Iterable[str],
) -> None:
    missing = sorted(missing)
    rows = list(rows)

    # Compute worst regression for summary
    worst = None  # (name, change)
    for name, _, _, ch, _ in rows:
        if worst is None or ch > worst[1]:
            worst = (name, ch)

    any_regression = any(not ok for _, _, _, _, ok in rows)

    status = "PASSED ✅" if (not missing and not any_regression) else "FAILED ❌"
    worst_str = "N/A"
    if worst is not None:
        worst_str = f"{fmt_pct(worst[1])} ({worst[0]})"

    lines = []
    lines.append("# Performance Regression Report\n\n")
    lines.append(f"**Status:** {status}  \n")
    lines.append(f"**Worst change:** {worst_str}  \n")
    lines.append(f"**Metric:** `{metric}`  \n")
    lines.append(f"**Threshold:** {fmt_pct(threshold)} (fail if above)  \n")

    if missing:
        lines.append("\n## Missing benchmarks (FAIL)\n")
        lines.append("Baseline benchmarks missing from current run:\n\n")
        for m in missing:
            lines.append(f"- `{m}`\n")

    lines.append("\n| Benchmark | Baseline (ms) | PR (ms) | Change | Status |\n")
    lines.append("|---|---:|---:|---:|:---:|\n")

    for name, b, c, ch, ok in rows:
        icon = "✅" if ok else "❌"
        lines.append(
            f"| `{name}` | {fmt_ms(b)} | {fmt_ms(c)} | {fmt_pct(ch)} | {icon} |\n"
        )

    if missing or any_regression:
        lines.append("\n## ❌ Gate FAILED\n")
        lines.append(
            "This PR introduces a performance regression beyond the threshold, "
            "or removed a benchmark.\n"
        )
    else:
        lines.append("\n## ✅ Gate PASSED\n")
        lines.append("No benchmarks regressed beyond the configured threshold.\n")

    report_path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare pytest-benchmark JSONs and gate on regressions.")
    ap.add_argument("--baseline", required=True, help="Path to baseline benchmark JSON")
    ap.add_argument("--current", required=True, help="Path to current/PR benchmark JSON")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.07,
        help="Allowed regression ratio (0.07 = 7%% slower allowed)",
    )
    ap.add_argument(
        "--metric",
        default="min",
        choices=["min", "median", "mean"],
        help="Statistic to compare (min is most CI-stable on shared runners)",
    )
    ap.add_argument("--report", required=True, help="Output Markdown report path")
    args = ap.parse_args()

    baseline_path = Path(args.baseline)
    current_path = Path(args.current)
    report_path = Path(args.report)

    baseline = load_stats(baseline_path, args.metric)
    current = load_stats(current_path, args.metric)

    missing = set(baseline.keys()) - set(current.keys())
    common = sorted(set(baseline.keys()) & set(current.keys()))

    rows = []
    failed = False
    for name in common:
        b = baseline[name].value_seconds
        c = current[name].value_seconds
        ch = pct_change(b, c)
        ok = ch <= args.threshold
        rows.append((name, b, c, ch, ok))
        if not ok:
            failed = True

    write_report(report_path, args.threshold, args.metric, rows, missing)

    if missing:
        failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
