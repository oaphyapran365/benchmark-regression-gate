# benchmark-regression-gate

CPU-only **performance regression gate** for PyTorch ops using `pytest-benchmark`.
Readme Update
## What this repo does
- Benchmarks a small set of PyTorch ops on every PR.
- Compares PR results to a committed baseline (`bench/baseline.json`).
- Fails CI if any benchmark regresses by more than **7%**.
- Produces a Markdown report (`bench/report.md`) as a workflow artifact.

## Local quick start
1) Install dependencies:
   - `python -m pip install -r requirements.txt`

2) Generate a baseline (commit this file):
   - `pytest -q bench --benchmark-json=bench/baseline.json --benchmark-disable-gc --benchmark-warmup=on --benchmark-warmup-iterations=15 --benchmark-sort=name`

3) Run benchmarks (current results):
   - `pytest -q bench --benchmark-json=bench/current.json --benchmark-disable-gc --benchmark-warmup=on --benchmark-warmup-iterations=15 --benchmark-sort=name`

4) Compare against baseline (exits non-zero on regression):
   - `python tools/compare.py --baseline bench/baseline.json --current bench/current.json --threshold 0.07 --metric min --report bench/report.md`

## CI
GitHub Actions runs the same steps on every Pull Request and uploads `bench/report.md` as an artifact.
