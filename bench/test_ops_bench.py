import os

import pytest
import torch

# ---- CI stability knobs: reduce variance on shared runners ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

torch.manual_seed(0)

DEVICE = "cpu"
DTYPE = torch.float32

# Pedantic benchmarking with multiple iterations per round is typically more stable
# on shared CI runners than single-call timing.
PEDANTIC_ROUNDS = 15
PEDANTIC_ITERS = 5


def _bench(benchmark, fn):
    benchmark.pedantic(fn, rounds=PEDANTIC_ROUNDS, iterations=PEDANTIC_ITERS)


@pytest.mark.parametrize("n", [512, 768])
def test_bench_matmul(benchmark, n):
    a = torch.randn((n, n), device=DEVICE, dtype=DTYPE)
    b = torch.randn((n, n), device=DEVICE, dtype=DTYPE)

    def op():
        torch.matmul(a, b)

    _bench(benchmark, op)


def test_bench_addmm(benchmark):
    # addmm ~= GEMM + bias (common in linear layers)
    a = torch.randn((1024, 512), device=DEVICE, dtype=DTYPE)
    b = torch.randn((512, 512), device=DEVICE, dtype=DTYPE)
    bias = torch.randn((1024, 512), device=DEVICE, dtype=DTYPE)

    def op():
        torch.addmm(bias, a, b)

    _bench(benchmark, op)


def test_bench_conv2d(benchmark):
    import torch.nn.functional as F

    # Moderate, stable conv workload (not too small, not memory-heavy)
    x = torch.randn((4, 32, 64, 64), device=DEVICE, dtype=DTYPE)
    w = torch.randn((64, 32, 3, 3), device=DEVICE, dtype=DTYPE)

    def op():
        F.conv2d(x, w, stride=1, padding=1)

    _bench(benchmark, op)


def test_bench_layernorm(benchmark):
    # Larger layernorm workload to reduce relative noise on shared CI
    import torch.nn.functional as F

    x = torch.randn((2048, 4096), device=DEVICE, dtype=DTYPE)
    weight = torch.ones((4096,), device=DEVICE, dtype=DTYPE)
    bias = torch.zeros((4096,), device=DEVICE, dtype=DTYPE)

    def op():
        F.layer_norm(x, (4096,), weight=weight, bias=bias)

    _bench(benchmark, op)


def test_bench_softmax(benchmark):
    # Larger softmax workload to reduce relative noise on shared CI
    x = torch.randn((2048, 4096), device=DEVICE, dtype=DTYPE)

    def op():
        torch.softmax(x, dim=-1)

    _bench(benchmark, op)
