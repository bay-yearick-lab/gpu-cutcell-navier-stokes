"""GPU versus CPU sparse-solver benchmark on 3D aggregated cut-cell systems.

Measures, per background resolution, how a consumer GPU (RTX 5060 Ti, 16 GB)
compares with the CPU sparse toolchain on the screened-Poisson proxy ``A = K + H`` assembled from the aggregated hollow-cube
bell?  The benchmark reports, per background resolution:

* Jacobi-preconditioned CG wall time on the CPU (SciPy, single-threaded
  sparse matvec) and on the GPU (CuPy CSR), with the device transfer time
  charged separately;
* the direct sparse factorization route (SuperLU factor + solve) that the
  production drivers actually use;
* iteration counts, which must agree between backends.

Requires CuPy (``uv sync --group gpu``).  Outputs
``benchmarks/results/gpu_solver_benchmark_3d.{json,csv}``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Sequence

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from bell_3d_core import (
    Bell3DGeometry,
    assemble_bell_3d,
    polynomial_preserving_extension_3d,
    restrict_spd,
)
from fast_solver_backends import solve_spd

RESULTS = ROOT / "benchmarks" / "results"


def build_system(n_elements: int, degree: int, epsilon: float) -> tuple[sp.csr_matrix, np.ndarray]:
    geom = Bell3DGeometry.from_grid(20, epsilon)
    asm = assemble_bell_3d(geom, n_elements=n_elements, degree=degree, with_flux=False)
    agg = polynomial_preserving_extension_3d(asm)
    M = restrict_spd(asm.H, agg.E)
    K = restrict_spd(asm.K, agg.E)
    A = (K + M).tocsr()
    b = np.asarray(agg.E.T @ asm.load_full)
    return A, b


def bench_case(n_elements: int, degree: int, epsilon: float, *, tol: float, maxiter: int) -> dict:
    t0 = perf_counter()
    A, b = build_system(n_elements, degree, epsilon)
    t_assembly = perf_counter() - t0

    row: dict = {
        "n_elements": int(n_elements),
        "degree": int(degree),
        "epsilon": float(epsilon),
        "n_unknowns": int(A.shape[0]),
        "nnz": int(A.nnz),
        "assembly_s": float(t_assembly),
    }

    x_cpu, rep_cpu = solve_spd(A, b, backend="cpu", tol=tol, maxiter=maxiter)
    row["cpu_cg_s"] = float(rep_cpu.elapsed_s)
    row["cpu_cg_iterations"] = int(rep_cpu.iterations)
    row["cpu_cg_residual"] = float(rep_cpu.relative_residual)

    t0 = perf_counter()
    lu = spla.splu(A.tocsc())
    t_factor = perf_counter() - t0
    t0 = perf_counter()
    x_lu = lu.solve(b)
    t_solve = perf_counter() - t0
    row["cpu_splu_factor_s"] = float(t_factor)
    row["cpu_splu_solve_s"] = float(t_solve)
    row["cpu_splu_residual"] = float(np.linalg.norm(A @ x_lu - b) / np.linalg.norm(b))

    x_gpu, rep_gpu = solve_spd(A, b, backend="gpu", tol=tol, maxiter=maxiter)
    row["gpu_cg_s"] = float(rep_gpu.elapsed_s)
    row["gpu_transfer_s"] = float(rep_gpu.transfer_s)
    row["gpu_cg_iterations"] = int(rep_gpu.iterations)
    row["gpu_cg_residual"] = float(rep_gpu.relative_residual)
    row["gpu_success"] = bool(rep_gpu.success)
    row["gpu_message"] = rep_gpu.message

    if rep_gpu.success and rep_cpu.success:
        row["gpu_speedup_vs_cpu_cg"] = float(rep_cpu.elapsed_s / max(rep_gpu.elapsed_s, 1.0e-12))
    return row


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--n-elements", type=int, nargs="*", default=[20, 28, 36, 44])
    parser.add_argument("--eps", type=float, default=1.0e-3)
    parser.add_argument("--tol", type=float, default=1.0e-8)
    parser.add_argument("--maxiter", type=int, default=20000)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    if args.quick:
        args.n_elements = [20]

    # GPU warmup so first-call kernel compilation does not pollute timings.
    A0, b0 = build_system(20, args.degree, args.eps)
    solve_spd(A0, b0, backend="gpu", tol=1.0e-6, maxiter=500)

    rows = []
    for n in args.n_elements:
        row = bench_case(n, args.degree, args.eps, tol=args.tol, maxiter=args.maxiter)
        rows.append(row)
        print(
            f"n={n} unknowns={row['n_unknowns']} nnz={row['nnz']} "
            f"cpu_cg={row['cpu_cg_s']:.3f}s ({row['cpu_cg_iterations']} it) "
            f"gpu_cg={row['gpu_cg_s']:.3f}s+{row['gpu_transfer_s']:.3f}s xfer "
            f"({row['gpu_cg_iterations']} it) "
            f"splu={row['cpu_splu_factor_s']:.3f}+{row['cpu_splu_solve_s']:.4f}s",
            flush=True,
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "gpu_solver_benchmark_3d.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
    keys = sorted(rows[0].keys()) if rows else []
    with (RESULTS / "gpu_solver_benchmark_3d.csv").open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
