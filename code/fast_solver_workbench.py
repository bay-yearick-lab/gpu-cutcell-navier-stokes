"""Fast-solver stress workbench for Section 6 cut-cell matrices.

Solver harness for Section 6.2: once the small cell has been denied an
independent coordinate, the linear solve behaves like a background-grid solve
in wall-clock time and Krylov iterations.

Examples
--------
Quick smoke run (writes ``fast_solver_stress_quick.*``)::

    python code/fast_solver_workbench.py --quick

CPU production sweep (``BSPLINE_JOBS`` sets the worker count)::

    python code/fast_solver_workbench.py --production --backend cpu

Optional GPU run on one large case if CuPy is installed::

    python code/fast_solver_workbench.py --backend gpu --degree 4 --n-elements 48 64 --eps 1e-3 1e-5

Outputs
-------
``code/results/fast_solver_stress.{csv,json}``,
``code/results/fast_solver_stress_summary.json``, and
``figures/fig_s6_solver_stress.{pdf,png}``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Sequence

for _thread_var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import eigvalsh

import plot_style
from cut_spline_sbp import make_open_uniform_basis
from fast_solver_backends import csr_memory_bytes, solve_spd
from oblique_cut_sbp import (
    HalfPlaneCut,
    assemble_oblique_cut_matrices,
    polynomial_preserving_oblique_extension,
    pullback_oblique_2d,
    restrict_oblique_2d,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "code" / "results"
FIGURES = ROOT / "figures"


@dataclass(frozen=True)
class SolverCase:
    degree: int
    n_elements: int
    epsilon: float
    backend: str
    tol: float
    maxiter: int
    compute_condition: bool


@dataclass(frozen=True)
class SolverRow:
    degree: int
    n_elements: int
    epsilon: float
    volume_ratio: float
    n_active: int
    n_reduced: int
    nnz: int
    csr_memory_mb: float
    assembly_s: float
    solve_s: float
    transfer_s: float
    iterations: int
    success: bool
    relative_residual: float
    condition_proxy: float | None
    backend: str
    method: str
    preconditioner: str
    message: str


def _quadrature_order(degree: int) -> int:
    return max(24, 2 * degree + 14)


def _sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def _condition_proxy(A: sp.csr_matrix) -> float:
    """Return a spectral condition proxy for an SPD matrix.

    Dense eigenvalues are cheaper and more reliable for the small paper-scale
    matrices.  For larger workbench matrices, use ``eigsh`` at both ends.
    """

    n = A.shape[0]
    if n <= 900:
        ev = eigvalsh(A.toarray())
        lo = float(ev[0])
        return float(ev[-1] / lo) if lo > 0.0 else float("inf")
    try:
        lam_max = float(spla.eigsh(A, k=1, which="LA", return_eigenvectors=False, tol=1.0e-4)[0])
        lam_min = float(spla.eigsh(A, k=1, which="SA", return_eigenvectors=False, tol=1.0e-4)[0])
        return float(lam_max / lam_min) if lam_min > 0.0 else float("inf")
    except Exception:
        return float("nan")


def run_case(case: SolverCase) -> SolverRow:
    tic = perf_counter()
    basis_x = make_open_uniform_basis(n_elements=case.n_elements, degree=case.degree)
    basis_y = make_open_uniform_basis(n_elements=case.n_elements, degree=case.degree)
    h = basis_x.h
    cut = HalfPlaneCut.from_angle(0.0, h * (1.0 - case.epsilon))
    matrices = assemble_oblique_cut_matrices(
        basis_x,
        basis_y,
        cut,
        quadrature_order=_quadrature_order(case.degree),
        capture_volume_quadrature=False,
    )
    agg = polynomial_preserving_oblique_extension(
        basis_x,
        basis_y,
        matrices,
        support_threshold=0.05,
        preserve_degree=case.degree,
        active_tol=1.0e-100,
    )
    phys = pullback_oblique_2d(matrices, agg.E)
    # The active matrix is built only to record active dimensions and to make
    # this driver mirror the Section 6.2 audit.  Solver timing is on the
    # aggregated SPD operator.
    active = restrict_oblique_2d(matrices, agg.active)
    _ = active.H.shape[0]
    A = sp.csr_matrix(_sym(phys.K + phys.H))
    assembly_s = perf_counter() - tic
    rng = np.random.default_rng(12345 + 17 * case.n_elements + case.degree)
    b = rng.standard_normal(A.shape[0])
    _x, report = solve_spd(
        A,
        b,
        backend=case.backend,  # type: ignore[arg-type]
        tol=case.tol,
        maxiter=case.maxiter,
        preconditioner="jacobi",
    )
    cond = _condition_proxy(A) if case.compute_condition else None
    return SolverRow(
        degree=int(case.degree),
        n_elements=int(case.n_elements),
        epsilon=float(case.epsilon),
        volume_ratio=float(1.0 / case.epsilon),
        n_active=int(len(agg.active)),
        n_reduced=int(agg.E.shape[1]),
        nnz=int(A.nnz),
        csr_memory_mb=float(csr_memory_bytes(A) / 1.0e6),
        assembly_s=float(assembly_s),
        solve_s=float(report.elapsed_s),
        transfer_s=float(report.transfer_s),
        iterations=int(report.iterations),
        success=bool(report.success),
        relative_residual=float(report.relative_residual),
        condition_proxy=None if cond is None else float(cond),
        backend=report.backend,
        method=report.method,
        preconditioner=report.preconditioner,
        message=report.message,
    )


def _run_cases(cases: list[SolverCase], jobs: int) -> list[SolverRow]:
    if not cases:
        return []
    # One GPU should not receive several independent Python workers.  CPU cases
    # are embarrassingly parallel because assembly and solves are independent.
    use_jobs = 1 if any(c.backend == "gpu" for c in cases) else max(1, int(jobs))
    rows: list[SolverRow] = []
    if use_jobs == 1:
        for c in cases:
            rows.append(run_case(c))
        return rows
    with ProcessPoolExecutor(max_workers=use_jobs) as ex:
        futs = {ex.submit(run_case, c): c for c in cases}
        for fut in as_completed(futs):
            rows.append(fut.result())
    rows.sort(key=lambda r: (r.degree, r.n_elements, r.epsilon))
    return rows


def _write_outputs(rows: list[SolverRow], prefix: str) -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = [asdict(r) for r in rows]
    summary = {
        "n_cases": len(rows),
        "backend_values": sorted({r.backend for r in rows}),
        "degree_values": sorted({r.degree for r in rows}),
        "n_elements_values": sorted({r.n_elements for r in rows}),
        "volume_ratio_range": [min((r.volume_ratio for r in rows), default=None), max((r.volume_ratio for r in rows), default=None)],
        "all_converged": all(r.success for r in rows),
        "iteration_range": [min((r.iterations for r in rows), default=None), max((r.iterations for r in rows), default=None)],
        "solve_time_s_range": [min((r.solve_s for r in rows), default=None), max((r.solve_s for r in rows), default=None)],
        "assembly_time_s_range": [min((r.assembly_s for r in rows), default=None), max((r.assembly_s for r in rows), default=None)],
        "max_relative_residual": max((r.relative_residual for r in rows), default=None),
        "max_csr_memory_mb": max((r.csr_memory_mb for r in rows), default=None),
    }
    (RESULTS / f"{prefix}.json").write_text(json.dumps({"rows": data, "summary": summary}, indent=2), encoding="utf-8")
    (RESULTS / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (RESULTS / f"{prefix}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else list(SolverRow.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(data)
    _write_figure(rows)
    return summary


def _write_figure(rows: list[SolverRow]) -> None:
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=plot_style.WIDE_SIZE)
    ns = sorted({r.n_elements for r in rows})
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for idx, n in enumerate(ns):
        group = sorted([r for r in rows if r.n_elements == n], key=lambda r: r.volume_ratio)
        x = np.array([r.volume_ratio for r in group], dtype=float)
        it = np.array([r.iterations for r in group], dtype=float)
        t = np.array([r.solve_s for r in group], dtype=float)
        label = rf"$n={n}$"
        axes[0].semilogx(x, it, marker=markers[idx % len(markers)], lw=1.4, ms=4.8, label=label)
        axes[1].loglog(x, np.maximum(t, 1.0e-9), marker=markers[idx % len(markers)], lw=1.4, ms=4.8, label=label)
    axes[0].set_xlabel("full / small cell volume ratio")
    axes[0].set_ylabel("CG iterations")
    axes[0].text(0.02, 0.98, "(a)", transform=axes[0].transAxes, ha="left", va="top")
    axes[1].set_xlabel("full / small cell volume ratio")
    axes[1].set_ylabel("solve wall-clock time (s)")
    axes[1].text(0.02, 0.98, "(b)", transform=axes[1].transAxes, ha="left", va="top")
    for ax in axes:
        plot_style.style_axes(ax)
    axes[0].legend(loc="best", fontsize=7.5, frameon=False)
    fig.tight_layout()
    plot_style.save_pdf_png(fig, FIGURES / "fig_s6_solver_stress.pdf")
    plt.close(fig)


def _parse_float_list(values: Sequence[str]) -> list[float]:
    out: list[float] = []
    for value in values:
        for piece in str(value).split(","):
            piece = piece.strip()
            if piece:
                out.append(float(piece))
    return out


def _parse_int_list(values: Sequence[str]) -> list[int]:
    return [int(round(v)) for v in _parse_float_list(values)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="one tiny case for CI/smoke testing")
    parser.add_argument("--production", action="store_true", help="use the recommended local production sweep")
    parser.add_argument("--backend", choices=("cpu", "gpu", "auto"), default="cpu")
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--n-elements", nargs="*", default=None, help="space or comma separated element counts")
    parser.add_argument("--eps", nargs="*", default=None, help="space or comma separated sliver widths")
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("BSPLINE_JOBS", "0") or 0))
    parser.add_argument("--tol", type=float, default=1.0e-8)
    parser.add_argument("--maxiter", type=int, default=5000)
    parser.add_argument("--condition", action="store_true", help="also estimate condition numbers")
    parser.add_argument("--out-prefix", default="fast_solver_stress")
    args = parser.parse_args(argv)

    if args.quick:
        n_elements = [5]
        eps_values = [1.0e-1, 1.0e-2, 1.0e-3]
        compute_condition = True
        if args.out_prefix == "fast_solver_stress":
            args.out_prefix = "fast_solver_stress_quick"
    elif args.production:
        n_elements = [8, 12, 16, 24, 32]
        eps_values = [1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5]
        compute_condition = args.condition
    else:
        n_elements = _parse_int_list(args.n_elements) if args.n_elements else [8, 12, 16]
        eps_values = _parse_float_list(args.eps) if args.eps else [1.0e-1, 1.0e-3, 1.0e-5]
        compute_condition = args.condition

    if any(e <= 0.0 or not math.isfinite(e) for e in eps_values):
        raise ValueError("all eps values must be positive finite numbers")
    jobs = args.jobs if args.jobs and args.jobs > 0 else min(20, max(1, (os.cpu_count() or 2) - 1))
    cases = [
        SolverCase(
            degree=args.degree,
            n_elements=n,
            epsilon=e,
            backend=args.backend,
            tol=args.tol,
            maxiter=args.maxiter,
            compute_condition=compute_condition,
        )
        for n in n_elements
        for e in eps_values
    ]
    tic = perf_counter()
    rows = _run_cases(cases, jobs=jobs)
    summary = _write_outputs(rows, args.out_prefix)
    elapsed = perf_counter() - tic
    print(json.dumps(summary, indent=2))
    print(f"[fast_solver_workbench] wrote code/results/{args.out_prefix}.* and figures/fig_s6_solver_stress.*")
    print(f"[fast_solver_workbench] total wall time {elapsed:.2f}s using jobs={jobs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
