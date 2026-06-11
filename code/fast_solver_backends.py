"""Reusable CPU/GPU linear-solver backends for cut-cell experiments.

The manuscript experiments mostly assemble modest matrices so that algebraic
identities, convergence rates, and small-cell constants can be audited exactly.
Large production runs need a different habit: keep the Section 6 matrices in
sparse form, measure Krylov iterations and wall-clock time, and use the GPU only
when the matrix is large enough to amortize host-device transfers.

This module keeps that policy explicit.  CPU SciPy is the default and is the
right choice for the current paper-scale matrices.  A CuPy backend is optional:
if CuPy is installed, ``solve_gpu_cg`` runs a Jacobi-preconditioned CG loop on an
NVIDIA GPU; otherwise it returns a clean report explaining that the backend is
not available.  No paper reproducer depends on CuPy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

Backend = Literal["cpu", "gpu"]
Preconditioner = Literal["none", "jacobi"]


@dataclass(frozen=True)
class SolverReport:
    """Summary returned by every solver backend."""

    backend: str
    method: str
    preconditioner: str
    success: bool
    iterations: int
    elapsed_s: float
    relative_residual: float
    message: str = ""
    transfer_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def as_csr(A: np.ndarray | sp.spmatrix) -> sp.csr_matrix:
    """Return ``A`` as a CSR matrix without changing its numerical values."""

    if sp.issparse(A):
        return A.tocsr()
    return sp.csr_matrix(np.asarray(A, dtype=float))


def csr_memory_bytes(A: np.ndarray | sp.spmatrix) -> int:
    """Estimated storage of a CSR matrix in bytes."""

    Acsr = as_csr(A)
    return int(Acsr.data.nbytes + Acsr.indices.nbytes + Acsr.indptr.nbytes)


def relative_residual(A: np.ndarray | sp.spmatrix, x: np.ndarray, b: np.ndarray) -> float:
    """Compute ``||Ax-b|| / ||b||`` with a safe zero-RHS convention."""

    Acsr = as_csr(A)
    b = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(b))
    if denom == 0.0:
        denom = 1.0
    return float(np.linalg.norm(Acsr @ x - b) / denom)


def jacobi_preconditioner(A: np.ndarray | sp.spmatrix) -> spla.LinearOperator:
    """Diagonal preconditioner for SPD systems.

    Zero diagonal entries are left unpreconditioned rather than raising.  This is
    deliberate for stress sweeps: a singular or nearly singular active-space
    matrix should fail in the Krylov report, not in setup.
    """

    Acsr = as_csr(A)
    d = Acsr.diagonal().astype(float, copy=True)
    inv = np.zeros_like(d)
    mask = np.abs(d) > 0.0
    inv[mask] = 1.0 / d[mask]
    return spla.LinearOperator(Acsr.shape, matvec=lambda x: inv * x, dtype=float)


def solve_cpu_cg(
    A: np.ndarray | sp.spmatrix,
    b: np.ndarray,
    *,
    tol: float = 1.0e-8,
    maxiter: int = 5000,
    preconditioner: Preconditioner = "jacobi",
) -> tuple[np.ndarray, SolverReport]:
    """Solve an SPD system with SciPy CG and return a counted report."""

    Acsr = as_csr(A)
    b = np.asarray(b, dtype=float)
    count = 0

    def callback(_: np.ndarray) -> None:
        nonlocal count
        count += 1

    M = jacobi_preconditioner(Acsr) if preconditioner == "jacobi" else None
    tic = perf_counter()
    try:
        x, info = spla.cg(Acsr, b, rtol=tol, atol=0.0, maxiter=maxiter, M=M, callback=callback)
    except TypeError:  # SciPy < 1.11 used ``tol``.
        x, info = spla.cg(Acsr, b, tol=tol, maxiter=maxiter, M=M, callback=callback)
    elapsed = perf_counter() - tic
    rel = relative_residual(Acsr, x, b)
    success = bool(info == 0 and np.isfinite(rel) and rel <= max(10.0 * tol, 1.0e-14))
    if info == 0:
        message = "converged"
    elif info > 0:
        message = f"not converged after {info} iterations"
    else:
        message = f"illegal input or breakdown (info={info})"
    return x, SolverReport(
        backend="cpu",
        method="scipy.sparse.linalg.cg",
        preconditioner=preconditioner,
        success=success,
        iterations=int(count if count else max(info, 0)),
        elapsed_s=float(elapsed),
        relative_residual=float(rel),
        message=message,
    )


def solve_cpu_gmres(
    A: np.ndarray | sp.spmatrix,
    b: np.ndarray,
    *,
    tol: float = 1.0e-8,
    maxiter: int = 5000,
    restart: int = 80,
    preconditioner: Preconditioner = "jacobi",
) -> tuple[np.ndarray, SolverReport]:
    """Solve a nonsymmetric system with restarted SciPy GMRES."""

    Acsr = as_csr(A)
    b = np.asarray(b, dtype=float)
    count = 0

    def callback(_: np.ndarray) -> None:
        nonlocal count
        count += 1

    M = jacobi_preconditioner(Acsr) if preconditioner == "jacobi" else None
    tic = perf_counter()
    try:
        x, info = spla.gmres(
            Acsr,
            b,
            rtol=tol,
            atol=0.0,
            restart=restart,
            maxiter=maxiter,
            M=M,
            callback=callback,
            callback_type="pr_norm",
        )
    except TypeError:
        x, info = spla.gmres(
            Acsr,
            b,
            tol=tol,
            restart=restart,
            maxiter=maxiter,
            M=M,
            callback=callback,
        )
    elapsed = perf_counter() - tic
    rel = relative_residual(Acsr, x, b)
    success = bool(info == 0 and np.isfinite(rel) and rel <= max(10.0 * tol, 1.0e-14))
    message = "converged" if info == 0 else f"not converged or breakdown (info={info})"
    return x, SolverReport(
        backend="cpu",
        method="scipy.sparse.linalg.gmres",
        preconditioner=preconditioner,
        success=success,
        iterations=int(count),
        elapsed_s=float(elapsed),
        relative_residual=float(rel),
        message=message,
    )


def solve_gpu_cg(
    A: np.ndarray | sp.spmatrix,
    b: np.ndarray,
    *,
    tol: float = 1.0e-8,
    maxiter: int = 5000,
    preconditioner: Preconditioner = "jacobi",
) -> tuple[np.ndarray | None, SolverReport]:
    """Optional CuPy Jacobi-CG backend for large sparse SPD systems.

    The implementation uses a transparent CG loop instead of relying on a CuPy
    callback API.  It counts iterations, synchronizes before reading timings, and
    reports host-device transfer time separately.  If CuPy is unavailable, the
    returned solution is ``None`` and ``success`` is false.
    """

    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpsp
    except Exception as exc:  # pragma: no cover - depends on local GPU stack
        return None, SolverReport(
            backend="gpu",
            method="cupy-custom-cg",
            preconditioner=preconditioner,
            success=False,
            iterations=0,
            elapsed_s=0.0,
            relative_residual=float("inf"),
            message=f"CuPy backend unavailable: {exc}",
        )

    Acsr = as_csr(A)
    b_np = np.asarray(b, dtype=float)
    transfer_tic = perf_counter()
    A_gpu = cpsp.csr_matrix(Acsr)
    b_gpu = cp.asarray(b_np)
    cp.cuda.Stream.null.synchronize()
    transfer_s = perf_counter() - transfer_tic

    tic = perf_counter()
    x = cp.zeros_like(b_gpu)
    r = b_gpu - A_gpu @ x
    if preconditioner == "jacobi":
        d = A_gpu.diagonal()
        z = cp.where(cp.abs(d) > 0.0, r / d, r)
    else:
        z = r.copy()
    p = z.copy()
    rz_old = cp.vdot(r, z).real
    b_norm = cp.linalg.norm(b_gpu)
    if float(b_norm.get()) == 0.0:
        b_norm = cp.asarray(1.0)
    success = False
    message = "maximum iterations reached"
    iterations = 0
    for k in range(1, maxiter + 1):
        Ap = A_gpu @ p
        denom = cp.vdot(p, Ap).real
        denom_host = float(denom.get())
        if denom_host <= 0.0 or not np.isfinite(denom_host):
            message = "CG breakdown: non-positive curvature"
            iterations = k
            break
        alpha = rz_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rel_gpu = cp.linalg.norm(r) / b_norm
        rel_host = float(rel_gpu.get())
        iterations = k
        if rel_host <= tol:
            success = True
            message = "converged"
            break
        if preconditioner == "jacobi":
            z = cp.where(cp.abs(d) > 0.0, r / d, r)
        else:
            z = r.copy()
        rz_new = cp.vdot(r, z).real
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new
    cp.cuda.Stream.null.synchronize()
    elapsed = perf_counter() - tic
    x_np = cp.asnumpy(x)
    rel = relative_residual(Acsr, x_np, b_np)
    return x_np, SolverReport(
        backend="gpu",
        method="cupy-custom-cg",
        preconditioner=preconditioner,
        success=bool(success and rel <= max(10.0 * tol, 1.0e-14)),
        iterations=int(iterations),
        elapsed_s=float(elapsed),
        relative_residual=float(rel),
        message=message,
        transfer_s=float(transfer_s),
    )


def solve_spd(
    A: np.ndarray | sp.spmatrix,
    b: np.ndarray,
    *,
    backend: Literal["cpu", "gpu", "auto"] = "cpu",
    tol: float = 1.0e-8,
    maxiter: int = 5000,
    preconditioner: Preconditioner = "jacobi",
) -> tuple[np.ndarray | None, SolverReport]:
    """Dispatch to the requested SPD solver backend."""

    if backend == "gpu":
        return solve_gpu_cg(A, b, tol=tol, maxiter=maxiter, preconditioner=preconditioner)
    if backend == "auto":
        x, report = solve_gpu_cg(A, b, tol=tol, maxiter=maxiter, preconditioner=preconditioner)
        if report.success:
            return x, report
        x_cpu, report_cpu = solve_cpu_cg(A, b, tol=tol, maxiter=maxiter, preconditioner=preconditioner)
        message = report_cpu.message
        if report.message:
            message = f"GPU skipped/fell back: {report.message}; CPU: {message}"
        report_cpu = SolverReport(
            backend=report_cpu.backend,
            method=report_cpu.method,
            preconditioner=report_cpu.preconditioner,
            success=report_cpu.success,
            iterations=report_cpu.iterations,
            elapsed_s=report_cpu.elapsed_s,
            relative_residual=report_cpu.relative_residual,
            message=message,
            transfer_s=report_cpu.transfer_s,
        )
        return x_cpu, report_cpu
    return solve_cpu_cg(A, b, tol=tol, maxiter=maxiter, preconditioner=preconditioner)
