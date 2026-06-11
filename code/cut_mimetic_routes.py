"""Prototype BMG^cut and BMF^cut route assembly in one space dimension.

This module is intentionally small.  It records the algebraic distinction
between the same-grid Galerkin cut route and the flux-space cut route using the
same sharply cut interval quadrature.  The theorem-facing identities are exact
matrix identities:

* BMG^cut: H D + D^T H = R^T B R.
* BMF^cut: H0 D + G^T H1 = B01.

The active cut spline space is an assembly space.  A later physical-space
injection can be applied by ordinary Galerkin pullback to every matrix below.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json

import numpy as np
from scipy.interpolate import BSpline

from cut_spline_sbp import (
    SplineBasis1D,
    assemble_cut_matrices,
    make_open_uniform_basis,
    _eval_all,
    _quadrature_panels,
)


@dataclass(frozen=True)
class ScalarMassMatrices1D:
    H: np.ndarray
    H_background: np.ndarray
    left_trace: np.ndarray
    right_trace: np.ndarray


@dataclass(frozen=True)
class CutBMGRoute1D:
    active: np.ndarray
    H: np.ndarray
    Q: np.ndarray
    boundary_form: np.ndarray
    D: np.ndarray
    green_residual: float


@dataclass(frozen=True)
class CutBMFRoute1D:
    active_primal: np.ndarray
    active_flux: np.ndarray
    H0: np.ndarray
    H1: np.ndarray
    G: np.ndarray
    D: np.ndarray
    B01: np.ndarray
    mixed_green_residual: float


@dataclass(frozen=True)
class CutRouteAuditRow:
    p: int
    n_elements: int
    epsilon: float
    left: float
    n_primal_active: int
    n_flux_active: int
    bmg_green_residual: float
    bmf_mixed_green_residual: float
    bmg_bmf_diffusion_difference: float
    bmg_radius_h_over_p2: float
    bmf_wave_radius_h_over_p2: float


# ---------------------------------------------------------------------------
# Basis and quadrature helpers
# ---------------------------------------------------------------------------


def make_derivative_basis(primal: SplineBasis1D) -> SplineBasis1D:
    """Return the degree-(p-1) derivative B-spline basis induced by ``primal``."""
    if primal.degree < 1:
        raise ValueError("The BMF flux route requires primal degree at least one.")
    degree = primal.degree - 1
    knots = primal.knots[1:-1].copy()
    n_basis = len(knots) - degree - 1
    coeffs = np.eye(n_basis)
    basis = [BSpline(knots, coeffs[i], degree, extrapolate=False) for i in range(n_basis)]
    if degree >= 1:
        dbasis = [b.derivative() for b in basis]
    else:
        # Degree-zero flux bases are not differentiated by the current audit.
        dbasis = basis
    if degree == 0:
        greville = 0.5 * (knots[:n_basis] + knots[1 : n_basis + 1])
    else:
        greville = np.array(
            [np.sum(knots[i + 1 : i + degree + 1]) / degree for i in range(n_basis)]
        )
    supports = np.array([(knots[i], knots[i + degree + 1]) for i in range(n_basis)])
    return SplineBasis1D(
        degree=degree,
        n_elements=primal.n_elements,
        knots=knots,
        basis=basis,
        dbasis=dbasis,
        greville=greville,
        supports=supports,
    )


def assemble_scalar_mass(
    basis: SplineBasis1D,
    left: float,
    right: float = 1.0,
    quadrature_order: int | None = None,
) -> ScalarMassMatrices1D:
    """Assemble cut and background mass matrices for an arbitrary spline basis."""
    p = basis.degree
    q = quadrature_order or max(8, p + 5)
    xg, wg = np.polynomial.legendre.leggauss(q)
    n = basis.n_basis
    H = np.zeros((n, n))
    H_background = np.zeros((n, n))

    for lo, hi in zip(*[_quadrature_panels(basis.knots, left, right)[i:] for i in (0, 1)]):
        if hi <= lo:
            continue
        x = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
        w = 0.5 * (hi - lo) * wg
        V = _eval_all(basis.basis, x)
        H += (V * w) @ V.T

    for lo, hi in zip(*[_quadrature_panels(basis.knots, 0.0, 1.0)[i:] for i in (0, 1)]):
        if hi <= lo:
            continue
        x = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
        w = 0.5 * (hi - lo) * wg
        V = _eval_all(basis.basis, x)
        H_background += (V * w) @ V.T

    left_trace = _eval_all(basis.basis, np.array([left])).ravel()
    right_trace = _eval_all(basis.basis, np.array([right])).ravel()
    return ScalarMassMatrices1D(H, H_background, left_trace, right_trace)


def assemble_cross_form(
    test_basis: SplineBasis1D,
    trial_functions: list[BSpline],
    left: float,
    right: float = 1.0,
    quadrature_order: int | None = None,
) -> np.ndarray:
    """Assemble int test_i trial_j over the cut interval."""
    degree = max(test_basis.degree, max(getattr(f, "k", 0) for f in trial_functions))
    q = quadrature_order or max(10, 2 * degree + 5)
    xg, wg = np.polynomial.legendre.leggauss(q)
    out = np.zeros((test_basis.n_basis, len(trial_functions)))
    all_knots = np.unique(np.concatenate([test_basis.knots, *[np.asarray(f.t) for f in trial_functions]]))
    interior = all_knots[(all_knots > left) & (all_knots < right)]
    panels = np.unique(np.concatenate(([left], interior, [right])))
    for lo, hi in zip(panels[:-1], panels[1:]):
        if hi <= lo:
            continue
        x = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
        w = 0.5 * (hi - lo) * wg
        Vt = _eval_all(test_basis.basis, x)
        Vf = _eval_all(trial_functions, x)
        out += (Vt * w) @ Vf.T
    return out


def _active_from_mass(H: np.ndarray, H_background: np.ndarray, active_tol: float = 1.0e-25) -> np.ndarray:
    gamma = np.zeros(H.shape[0])
    bg = np.diag(H_background)
    mask = bg > 0.0
    gamma[mask] = np.diag(H)[mask] / bg[mask]
    return np.where(gamma > active_tol)[0]


def _sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


# ---------------------------------------------------------------------------
# Route construction
# ---------------------------------------------------------------------------


def build_bmg_cut_route_1d(
    basis: SplineBasis1D,
    left: float,
    right: float = 1.0,
    active_tol: float = 1.0e-25,
) -> CutBMGRoute1D:
    """Build the same-grid Galerkin BMG^cut route on the active cut space."""
    mats = assemble_cut_matrices(basis, left=left, right=right)
    active = _active_from_mass(mats.H, mats.H_background, active_tol=active_tol)
    idx = np.ix_(active, active)
    H = _sym(mats.H[idx])
    Q = mats.Q[idx]
    B = mats.boundary_form[idx]
    D = np.linalg.solve(H, Q)
    residual = float(np.linalg.norm(H @ D + D.T @ H - B, ord=np.inf))
    return CutBMGRoute1D(active=active, H=H, Q=Q, boundary_form=B, D=D, green_residual=residual)


def build_bmf_cut_route_1d(
    basis: SplineBasis1D,
    left: float,
    right: float = 1.0,
    active_tol: float = 1.0e-25,
) -> CutBMFRoute1D:
    """Build the staggered flux-space BMF^cut route on active primal and flux spaces."""
    flux_basis = make_derivative_basis(basis)
    m0 = assemble_scalar_mass(basis, left=left, right=right)
    m1 = assemble_scalar_mass(flux_basis, left=left, right=right)
    active0 = _active_from_mass(m0.H, m0.H_background, active_tol=active_tol)
    active1 = _active_from_mass(m1.H, m1.H_background, active_tol=active_tol)

    H0 = _sym(m0.H[np.ix_(active0, active0)])
    H1 = _sym(m1.H[np.ix_(active1, active1)])
    cross_grad_full = assemble_cross_form(flux_basis, basis.dbasis, left=left, right=right)
    C = cross_grad_full[np.ix_(active1, active0)]
    G = np.linalg.solve(H1, C)

    left0 = m0.left_trace[active0]
    right0 = m0.right_trace[active0]
    left1 = m1.left_trace[active1]
    right1 = m1.right_trace[active1]
    B01 = np.outer(right0, right1) - np.outer(left0, left1)
    D = np.linalg.solve(H0, B01 - G.T @ H1)
    residual = float(np.linalg.norm(H0 @ D + G.T @ H1 - B01, ord=np.inf))
    return CutBMFRoute1D(
        active_primal=active0,
        active_flux=active1,
        H0=H0,
        H1=H1,
        G=G,
        D=D,
        B01=B01,
        mixed_green_residual=residual,
    )


def route_audit_row(
    n_elements: int = 16,
    degree: int = 3,
    epsilon: float = 0.1,
) -> CutRouteAuditRow:
    """Return one theorem-facing audit row for BMG^cut and BMF^cut."""
    basis = make_open_uniform_basis(n_elements=n_elements, degree=degree)
    left = epsilon * basis.h
    bmg = build_bmg_cut_route_1d(basis, left=left)
    bmf = build_bmf_cut_route_1d(basis, left=left)

    flux_basis = make_derivative_basis(basis)
    # The active sets agree on the primal side for the endpoint cuts used here.
    if not np.array_equal(bmg.active, bmf.active_primal):
        raise RuntimeError("Primal active sets differ in the route audit.")
    # The first-derivative projection agrees in this one-dimensional prototype.
    # The routes nevertheless remain distinct because BMF carries a typed
    # flux space and its compatible diffusion is D G, not D_BMG squared.
    diff = float(np.linalg.norm((bmf.D @ bmf.G) - (bmg.D @ bmg.D), ord=np.inf))

    eig_bmg = np.linalg.eigvals(np.linalg.solve(bmg.H, bmg.Q))
    bmg_radius = float(np.max(np.abs(eig_bmg)))
    # Staggered acoustic block [[0,D],[G,0]] in mass coordinates.
    top = np.concatenate([np.zeros((bmf.H0.shape[0], bmf.H0.shape[0])), bmf.D], axis=1)
    bottom = np.concatenate([bmf.G, np.zeros((bmf.H1.shape[0], bmf.H1.shape[0]))], axis=1)
    wave = np.concatenate([top, bottom], axis=0)
    wave_radius = float(np.max(np.abs(np.linalg.eigvals(wave))))

    scale = basis.h / float((degree + 1) ** 2)
    return CutRouteAuditRow(
        p=degree,
        n_elements=n_elements,
        epsilon=epsilon,
        left=left,
        n_primal_active=int(len(bmg.active)),
        n_flux_active=int(len(bmf.active_flux)),
        bmg_green_residual=bmg.green_residual,
        bmf_mixed_green_residual=bmf.mixed_green_residual,
        bmg_bmf_diffusion_difference=diff,
        bmg_radius_h_over_p2=bmg_radius * scale,
        bmf_wave_radius_h_over_p2=wave_radius * scale,
    )


def write_route_audit(rows: list[CutRouteAuditRow], out_dir: Path) -> dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "cut_mimetic_routes.json"
    csv_path = out_dir / "cut_mimetic_routes.csv"
    summary_path = out_dir / "cut_mimetic_routes_summary.json"
    data = [asdict(row) for row in rows]
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    if data:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)
    summary = {
        "rows": len(rows),
        "max_bmg_green_residual": max((r.bmg_green_residual for r in rows), default=0.0),
        "max_bmf_mixed_green_residual": max((r.bmf_mixed_green_residual for r in rows), default=0.0),
        "min_branch_diffusion_difference": min((r.bmg_bmf_diffusion_difference for r in rows), default=0.0),
        "max_bmg_radius_h_over_p2": max((r.bmg_radius_h_over_p2 for r in rows), default=0.0),
        "max_bmf_wave_radius_h_over_p2": max((r.bmf_wave_radius_h_over_p2 for r in rows), default=0.0),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
