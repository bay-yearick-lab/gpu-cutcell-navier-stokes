"""Two-dimensional BMG^cut and BMF^cut formulation audits on direct oblique cuts.

The active cut spline space is treated as an assembly space.  This diagnostic
builds the two cut-cell formulation objects that the theorem spine distinguishes:

* BMG^cut: same-space Galerkin derivative on scalar coefficients.
* BMF^cut: typed scalar-to-flux gradient and metric-adjoint divergence.

The module deliberately reports identities, not convergence rates.  It checks
that the cut-domain Green identities, the mixed scalar/flux Green identities,
the metric-adjoint skew-symmetric energy identity, and pressure-work cancellation are
roundoff-level matrix facts on oblique embedded boundaries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
from math import pi

import numpy as np

from cut_spline_sbp import SplineBasis1D, make_open_uniform_basis, _eval_all
from cut_mimetic_routes import make_derivative_basis
from oblique_cut_sbp import (
    HalfPlaneCut,
    _cell_polygon,
    _edge_quadrature,
    _tensor_values,
    assemble_oblique_cut_matrices,
    clip_polygon_halfplane,
    oblique_active_sets,
    polygon_area,
    tensor_background_mass,
)


@dataclass(frozen=True)
class CutBMGRoute2D:
    active: np.ndarray
    H: np.ndarray
    Qx: np.ndarray
    Qy: np.ndarray
    Sx: np.ndarray
    Sy: np.ndarray
    Dx: np.ndarray
    Dy: np.ndarray
    green_residual_x: float
    green_residual_y: float


@dataclass(frozen=True)
class CutBMFDirection2D:
    direction: str
    active_scalar: np.ndarray
    active_flux: np.ndarray
    H0: np.ndarray
    H1: np.ndarray
    C: np.ndarray
    B01_boundary: np.ndarray
    B01_volume: np.ndarray
    G: np.ndarray
    D: np.ndarray
    mixed_green_residual: float
    boundary_volume_residual: float


@dataclass(frozen=True)
class CutBMFRoute2D:
    x: CutBMFDirection2D
    y: CutBMFDirection2D
    pressure_work_residual: float
    skew_energy_residual: float
    bmf_wave_radius_h_over_p2: float


@dataclass(frozen=True)
class CutRouteAuditRow2D:
    family: str
    p: int
    n_elements: int
    angle: float
    offset: float
    n_scalar_active: int
    n_flux_x_active: int
    n_flux_y_active: int
    bmg_green_residual_x: float
    bmg_green_residual_y: float
    bmf_mixed_green_residual_x: float
    bmf_mixed_green_residual_y: float
    bmf_boundary_volume_residual_x: float
    bmf_boundary_volume_residual_y: float
    pressure_work_residual: float
    skew_energy_residual: float
    bmg_radius_h_over_p2: float
    bmf_wave_radius_h_over_p2: float


# ---------------------------------------------------------------------------
# Tensor-product basis evaluation helpers
# ---------------------------------------------------------------------------


def _tensor_eval(
    basis_x: SplineBasis1D,
    basis_y: SplineBasis1D,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return tensor basis values and derivatives with x as the fast index."""
    vx = _eval_all(basis_x.basis, x)
    vy = _eval_all(basis_y.basis, y)
    dx = _eval_all(basis_x.dbasis, x)
    dy = _eval_all(basis_y.dbasis, y)
    # y-major rows, x-fast columns: row = iy * nx + ix.
    values = (vy[:, None, :] * vx[None, :, :]).reshape(
        basis_y.n_basis * basis_x.n_basis, x.size
    )
    grad_x = (vy[:, None, :] * dx[None, :, :]).reshape(
        basis_y.n_basis * basis_x.n_basis, x.size
    )
    grad_y = (dy[:, None, :] * vx[None, :, :]).reshape(
        basis_y.n_basis * basis_x.n_basis, x.size
    )
    return values, grad_x, grad_y


def _mass_from_volume(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return (values * weights) @ values.T


def _active_from_mass(
    H: np.ndarray,
    H_background: np.ndarray,
    active_tol: float = 1.0e-22,
) -> np.ndarray:
    bg = np.diag(H_background)
    physical = np.diag(H)
    gamma = np.zeros_like(physical)
    mask = bg > 0.0
    gamma[mask] = physical[mask] / bg[mask]
    return np.where(gamma > active_tol)[0]


def _sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


# ---------------------------------------------------------------------------
# BMG route
# ---------------------------------------------------------------------------


def build_bmg_cut_route_2d(
    basis_x: SplineBasis1D,
    basis_y: SplineBasis1D,
    cut: HalfPlaneCut,
    active_tol: float = 1.0e-22,
) -> CutBMGRoute2D:
    """Build the same-space Galerkin BMG^cut route on an oblique cut."""
    matrices = assemble_oblique_cut_matrices(basis_x, basis_y, cut)
    sets = oblique_active_sets(matrices, active_tol=active_tol)
    active = sets.active
    idx = np.ix_(active, active)
    H = _sym(matrices.H[idx])
    Qx = matrices.Qx[idx]
    Qy = matrices.Qy[idx]
    Sx = matrices.Sx[idx]
    Sy = matrices.Sy[idx]
    Dx = np.linalg.solve(H, Qx)
    Dy = np.linalg.solve(H, Qy)
    rx = float(np.linalg.norm(H @ Dx + Dx.T @ H - Sx, ord=np.inf))
    ry = float(np.linalg.norm(H @ Dy + Dy.T @ H - Sy, ord=np.inf))
    return CutBMGRoute2D(active, H, Qx, Qy, Sx, Sy, Dx, Dy, rx, ry)


# ---------------------------------------------------------------------------
# BMF route
# ---------------------------------------------------------------------------


def _mixed_boundary_forms_oblique(
    scalar_x: SplineBasis1D,
    scalar_y: SplineBasis1D,
    flux_x: SplineBasis1D,
    flux_y: SplineBasis1D,
    cut: HalfPlaneCut,
    direction_index: int,
    order: int,
) -> np.ndarray:
    """Assemble int_boundary n_i Phi_a Psi_b ds for an oblique half-plane cut."""
    n_scalar = scalar_x.n_basis * scalar_y.n_basis
    n_flux = flux_x.n_basis * flux_y.n_basis
    out = np.zeros((n_scalar, n_flux))

    def is_physical_boundary_edge(a: np.ndarray, b: np.ndarray) -> bool:
        mid = 0.5 * (a + b)
        on_cut = abs(float(np.dot(cut.normal, mid) - cut.offset)) <= 1.0e-10
        on_square = (
            abs(mid[0]) <= 1.0e-12
            or abs(mid[0] - 1.0) <= 1.0e-12
            or abs(mid[1]) <= 1.0e-12
            or abs(mid[1] - 1.0) <= 1.0e-12
        )
        return bool(on_cut or on_square)

    n_elements = scalar_x.n_elements
    for iy in range(n_elements):
        for ix in range(n_elements):
            cell = _cell_polygon(ix, iy, n_elements)
            clipped = clip_polygon_halfplane(cell, cut)
            if len(clipped) < 3 or abs(polygon_area(clipped)) <= 1.0e-16:
                continue
            if polygon_area(clipped) < 0.0:
                clipped = clipped[::-1]
            for k in range(len(clipped)):
                a = clipped[k]
                b = clipped[(k + 1) % len(clipped)]
                if not is_physical_boundary_edge(a, b):
                    continue
                xq, yq, wq, normal = _edge_quadrature(a, b, order)
                if np.allclose(normal, 0.0):
                    continue
                phi, _phix, _phiy = _tensor_eval(scalar_x, scalar_y, xq, yq)
                psi, _psix, _psiy = _tensor_eval(flux_x, flux_y, xq, yq)
                out += normal[direction_index] * (phi * wq) @ psi.T
    return out


def _build_bmf_direction_2d(
    scalar_x: SplineBasis1D,
    scalar_y: SplineBasis1D,
    flux_x: SplineBasis1D,
    flux_y: SplineBasis1D,
    matrices,
    cut: HalfPlaneCut,
    active_scalar: np.ndarray,
    direction: str,
    active_tol: float = 1.0e-22,
) -> CutBMFDirection2D:
    if matrices.volume_quadrature is None:
        raise ValueError("BMF assembly requires captured volume quadrature.")
    q = matrices.volume_quadrature
    scalar_values = q.basis_values
    scalar_grad = q.basis_grad_x if direction == "x" else q.basis_grad_y
    if scalar_grad is None:
        raise ValueError("Volume quadrature gradients are missing.")
    flux_values, flux_grad_x, flux_grad_y = _tensor_eval(flux_x, flux_y, q.x, q.y)
    flux_grad = flux_grad_x if direction == "x" else flux_grad_y

    H0_full = matrices.H
    H1_full = _mass_from_volume(flux_values, q.weights)
    H1_bg = tensor_background_mass(flux_x, flux_y)
    active_flux = _active_from_mass(H1_full, H1_bg, active_tol=active_tol)

    H0 = _sym(H0_full[np.ix_(active_scalar, active_scalar)])
    H1 = _sym(H1_full[np.ix_(active_flux, active_flux)])
    C_full = (flux_values * q.weights) @ scalar_grad.T
    C = C_full[np.ix_(active_flux, active_scalar)]
    C_dual_full = (scalar_values * q.weights) @ flux_grad.T
    B01_volume_full = C_full.T + C_dual_full
    direction_index = 0 if direction == "x" else 1
    order = max(10, scalar_x.degree + scalar_y.degree + 5)
    B01_boundary_full = _mixed_boundary_forms_oblique(
        scalar_x, scalar_y, flux_x, flux_y, cut, direction_index, order
    )
    B01_volume = B01_volume_full[np.ix_(active_scalar, active_flux)]
    B01_boundary = B01_boundary_full[np.ix_(active_scalar, active_flux)]
    G = np.linalg.solve(H1, C)
    D = np.linalg.solve(H0, B01_boundary - G.T @ H1)
    mixed_residual = float(np.linalg.norm(H0 @ D + G.T @ H1 - B01_boundary, ord=np.inf))
    boundary_volume_residual = float(np.linalg.norm(B01_boundary - B01_volume, ord=np.inf))
    return CutBMFDirection2D(
        direction=direction,
        active_scalar=active_scalar,
        active_flux=active_flux,
        H0=H0,
        H1=H1,
        C=C,
        B01_boundary=B01_boundary,
        B01_volume=B01_volume,
        G=G,
        D=D,
        mixed_green_residual=mixed_residual,
        boundary_volume_residual=boundary_volume_residual,
    )


def _projected_product_map(
    direction: CutBMFDirection2D,
    flux_values_active: np.ndarray,
    scalar_values_active: np.ndarray,
    weights: np.ndarray,
    a: np.ndarray,
) -> np.ndarray:
    """Return A(a): scalar coefficients -> flux coefficients by L2 projection."""
    a_values = flux_values_active.T @ a
    rhs = (flux_values_active * (weights * a_values)) @ scalar_values_active.T
    return np.linalg.solve(direction.H1, rhs)


def _skew_residual_2d(
    xdir: CutBMFDirection2D,
    ydir: CutBMFDirection2D,
    x_flux_values: np.ndarray,
    y_flux_values: np.ndarray,
    scalar_values: np.ndarray,
    weights: np.ndarray,
    seed: int = 7,
) -> float:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=xdir.H0.shape[0])
    ax = rng.normal(size=xdir.H1.shape[0])
    ay = rng.normal(size=ydir.H1.shape[0])
    Ax = _projected_product_map(xdir, x_flux_values, scalar_values, weights, ax)
    Ay = _projected_product_map(ydir, y_flux_values, scalar_values, weights, ay)
    Ax_adj = np.linalg.solve(xdir.H0, Ax.T @ xdir.H1)
    Ay_adj = np.linalg.solve(ydir.H0, Ay.T @ ydir.H1)
    Cw = 0.5 * (
        xdir.D @ (Ax @ w)
        + Ax_adj @ (xdir.G @ w)
        + ydir.D @ (Ay @ w)
        + Ay_adj @ (ydir.G @ w)
    )
    lhs = float(w.T @ xdir.H0 @ Cw)
    rhs = 0.5 * float(
        w.T @ xdir.B01_boundary @ (Ax @ w)
        + w.T @ ydir.B01_boundary @ (Ay @ w)
    )
    return abs(lhs - rhs)


def _pressure_work_residual_2d(xdir: CutBMFDirection2D, ydir: CutBMFDirection2D, seed: int = 11) -> float:
    rng = np.random.default_rng(seed)
    p = rng.normal(size=xdir.H0.shape[0])
    ux = rng.normal(size=xdir.H1.shape[0])
    uy = rng.normal(size=ydir.H1.shape[0])
    lhs = float(ux.T @ xdir.H1 @ (xdir.G @ p) + uy.T @ ydir.H1 @ (ydir.G @ p))
    rhs = float(
        -p.T @ xdir.H0 @ (xdir.D @ ux + ydir.D @ uy)
        + p.T @ xdir.B01_boundary @ ux
        + p.T @ ydir.B01_boundary @ uy
    )
    return abs(lhs - rhs)


def build_bmf_cut_route_2d(
    basis_x: SplineBasis1D,
    basis_y: SplineBasis1D,
    cut: HalfPlaneCut,
    active_tol: float = 1.0e-22,
) -> CutBMFRoute2D:
    """Build the typed flux-space BMF^cut route on an oblique cut."""
    matrices = assemble_oblique_cut_matrices(
        basis_x, basis_y, cut, capture_volume_quadrature=True
    )
    active_scalar = oblique_active_sets(matrices, active_tol=active_tol).active
    basis_dx = make_derivative_basis(basis_x)
    basis_dy = make_derivative_basis(basis_y)
    xdir = _build_bmf_direction_2d(
        basis_x, basis_y, basis_dx, basis_y, matrices, cut, active_scalar, "x", active_tol
    )
    ydir = _build_bmf_direction_2d(
        basis_x, basis_y, basis_x, basis_dy, matrices, cut, active_scalar, "y", active_tol
    )

    q = matrices.volume_quadrature
    if q is None:
        raise ValueError("BMF assembly requires captured volume quadrature.")
    scalar_values = q.basis_values[active_scalar, :]
    basis_dx = make_derivative_basis(basis_x)
    basis_dy = make_derivative_basis(basis_y)
    x_flux_values = _tensor_eval(basis_dx, basis_y, q.x, q.y)[0][xdir.active_flux, :]
    y_flux_values = _tensor_eval(basis_x, basis_dy, q.x, q.y)[0][ydir.active_flux, :]
    skew_residual = _skew_residual_2d(
        xdir, ydir, x_flux_values, y_flux_values, scalar_values, q.weights
    )
    pressure_residual = _pressure_work_residual_2d(xdir, ydir)

    wave = np.block(
        [
            [
                np.zeros((xdir.H0.shape[0], xdir.H0.shape[0])),
                xdir.D,
                ydir.D,
            ],
            [
                xdir.G,
                np.zeros((xdir.H1.shape[0], xdir.H1.shape[0])),
                np.zeros((xdir.H1.shape[0], ydir.H1.shape[0])),
            ],
            [
                ydir.G,
                np.zeros((ydir.H1.shape[0], xdir.H1.shape[0])),
                np.zeros((ydir.H1.shape[0], ydir.H1.shape[0])),
            ],
        ]
    )
    radius = float(np.max(np.abs(np.linalg.eigvals(wave))))
    scale = basis_x.h / float((basis_x.degree + 1) ** 2)
    return CutBMFRoute2D(
        x=xdir,
        y=ydir,
        pressure_work_residual=pressure_residual,
        skew_energy_residual=skew_residual,
        bmf_wave_radius_h_over_p2=radius * scale,
    )


# ---------------------------------------------------------------------------
# Audit driver
# ---------------------------------------------------------------------------


def route_audit_row_2d(
    n_elements: int = 4,
    degree: int = 2,
    angle: float = 0.31 * pi,
    offset_fraction: float = 0.55,
) -> CutRouteAuditRow2D:
    basis_x = make_open_uniform_basis(n_elements=n_elements, degree=degree)
    basis_y = make_open_uniform_basis(n_elements=n_elements, degree=degree)
    # Offset is measured in background-cell units along the chosen normal, so
    # offset_fraction=0.55 cuts a first-layer cell without degeneracy.
    offset = offset_fraction * basis_x.h
    cut = HalfPlaneCut.from_angle(angle, offset)
    bmg = build_bmg_cut_route_2d(basis_x, basis_y, cut)
    bmf = build_bmf_cut_route_2d(basis_x, basis_y, cut)
    eig_x = np.linalg.eigvals(bmg.Dx)
    eig_y = np.linalg.eigvals(bmg.Dy)
    bmg_radius = float(max(np.max(np.abs(eig_x)), np.max(np.abs(eig_y))))
    scale = basis_x.h / float((degree + 1) ** 2)
    return CutRouteAuditRow2D(
        family="oblique2d",
        p=degree,
        n_elements=n_elements,
        angle=float(angle),
        offset=float(offset),
        n_scalar_active=int(len(bmg.active)),
        n_flux_x_active=int(len(bmf.x.active_flux)),
        n_flux_y_active=int(len(bmf.y.active_flux)),
        bmg_green_residual_x=bmg.green_residual_x,
        bmg_green_residual_y=bmg.green_residual_y,
        bmf_mixed_green_residual_x=bmf.x.mixed_green_residual,
        bmf_mixed_green_residual_y=bmf.y.mixed_green_residual,
        bmf_boundary_volume_residual_x=bmf.x.boundary_volume_residual,
        bmf_boundary_volume_residual_y=bmf.y.boundary_volume_residual,
        pressure_work_residual=bmf.pressure_work_residual,
        skew_energy_residual=bmf.skew_energy_residual,
        bmg_radius_h_over_p2=bmg_radius * scale,
        bmf_wave_radius_h_over_p2=bmf.bmf_wave_radius_h_over_p2,
    )


def write_route_audit_2d(rows: list[CutRouteAuditRow2D], out_dir: Path) -> dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "cut_mimetic_routes_2d.json"
    csv_path = out_dir / "cut_mimetic_routes_2d.csv"
    summary_path = out_dir / "cut_mimetic_routes_2d_summary.json"
    data = [asdict(row) for row in rows]
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    if data:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)
    summary = {
        "rows": len(rows),
        "max_bmg_green_residual_x": max((r.bmg_green_residual_x for r in rows), default=0.0),
        "max_bmg_green_residual_y": max((r.bmg_green_residual_y for r in rows), default=0.0),
        "max_bmf_mixed_green_residual_x": max((r.bmf_mixed_green_residual_x for r in rows), default=0.0),
        "max_bmf_mixed_green_residual_y": max((r.bmf_mixed_green_residual_y for r in rows), default=0.0),
        "max_bmf_boundary_volume_residual_x": max((r.bmf_boundary_volume_residual_x for r in rows), default=0.0),
        "max_bmf_boundary_volume_residual_y": max((r.bmf_boundary_volume_residual_y for r in rows), default=0.0),
        "max_pressure_work_residual": max((r.pressure_work_residual for r in rows), default=0.0),
        "max_skew_energy_residual": max((r.skew_energy_residual for r in rows), default=0.0),
        "max_bmg_radius_h_over_p2": max((r.bmg_radius_h_over_p2 for r in rows), default=0.0),
        "max_bmf_wave_radius_h_over_p2": max((r.bmf_wave_radius_h_over_p2 for r in rows), default=0.0),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
