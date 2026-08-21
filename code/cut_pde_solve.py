"""Genuine PDE solves on sharp-interface cut domains (not projection studies).

This module turns the cut-domain *assembly* infrastructure
(`assemble_circular_corner_matrices`, the polynomial-preserving extension, and
the physical-space pullback) into an actual finite-element **solver** for the
Poisson problem on the unit square minus a quarter disk at the origin:

    -Lap u = f   in  Omega = (0,1)^2 \\ {x^2 + y^2 < R^2},
         u = g   on  partial Omega   (full boundary: square sides + arc).

Dirichlet data are imposed weakly by the symmetric weak (penalty) method on the
*physical* spline space.  Because the physical space is polynomial-protected and
cut-mass-visible, the resulting stiffness is well conditioned independent of the
small cut-cell volume fraction, and the Galerkin solution converges at the
optimal rate (order p+1 in L2, order p in H1).

The driver `run_poisson_curved_solve` performs an h-convergence study over
spline degrees p = 2..5 and records measured slopes, a manufactured-solution
convergence study on the physical B-spline SBP space.  This is the genuine PDE-solve counterpart of the L2
curved-cut Poisson convergence study, swept across degrees in
`s6_curved_poisson_degree_sweep`.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from math import pi
from time import perf_counter
from typing import Callable

import numpy as np

from cut_spline_sbp import SplineBasis1D, make_open_uniform_basis
from circular_corner_sbp import (
    CircularCornerCut,
    CircularStructureMode,
    assemble_circular_corner_matrices,
    directional_sbp_residuals_circular,
    polynomial_preserving_circular_extension,
    pullback_circular_corner_2d,
    require_certified_structure,
    _angle_breaks,
    _node_chunk,
    _unique_sorted,
)
from oblique_cut_sbp import _tensor_values

ArrayFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
GradFn = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def _chunked_rhs(
    basis_x: SplineBasis1D,
    basis_y: SplineBasis1D,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    fvals: np.ndarray,
    chunk: int | None = None,
) -> np.ndarray:
    """``(basis_values * weights) @ fvals`` without materialising the dense tensor.

    The per-chunk arithmetic is identical to the dense expression
    ``(q.basis_values * q.weights) @ fvals``; the memory-budgeted chunk size
    (``_node_chunk``) keeps coarse and mid grids in a single chunk, so those
    cases are bit-identical to the dense path.  On the finest grids the only
    difference is the summation order across chunks (a fraction of an ULP).
    """
    n = basis_x.n_basis * basis_y.n_basis
    if chunk is None:
        chunk = _node_chunk(n, x.size)
    out = np.zeros(n)
    for s in range(0, x.size, chunk):
        e = min(s + chunk, x.size)
        phi, _dx, _dy = _tensor_values(basis_x, basis_y, x[s:e], y[s:e])
        out += (phi * weights[s:e]) @ fvals[s:e]
    return out


def _chunked_gram(
    basis_x: SplineBasis1D,
    basis_y: SplineBasis1D,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    chunk: int | None = None,
) -> np.ndarray:
    """Cut-volume Gram matrix ``(basis_values * weights) @ basis_values.T``.

    Accumulated over the same node chunks as ``_chunked_rhs`` so it never
    materialises the dense node-by-basis tensor; used by the best-approximation
    separation diagnostic (fixed-domain convergence).
    """
    n = basis_x.n_basis * basis_y.n_basis
    if chunk is None:
        chunk = _node_chunk(n, x.size)
    out = np.zeros((n, n))
    for s in range(0, x.size, chunk):
        e = min(s + chunk, x.size)
        phi, _dx, _dy = _tensor_values(basis_x, basis_y, x[s:e], y[s:e])
        out += (phi * weights[s:e]) @ phi.T
    return out


def _chunked_eval_field(
    basis_x: SplineBasis1D,
    basis_y: SplineBasis1D,
    x: np.ndarray,
    y: np.ndarray,
    alpha: np.ndarray,
    chunk: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate ``(u_h, du_h/dx, du_h/dy)`` at the quadrature nodes in chunks.

    Each output element is an independent dot product ``phi[:, q].T @ alpha``, so
    the result is bit-identical to the dense ``q.basis_values.T @ alpha`` for any
    chunk size (no cross-chunk summation).
    """
    if chunk is None:
        chunk = _node_chunk(basis_x.n_basis * basis_y.n_basis, x.size)
    uh = np.empty(x.size)
    uhx = np.empty(x.size)
    uhy = np.empty(x.size)
    for s in range(0, x.size, chunk):
        e = min(s + chunk, x.size)
        phi, dphix, dphiy = _tensor_values(basis_x, basis_y, x[s:e], y[s:e])
        uh[s:e] = phi.T @ alpha
        uhx[s:e] = dphix.T @ alpha
        uhy[s:e] = dphiy.T @ alpha
    return uh, uhx, uhy


# ---------------------------------------------------------------------------
# Manufactured solution: transcendental bubble vanishing on the exact boundary
# ---------------------------------------------------------------------------
#   u = sin(pi x) sin(pi y) (x^2 + y^2 - R^2)
# vanishes on the outer square (sin factors) and on the arc x^2+y^2=R^2 (last
# factor), so it satisfies homogeneous Dirichlet data on all of partial Omega.
# It is not a polynomial, so it is a genuine approximation target at every p.


def manufactured_poisson(radius: float) -> tuple[ArrayFn, GradFn, ArrayFn]:
    r2 = float(radius * radius)

    def u(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.sin(pi * x) * np.sin(pi * y) * (x * x + y * y - r2)

    def grad_u(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = np.sin(pi * x) * np.sin(pi * y)
        sx = pi * np.cos(pi * x) * np.sin(pi * y)
        sy = pi * np.sin(pi * x) * np.cos(pi * y)
        g = x * x + y * y - r2
        return sx * g + 2.0 * x * s, sy * g + 2.0 * y * s

    def f(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # f = -Lap u = 2 pi^2 S G - 4 x S_x - 4 y S_y - 4 S
        s = np.sin(pi * x) * np.sin(pi * y)
        sx = pi * np.cos(pi * x) * np.sin(pi * y)
        sy = pi * np.sin(pi * x) * np.cos(pi * y)
        g = x * x + y * y - r2
        return 2.0 * pi * pi * s * g - 4.0 * x * sx - 4.0 * y * sy - 4.0 * s

    return u, grad_u, f


# ---------------------------------------------------------------------------
# Boundary quadrature on Omega's full boundary (arc + the four square sides),
# returning nodes, weights, and the OUTWARD unit normal of Omega at each node.
# ---------------------------------------------------------------------------


def _segment_nodes(
    fixed_axis: str,
    fixed_value: float,
    lo: float,
    hi: float,
    normal: tuple[float, float],
    knots_perp: np.ndarray,
    order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g, w = np.polynomial.legendre.leggauss(order)
    panels = _unique_sorted([float(k) for k in np.unique(knots_perp) if lo < k < hi], lo, hi)
    xs, ys, ws, nxs, nys = [], [], [], [], []
    for a, b in zip(panels[:-1], panels[1:]):
        if b <= a:
            continue
        t = 0.5 * (b - a) * g + 0.5 * (b + a)
        wt = 0.5 * (b - a) * w
        if fixed_axis == "x":
            x = np.full_like(t, fixed_value)
            y = t
        else:
            x = t
            y = np.full_like(t, fixed_value)
        xs.append(x)
        ys.append(y)
        ws.append(wt)
        nxs.append(np.full_like(t, normal[0]))
        nys.append(np.full_like(t, normal[1]))
    if not xs:
        z = np.empty(0)
        return z, z, z, z, z
    return (
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(ws),
        np.concatenate(nxs),
        np.concatenate(nys),
    )


def _boundary_quadrature_circular(
    basis_x: SplineBasis1D,
    basis_y: SplineBasis1D,
    radius: float,
    order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """All of partial Omega: arc + right/top/left/bottom square sides."""
    xs, ys, ws, nxs, nys = [], [], [], [], []

    # Arc x^2+y^2=R^2.  Outward normal of Omega points into the removed disk: -e_r.
    g, w = np.polynomial.legendre.leggauss(order)
    panels = _angle_breaks(radius, basis_x.knots, basis_y.knots)
    for a, b in zip(panels[:-1], panels[1:]):
        if b <= a:
            continue
        theta = 0.5 * (b - a) * g + 0.5 * (b + a)
        wt = 0.5 * (b - a) * w * radius
        c = np.cos(theta)
        s = np.sin(theta)
        xs.append(radius * c)
        ys.append(radius * s)
        ws.append(wt)
        nxs.append(-c)
        nys.append(-s)

    # Square sides (left/bottom start after the removed quarter disk).
    sides = [
        ("x", 1.0, 0.0, 1.0, (1.0, 0.0), basis_y.knots),
        ("y", 1.0, 0.0, 1.0, (0.0, 1.0), basis_x.knots),
        ("x", 0.0, radius, 1.0, (-1.0, 0.0), basis_y.knots),
        ("y", 0.0, radius, 1.0, (0.0, -1.0), basis_x.knots),
    ]
    for fixed_axis, fixed_value, lo, hi, normal, knots_perp in sides:
        sx, sy, sw, snx, sny = _segment_nodes(
            fixed_axis, fixed_value, lo, hi, normal, knots_perp, order
        )
        if sx.size:
            xs.append(sx)
            ys.append(sy)
            ws.append(sw)
            nxs.append(snx)
            nys.append(sny)

    return (
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(ws),
        np.concatenate(nxs),
        np.concatenate(nys),
    )


def _weak_boundary_operators(
    basis_x: SplineBasis1D,
    basis_y: SplineBasis1D,
    radius: float,
    order: int,
    boundary_quadrature=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return full-space (M_bnd, N_flux, bx, by, nx, ny, w, normal-deriv basis).

    M_bnd[i,j] = int_dOmega phi_i phi_j ds                (boundary mass)
    N_flux[i,j] = int_dOmega (dn phi_j) phi_i ds          (weak-penalty consistency)
    plus the boundary basis values / normal derivative for RHS assembly.

    ``boundary_quadrature`` overrides the default split-Gauss boundary rule with
    an alternative (the curved arc rule) sharing the signature
    ``(basis_x, basis_y, radius, order) -> (x, y, w, nx, ny)``.
    """
    bq = boundary_quadrature or _boundary_quadrature_circular
    bx, by, bw, bnx, bny = bq(basis_x, basis_y, radius, order)
    phi, dphix, dphiy = _tensor_values(basis_x, basis_y, bx, by)
    dphin = bnx * dphix + bny * dphiy  # (n_full, nq)
    M_bnd = (phi * bw) @ phi.T
    N_flux = (phi * bw) @ dphin.T
    return M_bnd, N_flux, phi, dphin, bx, by, bw, bny  # bny unused downstream but kept


@dataclass(frozen=True)
class PoissonSolveCase:
    family: str
    p: int
    n_elements: int
    h: float
    radius: float
    n_active: int
    n_physical: int
    penalty_beta: float
    spd_min_eig: float
    cond_stiffness: float
    l2_error: float
    h1_error: float
    boundary_l2_error: float
    boundary_l2_over_solution_l2: float
    relative_l2_error: float
    relative_h1_error: float
    relative_linf_error: float
    # L2 best approximation of the exact solution in the same aggregated space
    # under the same quadrature (NaN unless with_best_approx=True): separates
    # what the space can represent from what the weak-penalty solve delivers.
    best_l2_error: float = float("nan")
    relative_best_l2_error: float = float("nan")


def solve_poisson_circular(
    p: int,
    n_elements: int,
    radius: float = 0.42,
    beta_factor: float = 10.0,
    quadrature_order: int | None = None,
    return_fields: bool = False,
    extension_builder=None,
    quadrature: str = "split_gauss",
    support_threshold: float = 0.04,
    with_best_approx: bool = False,
    return_diagnostics: bool = False,
    structure_mode: CircularStructureMode = CircularStructureMode.RAW,
) -> PoissonSolveCase:
    """Solve -Lap u = f with weak-penalty Dirichlet on the physical spline space.

    With ``return_fields=True`` also returns the basis and physical-space
    solution coefficients for plotting the solution on the cut domain.

    ``extension_builder`` selects the small-cell reduction.  It defaults to the
    polynomial-pinning extension; passing an alternative (plain deletion, cell
    agglomeration) runs the identical solve through a different reduction so the
    head-to-head in Section 6 is fair.  The builder must accept
    ``(basis_x, basis_y, matrices, support_threshold=, preserve_degree=)`` and
    return an aggregation object exposing ``.E``, ``.active``, and
    ``.polynomial_residual``.

    ``quadrature`` selects the cut-integration rule.  ``"split_gauss"`` (the
    default, unchanged) triangulates the clipped cell and lays a per-panel Gauss
    rule of order ``quadrature_order``; ``"curved"`` uses the implicit-geometry
    dimension-reduction quadrature of ``quad_curved`` , which
    reaches the roundoff Green-identity floor at panel order ``q ~ p+4`` instead
    of the ``6p+2`` the split-Gauss rule needs on a curved cell.  With the curved rule the
    default order is ``q = p + 4`` (already at the roundoff floor) unless
    ``quadrature_order`` overrides it.
    """
    mission_start = perf_counter()
    stage_seconds: dict[str, float] = {}

    stage_start = perf_counter()
    basis_x = make_open_uniform_basis(n_elements=n_elements, degree=p)
    basis_y = make_open_uniform_basis(n_elements=n_elements, degree=p)
    h = 1.0 / n_elements
    cut = CircularCornerCut(radius=radius)
    stage_seconds["basis_setup"] = perf_counter() - stage_start

    stage_start = perf_counter()
    if quadrature == "curved":
        from quad_curved import (
            assemble_curved_circular_matrices,
            curved_boundary_quadrature,
        )

        q_order = quadrature_order or (p + 4)
        # capture_basis=False: the Poisson solver streams the basis in chunks, so
        # the dense O(n^4) tensor is skipped (fits p=8, n=32 under the budget).
        matrices = assemble_curved_circular_matrices(
            basis_x, basis_y, cut, q_order,
            capture_volume_quadrature=True, capture_basis=False,
            structure_mode=structure_mode,
        )
        boundary_quadrature = curved_boundary_quadrature
    elif quadrature == "split_gauss":
        q_order = quadrature_order or max(32, 2 * p + 14)
        # Capture nodes/weights only; the solver streams the basis in chunks for
        # RHS and error norms (see _chunked_rhs / _chunked_eval_field), so the
        # dense O(n^4) basis tensor never materialises.  This keeps the finest
        # high-order meshes (p=6, n=32) well under the memory budget.
        matrices = assemble_circular_corner_matrices(
            basis_x, basis_y, cut, quadrature_order=q_order,
            capture_volume_quadrature=True, capture_basis=False,
            structure_mode=structure_mode,
        )
        boundary_quadrature = None
    else:
        raise ValueError(f"unknown quadrature rule: {quadrature!r}")
    stage_seconds["sharp_assembly"] = perf_counter() - stage_start

    if CircularStructureMode(structure_mode) is CircularStructureMode.CERTIFIED:
        require_certified_structure(matrices)

    stage_start = perf_counter()
    builder = extension_builder or polynomial_preserving_circular_extension
    agg = builder(
        basis_x, basis_y, matrices, support_threshold=support_threshold, preserve_degree=p
    )
    E = agg.E  # (n_full, n_physical)
    stage_seconds["extension_construction"] = perf_counter() - stage_start

    q = matrices.volume_quadrature
    if q is None or q.basis_grad_x is None:
        raise RuntimeError("volume quadrature was not captured")
    # A captured quadrature may carry the dense basis tensor (split-Gauss) or
    # only its nodes/weights with an empty basis tensor (curved rule, to stay within a
    # memory budget on fine high-order grids); the chunked helpers below cover
    # both so the solve never depends on the dense tensor being present.
    chunked_basis = q.basis_values.shape[1] == 0 and q.x.size > 0

    u_exact, grad_exact, f_rhs = manufactured_poisson(radius)

    # weak-penalty operators in full space, then pull back to the physical space.
    stage_start = perf_counter()
    M_bnd, N_flux, boundary_phi, _boundary_dphin, bx, by, bw, _bny = _weak_boundary_operators(
        basis_x, basis_y, radius, q_order, boundary_quadrature=boundary_quadrature
    )
    stage_seconds["boundary_assembly"] = perf_counter() - stage_start

    stage_start = perf_counter()
    reduced = pullback_circular_corner_2d(matrices, E)
    K_r = reduced.K
    M_r = E.T @ M_bnd @ E
    N_r = E.T @ N_flux @ E

    beta = beta_factor * (p + 1) ** 2
    # Symmetric weak-penalty: a(u,v)=int grad u.grad v - int dn u v - int u dn v + beta/h int u v
    A = K_r - N_r - N_r.T + (beta / h) * M_r
    A = 0.5 * (A + A.T)
    stage_seconds["restriction"] = perf_counter() - stage_start

    # Load: int_Omega f v  (homogeneous Dirichlet => weak-penalty RHS boundary terms vanish)
    stage_start = perf_counter()
    fvals = f_rhs(q.x, q.y)
    if chunked_basis:
        b_full = _chunked_rhs(basis_x, basis_y, q.x, q.y, q.weights, fvals)
    else:
        b_full = (q.basis_values * q.weights) @ fvals
    b_r = E.T @ b_full
    stage_seconds["load_assembly"] = perf_counter() - stage_start

    stage_start = perf_counter()
    coeff = np.linalg.solve(A, b_r)
    alpha = E @ coeff
    stage_seconds["linear_solve"] = perf_counter() - stage_start

    # Errors on the cut-domain volume quadrature.
    stage_start = perf_counter()
    u = u_exact(q.x, q.y)
    ux, uy = grad_exact(q.x, q.y)
    w = q.weights
    if chunked_basis:
        uh, uhx, uhy = _chunked_eval_field(basis_x, basis_y, q.x, q.y, alpha)
    else:
        uh = q.basis_values.T @ alpha
        uhx = q.basis_grad_x.T @ alpha
        uhy = q.basis_grad_y.T @ alpha
    l2 = float(np.sqrt(np.sum(w * (uh - u) ** 2)))
    h1 = float(np.sqrt(np.sum(w * ((uhx - ux) ** 2 + (uhy - uy) ** 2))))
    l2_ref = float(np.sqrt(np.sum(w * u**2)))
    h1_ref = float(np.sqrt(np.sum(w * (ux**2 + uy**2))))
    # maximum-norm solve error over the cut volume quadrature (pointwise certificate)
    linf = float(np.max(np.abs(uh - u)))
    linf_ref = float(np.max(np.abs(u)))
    ub = boundary_phi.T @ alpha
    gb = u_exact(bx, by)
    boundary_l2 = float(np.sqrt(np.sum(bw * (ub - gb) ** 2)))
    stage_seconds["error_evaluation"] = perf_counter() - stage_start

    stage_start = perf_counter()
    best_l2 = float("nan")
    if with_best_approx:
        # L2 projection of u onto the same aggregated space with the same cut
        # quadrature: the best-approximation floor of the space, independent of
        # the boundary enforcement and the linear solve.
        M_full = _chunked_gram(basis_x, basis_y, q.x, q.y, w)
        rhs_u = _chunked_rhs(basis_x, basis_y, q.x, q.y, w, u)
        projection_mass_r = E.T @ M_full @ E
        rhs_r = E.T @ rhs_u
        coeff_best = np.linalg.lstsq(
            0.5 * (projection_mass_r + projection_mass_r.T), rhs_r, rcond=None
        )[0]
        alpha_best = E @ coeff_best
        if chunked_basis:
            ub_best, _bx, _by = _chunked_eval_field(basis_x, basis_y, q.x, q.y, alpha_best)
        else:
            ub_best = q.basis_values.T @ alpha_best
        best_l2 = float(np.sqrt(np.sum(w * (ub_best - u) ** 2)))
    stage_seconds["best_approximation"] = perf_counter() - stage_start

    stage_start = perf_counter()
    eigs = np.linalg.eigvalsh(A)
    stage_seconds["spectrum"] = perf_counter() - stage_start
    case = PoissonSolveCase(
        family="poisson_curved_solve",
        p=p,
        n_elements=n_elements,
        h=h,
        radius=radius,
        n_active=int(len(agg.active)),
        n_physical=int(E.shape[1]),
        penalty_beta=float(beta),
        spd_min_eig=float(eigs[0]),
        cond_stiffness=float(eigs[-1] / eigs[0]) if eigs[0] > 0 else float("inf"),
        l2_error=l2,
        h1_error=h1,
        boundary_l2_error=boundary_l2,
        boundary_l2_over_solution_l2=boundary_l2 / max(l2_ref, 1.0e-300),
        relative_l2_error=l2 / l2_ref,
        relative_h1_error=h1 / h1_ref,
        relative_linf_error=linf / linf_ref,
        best_l2_error=best_l2,
        relative_best_l2_error=best_l2 / l2_ref,
    )
    stage_seconds["total_internal"] = perf_counter() - mission_start
    if return_fields or return_diagnostics:
        payload = {}
        if return_fields:
            payload.update({
            "basis_x": basis_x,
            "basis_y": basis_y,
            "alpha": alpha,
            "radius": float(radius),
            "degree": int(p),
            })
        if return_diagnostics:
            payload["diagnostics"] = {
                "aggregation": agg,
                "extension": E,
                "sharp_matrices": matrices,
                "reduced_matrices": reduced,
                "boundary_mass": M_r,
                "boundary_flux": N_r,
                "weak_bc_matrix": A,
                "reduced_load": b_r,
                "solution_coefficients": coeff,
                "stiffness_eigenvalues": eigs,
                "sharp_green_residuals": directional_sbp_residuals_circular(matrices),
                "reduced_green_residuals": directional_sbp_residuals_circular(reduced),
                "sharp_green_diagnostics": {
                    "x": matrices.green_diagnostics_x,
                    "y": matrices.green_diagnostics_y,
                },
                "structure_mode": matrices.structure_mode,
                "structure_status": matrices.structure_status,
                "green_certificate": matrices.green_certificate,
                "sharp_structure_lineage": matrices.structure_lineage,
                "reduced_structure_lineage": reduced.structure_lineage,
                "stage_seconds": stage_seconds,
            }
        return case, payload
    return case


def _fit_slope(h: np.ndarray, err: np.ndarray, last: int | None = None) -> float:
    mask = err > 0
    hh, ee = h[mask], err[mask]
    if last is not None and hh.size > last:
        order = np.argsort(hh)  # ascending h
        hh, ee = hh[order][:last], ee[order][:last]  # finest `last` grids
    if hh.size < 2:
        return float("nan")
    return float(np.polyfit(np.log(hh), np.log(ee), 1)[0])


def run_poisson_curved_solve(
    degrees: tuple[int, ...] = (2, 3, 4, 5),
    elements: tuple[int, ...] = (8, 12, 16, 24, 32),
    radius: float = 0.42,
    write: bool = True,
) -> dict:
    """h-convergence study of the genuine Poisson Galerkin solve."""
    cases = [solve_poisson_circular(p, n, radius=radius) for p in degrees for n in elements]
    rows = [asdict(c) for c in cases]
    slopes: dict[str, dict[str, float]] = {}
    for p in degrees:
        sub = sorted([c for c in cases if c.p == p], key=lambda c: c.h, reverse=True)
        harr = np.array([c.h for c in sub])
        l2 = np.array([c.relative_l2_error for c in sub])
        h1 = np.array([c.relative_h1_error for c in sub])
        slopes[str(p)] = {
            "l2_slope": _fit_slope(harr, l2),
            "h1_slope": _fit_slope(harr, h1),
            "boundary_l2_slope": _fit_slope(
                harr, np.array([c.boundary_l2_error for c in sub])
            ),
            "l2_slope_asymptotic": _fit_slope(harr, l2, last=3),
            "h1_slope_asymptotic": _fit_slope(harr, h1, last=3),
            "boundary_l2_slope_asymptotic": _fit_slope(
                harr, np.array([c.boundary_l2_error for c in sub]), last=3
            ),
            "l2_target": p + 1,
            "h1_target": p,
            "boundary_l2_reference": p + 0.5,
        }
    summary = {
        "rows": len(rows),
        "radius": radius,
        "degree_slopes": slopes,
        "min_spd_eig": min(r["spd_min_eig"] for r in rows),
        "max_cond_stiffness": max(r["cond_stiffness"] for r in rows),
        "min_boundary_l2_slope_asymptotic": min(
            slopes[str(p)]["boundary_l2_slope_asymptotic"] for p in degrees
        ),
        "max_boundary_l2_error": max(r["boundary_l2_error"] for r in rows),
    }
    if write:
        _write_artifacts(rows, slopes, summary)
    return {"rows": rows, "degree_slopes": slopes, "summary": summary}


def _write_artifacts(rows, slopes, summary) -> None:
    import csv
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    results = root / "code" / "results"
    figures = root / "figures"
    results.mkdir(parents=True, exist_ok=True)
    (results / "poisson_curved_solve.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (results / "poisson_curved_solve.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (results / "poisson_curved_solve_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Convergence figure (compact, half-page width when included).
    try:
        import matplotlib

        matplotlib.use("Agg")
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
        markers = {2: "o", 3: "s", 4: "^", 5: "D"}
        for ax, key, lab in (
            (axes[0], "relative_l2_error", r"relative $L^2$ error"),
            (axes[1], "relative_h1_error", r"relative $H^1$ error"),
        ):
            for p in sorted({r["p"] for r in rows}):
                sub = sorted([r for r in rows if r["p"] == p], key=lambda r: r["h"], reverse=True)
                hh = np.array([r["h"] for r in sub])
                ee = np.array([r[key] for r in sub])
                ax.loglog(hh, ee, marker=markers.get(p, "o"), lw=1.2, label=fr"$p={p}$")
                # reference triangle at optimal order
                order = (p + 1) if key == "relative_l2_error" else p
                ref = ee[0] * (hh / hh[0]) ** order
                ax.loglog(hh, ref, color="0.7", ls="--", lw=0.8)
            ax.invert_xaxis()
            ax.set_xlabel(r"$h$")
            ax.set_ylabel(lab)
            ax.grid(True, which="both", alpha=0.25)
        axes[0].set_title("(a) Poisson solve, $L^2$")
        axes[1].set_title("(b) Poisson solve, $H^1$")
        axes[0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / "fig_poisson_curved_solve.pdf")
        fig.savefig(figures / "fig_poisson_curved_solve.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.8, 3.2))
        for p in sorted({r["p"] for r in rows}):
            sub = sorted([r for r in rows if r["p"] == p], key=lambda r: r["h"], reverse=True)
            hh = np.array([r["h"] for r in sub])
            ee = np.array([r["boundary_l2_error"] for r in sub])
            ax.loglog(hh, ee, marker=markers.get(p, "o"), lw=1.2, label=fr"$p={p}$")
            ref_order = p + 0.5
            ref = ee[0] * (hh / hh[0]) ** ref_order
            ax.loglog(hh, ref, color="0.7", ls="--", lw=0.8)
        ax.invert_xaxis()
        ax.set_xlabel(r"$h$")
        ax.set_ylabel(r"boundary $L^2$ error")
        ax.set_title("wall-order trace")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / "fig_poisson_wall_order.pdf")
        fig.savefig(figures / "fig_poisson_wall_order.png", dpi=150)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        print(f"[cut_pde_solve] figure skipped: {exc}")


def write_poisson_solve_table() -> str:
    """Emit paper/generated_poisson_solve_table.tex from the stored summary."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    summary = json.loads(
        (root / "code" / "results" / "poisson_curved_solve_summary.json").read_text()
    )
    slopes = summary["degree_slopes"]
    lines = [
        "% Auto-generated by code/cut_pde_solve.py (write_poisson_solve_table).",
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"$p$ & $L^2$ slope & target $p{+}1$ & $H^1$ slope & target $p$\\",
        r"\midrule",
    ]
    for p in sorted(slopes, key=int):
        s = slopes[p]
        lines.append(
            f"${p}$ & ${s['l2_slope_asymptotic']:.2f}$ & ${s['l2_target']}$ & "
            f"${s['h1_slope_asymptotic']:.2f}$ & ${s['h1_target']}$" + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = root / "paper" / "generated_poisson_solve_table.tex"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(out)


if __name__ == "__main__":
    import json

    result = run_poisson_curved_solve()
    print(json.dumps(result["summary"], indent=2))
    print(write_poisson_solve_table())
