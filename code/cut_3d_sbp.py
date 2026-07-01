"""Three-dimensional tensor-product cut-cell B-spline SBP machinery.

This module lifts the 1D cut-domain spline operators of :mod:`cut_spline_sbp`
to genuine three-dimensional cut boxes by Kronecker products, and provides the
3D evidence that the small-cell relief, exact Green identities, polynomial
reproduction, and cut-independent spectra of the planar/oblique 2D study carry
over to three dimensions.

The cut box is ``Omega = [a_x,1] x [a_y,1] x [a_z,1]`` with ``a_i = h*frac_i``.
A fraction near one places the cut just below the first interior knot, so the
first physical element on that axis is a sliver of width ``h*(1-frac)``.
Cutting on one axis gives a thin slab; cutting on two or three axes
simultaneously gives genuine 3D edge and corner slivers whose smallest cut-cell
volume ratio is ``(1-frac)^k`` for ``k`` cut axes, i.e. arbitrarily small.  The
central statement is that no such sliver owns an independent numerical
direction: the naive active mass floor collapses in proportion to the sliver
volume, but after the polynomial-preserving pullback the mass, stiffness,
advection, and Green operators have constants independent of the cut fractions.

Index convention: global degrees of freedom are flattened with the x-index
fastest, so a separable 3D operator is ``np.kron(np.kron(Az, Ay), Ax)`` and the
3D coefficient extension is ``np.kron(np.kron(Ez, Ey), Ex)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.linalg import eigvalsh

from cut_spline_sbp import (
    CutMatrices1D,
    SplineBasis1D,
    _eval_all,
    _quadrature_panels,
    active_sets,
    assemble_cut_matrices,
    make_open_uniform_basis,
    polynomial_coefficients,
    polynomial_preserving_extension,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "code" / "results"


def _kron3(az: np.ndarray, ay: np.ndarray, ax: np.ndarray) -> np.ndarray:
    return np.kron(np.kron(az, ay), ax)


@dataclass(frozen=True)
class Cut3DOperators:
    h: float
    degree: int
    n_full: int
    n_phys: int
    cut_fractions: tuple[float, float, float]
    volume_fraction: float
    naive_scaled_mass_min_eig: float
    H: np.ndarray
    K: np.ndarray
    Qx: np.ndarray
    Qy: np.ndarray
    Qz: np.ndarray
    Sx: np.ndarray
    Sy: np.ndarray
    Sz: np.ndarray
    E: np.ndarray


def _axis_matrices(
    basis: SplineBasis1D, left: float, support_threshold: float, preserve_degree: int
) -> tuple[CutMatrices1D, np.ndarray, float]:
    m = assemble_cut_matrices(basis, left=left)
    agg = polynomial_preserving_extension(
        basis, m, support_threshold=support_threshold, preserve_degree=preserve_degree
    )
    # Naive active-space mass floor on this axis (before any extension): the
    # smallest mass eigenvalue carried by an active basis function.  This is the
    # quantity that collapses with the sliver width.
    active, _interior, _small, _gamma = active_sets(m, support_threshold=support_threshold)
    naive_min_eig = float(eigvalsh(m.H[np.ix_(active, active)])[0]) if len(active) else 0.0
    return m, agg.E, naive_min_eig


def assemble_cut_3d(
    n_elements: int = 5,
    degree: int = 3,
    cut_fractions: Sequence[float] = (0.0, 0.0, 0.0),
    support_threshold: float = 0.05,
    preserve_degree: int | None = None,
) -> Cut3DOperators:
    """Assemble pulled-back 3D cut operators for a tensor-product cut box.

    ``cut_fractions[i]`` is the fraction of one background element removed from
    the low side of axis ``i`` (so the left cut sits at ``a_i = h*frac``).  A
    nonzero fraction on several axes simultaneously produces a 3D corner sliver.
    """
    basis = make_open_uniform_basis(n_elements=n_elements, degree=degree)
    h = basis.h
    pdeg = degree if preserve_degree is None else preserve_degree
    fracs = tuple(float(f) for f in cut_fractions)
    lefts = tuple(h * f for f in fracs)

    mx, Ex, nx = _axis_matrices(basis, lefts[0], support_threshold, pdeg)
    my, Ey, ny = _axis_matrices(basis, lefts[1], support_threshold, pdeg)
    mz, Ez, nz = _axis_matrices(basis, lefts[2], support_threshold, pdeg)

    E = _kron3(Ez, Ey, Ex)
    H = _kron3(mz.H, my.H, mx.H)
    K = _kron3(mz.H, my.H, mx.K) + _kron3(mz.H, my.K, mx.H) + _kron3(mz.K, my.H, mx.H)
    Qx = _kron3(mz.H, my.H, mx.Q)
    Qy = _kron3(mz.H, my.Q, mx.H)
    Qz = _kron3(mz.Q, my.H, mx.H)
    Sx = _kron3(mz.H, my.H, mx.boundary_form)
    Sy = _kron3(mz.H, my.boundary_form, mx.H)
    Sz = _kron3(mz.boundary_form, my.H, mx.H)

    def pb(a: np.ndarray) -> np.ndarray:
        return E.T @ a @ E

    # Smallest cut-cell volume ratio: product over cut axes of the sliver width
    # fraction (1-frac).  Axes with frac==0 are uncut and do not shrink the cell.
    sliver_ratio = float(np.prod([1.0 - f for f in fracs if f > 0.0])) if any(f > 0.0 for f in fracs) else 1.0
    naive_floor = float(nx * ny * nz / h**3)
    return Cut3DOperators(
        h=h,
        degree=degree,
        n_full=int(E.shape[0]),
        n_phys=int(E.shape[1]),
        cut_fractions=fracs,
        volume_fraction=sliver_ratio,
        naive_scaled_mass_min_eig=naive_floor,
        H=0.5 * (pb(H) + pb(H).T),
        K=0.5 * (pb(K) + pb(K).T),
        Qx=pb(Qx),
        Qy=pb(Qy),
        Qz=pb(Qz),
        Sx=pb(Sx),
        Sy=pb(Sy),
        Sz=pb(Sz),
        E=E,
    )


def green_residuals_3d(ops: Cut3DOperators) -> tuple[float, float, float]:
    """Directional SBP/Green residuals ``||Q_i + Q_i^T - S_i||_inf`` after pullback."""
    rx = float(np.linalg.norm(ops.Qx + ops.Qx.T - ops.Sx, ord=np.inf))
    ry = float(np.linalg.norm(ops.Qy + ops.Qy.T - ops.Sy, ord=np.inf))
    rz = float(np.linalg.norm(ops.Qz + ops.Qz.T - ops.Sz, ord=np.inf))
    return rx, ry, rz


def scaled_mass_min_eig(ops: Cut3DOperators) -> float:
    """Smallest mass eigenvalue scaled by ``h^3`` (bounded below = no collapse)."""
    return float(eigvalsh(ops.H)[0] / ops.h**3)


def mass_condition(ops: Cut3DOperators) -> float:
    eig = eigvalsh(ops.H)
    return float(eig[-1] / max(eig[0], 1.0e-300))


def scaled_advection_radius(ops: Cut3DOperators) -> float:
    """Scaled spectral radius ``rho(H^{-1} Q_x) * h`` (the CFL-relief proxy)."""
    A = np.linalg.solve(ops.H, ops.Qx)
    return float(np.max(np.abs(np.linalg.eigvals(A))) * ops.h)


def polynomial_reproduction_residual_3d(
    n_elements: int, degree: int, cut_fractions: Sequence[float], preserve_degree: int | None = None
) -> float:
    """Max residual reproducing tensor monomials x^a y^b z^c (a,b,c <= p) after pullback.

    The extension is exact on the protected sector, so the residual should sit
    at roundoff regardless of the cut fractions.
    """
    basis = make_open_uniform_basis(n_elements=n_elements, degree=degree)
    h = basis.h
    pdeg = degree if preserve_degree is None else preserve_degree
    fracs = [float(f) for f in cut_fractions]
    coeffs = polynomial_coefficients(basis, pdeg)  # (pdeg+1, n_basis): greville-fit monomials

    Es = []
    for f in fracs:
        m = assemble_cut_matrices(basis, left=h * f)
        agg = polynomial_preserving_extension(basis, m, preserve_degree=pdeg)
        Es.append(agg.E)
    Ex, Ey, Ez = Es

    # Each 1D extension reproduces 1D monomials: pick the physical coefficients
    # that minimise || E c_phys - c_poly ||, then measure the reproduction gap.
    def axis_gap(E: np.ndarray) -> float:
        gap = 0.0
        for r in range(pdeg + 1):
            target = coeffs[r]
            c_phys, *_ = np.linalg.lstsq(E, target, rcond=None)
            gap = max(gap, float(np.linalg.norm(E @ c_phys - target, ord=np.inf)))
        return gap

    return max(axis_gap(Ex), axis_gap(Ey), axis_gap(Ez))


def degree_p_cut_frontier(
    degree: int,
    n_elements: int = 8,
    fractions: Sequence[float] | None = None,
    tol: float = 1.0e-10,
) -> dict:
    """Quantify the extreme-cut conditioning frontier for degree-``p`` reproduction.

    Sweeps the sliver fraction toward one and records the most extreme fraction
    (smallest 1D cut-cell width) at which the polynomial-preserving extension
    still reproduces all degree-``p`` monomials to within ``tol``.  Beyond the
    frontier the local polynomial-moment stencil of the rebuilt small-support
    function goes rank deficient: this is the high-degree roundoff regime, not a
    failure of the formulation theorem.
    """
    if fractions is None:
        fractions = (0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999)
    basis = make_open_uniform_basis(n_elements=n_elements, degree=degree)
    last_ok = 0.0
    rows = []
    for frac in fractions:
        m = assemble_cut_matrices(basis, left=basis.h * frac)
        agg = polynomial_preserving_extension(basis, m, preserve_degree=degree)
        ok = agg.polynomial_residual < tol
        rows.append({"fraction": float(frac), "sliver_width_ratio": float(1.0 - frac), "residual": float(agg.polynomial_residual), "reproduced": bool(ok)})
        if ok:
            last_ok = float(frac)
    return {
        "degree": degree,
        "n_elements": n_elements,
        "max_reproduced_fraction": last_ok,
        "min_reproduced_sliver_ratio": float(1.0 - last_ok) if last_ok else 1.0,
        "rows": rows,
    }


def run_high_degree_frontier(
    degrees: Sequence[int] = (2, 3, 4, 5, 6),
    n_elements: int = 8,
    write: bool = True,
) -> dict:
    """Frontier of extreme-cut degree-p reproduction across degrees."""
    frontiers = [degree_p_cut_frontier(p, n_elements=n_elements) for p in degrees]
    summary = {
        "n_elements": n_elements,
        "degrees": list(degrees),
        "frontier": {
            f["degree"]: {
                "max_reproduced_fraction": f["max_reproduced_fraction"],
                "min_reproduced_sliver_ratio": f["min_reproduced_sliver_ratio"],
            }
            for f in frontiers
        },
    }
    if write:
        import json

        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "high_degree_cut_frontier.json").write_text(
            json.dumps({"summary": summary, "detail": frontiers}, indent=2), encoding="utf-8"
        )
    return summary


def run_small_cell_3d_sweep(
    n_elements: int = 8,
    degrees: Sequence[int] = (2, 3),
    fractions: Sequence[float] = (0.5, 0.9, 0.99, 0.999),
    write: bool = True,
) -> dict:
    """Sweep 3D cut configurations and certify cut-independent operators.

    For each degree and each sliver fraction (the cut sits at ``h*frac``, so the
    first cell has width ``h*(1-frac)``) we test three geometries: a single-axis
    slab, a two-axis edge sliver, and a three-axis corner sliver.  At
    ``frac=0.999`` the corner cut cell has volume ratio ``(1e-3)^3 = 1e-9``.  We
    report the naive (pre-extension) scaled mass floor, which collapses with the
    sliver, alongside the pulled-back scaled mass floor, conditioning, scaled
    advection radius, Green residual, and polynomial reproduction residual.
    """
    geometries = {
        "slab_1axis": lambda f: (f, 0.0, 0.0),
        "edge_2axis": lambda f: (f, f, 0.0),
        "corner_3axis": lambda f: (f, f, f),
    }
    rows: list[dict] = []
    for p in degrees:
        for geo_name, geo in geometries.items():
            for frac in fractions:
                fr = geo(frac)
                ops = assemble_cut_3d(n_elements=n_elements, degree=p, cut_fractions=fr)
                rx, ry, rz = green_residuals_3d(ops)
                poly = polynomial_reproduction_residual_3d(n_elements, p, fr)
                pulled = scaled_mass_min_eig(ops)
                naive = ops.naive_scaled_mass_min_eig
                rows.append(
                    {
                        "degree": p,
                        "geometry": geo_name,
                        "cut_fraction": float(frac),
                        "volume_fraction": ops.volume_fraction,
                        "n_phys": ops.n_phys,
                        "naive_scaled_mass_min_eig": naive,
                        "pulled_scaled_mass_min_eig": pulled,
                        "mass_floor_relief": float(pulled / max(naive, 1.0e-300)),
                        "mass_condition": mass_condition(ops),
                        "scaled_advection_radius": scaled_advection_radius(ops),
                        "green_residual": max(rx, ry, rz),
                        "poly_reproduction_residual": poly,
                    }
                )

    def _agg(key: str, fn) -> float:
        return float(fn(r[key] for r in rows))

    summary = {
        "n_elements": n_elements,
        "degrees": list(degrees),
        "fractions": list(fractions),
        "n_cases": len(rows),
        "min_volume_fraction": _agg("volume_fraction", min),
        "min_naive_scaled_mass_min_eig": _agg("naive_scaled_mass_min_eig", min),
        "min_pulled_scaled_mass_min_eig": _agg("pulled_scaled_mass_min_eig", min),
        "max_mass_floor_relief": _agg("mass_floor_relief", max),
        "max_mass_condition": _agg("mass_condition", max),
        "max_scaled_advection_radius": _agg("scaled_advection_radius", max),
        "max_green_residual": _agg("green_residual", max),
        "max_poly_reproduction_residual": _agg("poly_reproduction_residual", max),
    }
    if write:
        import csv
        import json

        RESULTS.mkdir(parents=True, exist_ok=True)
        with (RESULTS / "small_cell_3d_sweep.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        (RESULTS / "small_cell_3d_sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        (RESULTS / "small_cell_3d_sweep_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    return {"rows": rows, "summary": summary}


def _axis_cut_quadrature(basis: SplineBasis1D, left: float):
    """1D Gauss quadrature on the cut interval [left, 1] with basis/grad values."""
    p = basis.degree
    q = max(8, p + 4)
    xg, wg = np.polynomial.legendre.leggauss(q)
    nodes_list, w_list = [], []
    panels = _quadrature_panels(basis.knots, left, 1.0)
    for lo, hi in zip(panels[:-1], panels[1:]):
        if hi <= lo:
            continue
        nodes_list.append(0.5 * (hi - lo) * xg + 0.5 * (hi + lo))
        w_list.append(0.5 * (hi - lo) * wg)
    x = np.concatenate(nodes_list)
    w = np.concatenate(w_list)
    V = _eval_all(basis.basis, x)
    dV = _eval_all(basis.dbasis, x)
    return x, w, V, dV


def _manufactured_3d():
    """Smooth non-polynomial manufactured solution u and forcing f = -Lap u."""
    import math

    k = math.pi

    def u(x, y, z):
        return np.exp(0.6 * x) * np.cos(k * y) * np.sin(k * z)

    def f(x, y, z):  # -Lap u
        lap = (0.36 - 2.0 * k * k) * np.exp(0.6 * x) * np.cos(k * y) * np.sin(k * z)
        return -lap

    return u, f


def solve_poisson_3d(
    n_elements: int = 6,
    degree: int = 3,
    cut_fractions: Sequence[float] = (0.6, 0.6, 0.6),
    beta_factor: float = 12.0,
) -> dict:
    """Genuine 3D cut-box Poisson solve with symmetric weak-penalty Dirichlet data.

    Solves ``-Lap u = f`` on the tensor cut box with a smooth non-polynomial
    manufactured solution, imposing the exact Dirichlet data on all six cut-box
    faces by symmetric weak-penalty.  Returns the relative ``L^2`` and ``H^1`` errors
    measured against the analytic solution on the cut-domain quadrature.
    """
    basis = make_open_uniform_basis(n_elements=n_elements, degree=degree)
    h = basis.h
    fr = tuple(float(f) for f in cut_fractions)
    lefts = tuple(h * f for f in fr)
    u_fn, f_fn = _manufactured_3d()

    # Per-axis cut operators, extensions, quadrature, and face traces.
    axes = []
    for left in lefts:
        m = assemble_cut_matrices(basis, left=left)
        agg = polynomial_preserving_extension(basis, m, preserve_degree=degree)
        x, w, V, dV = _axis_cut_quadrature(basis, left)
        tL = _eval_all(basis.basis, np.array([left])).ravel()
        dL = _eval_all(basis.dbasis, np.array([left])).ravel()
        tR = _eval_all(basis.basis, np.array([1.0])).ravel()
        dR = _eval_all(basis.dbasis, np.array([1.0])).ravel()
        axes.append(dict(m=m, E=agg.E, x=x, w=w, V=V, dV=dV, tL=tL, dL=dL, tR=tR, dR=dR))
    ax, ay, az = axes
    E = _kron3(az["E"], ay["E"], ax["E"])

    # Pulled-back stiffness.
    K = _kron3(az["m"].H, ay["m"].H, ax["m"].K)
    K = K + _kron3(az["m"].H, ay["m"].K, ax["m"].H)
    K = K + _kron3(az["m"].K, ay["m"].H, ax["m"].H)
    K = E.T @ K @ E

    beta = beta_factor * (degree + 1) ** 2 / h

    # Face weak-penalty operators. For each axis the low face uses outward normal -1
    # and the high face +1; the other two axes carry their full 1D mass.
    def face_ops(axis_idx: int):
        a = axes[axis_idx]
        others = [axes[j]["m"].H for j in range(3) if j != axis_idx]
        Mo = others  # 1D mass matrices on the in-face axes
        M_total = np.zeros((E.shape[0], E.shape[0]))
        N_total = np.zeros((E.shape[0], E.shape[0]))
        for trace, dtrace, sign in ((a["tL"], a["dL"], -1.0), (a["tR"], a["dR"], +1.0)):
            Mface_axis = np.outer(trace, trace)
            Nface_axis = np.outer(trace, sign * dtrace)  # entry (i,j): phi_i * dn phi_j
            parts_M = [None, None, None]
            parts_N = [None, None, None]
            parts_M[axis_idx] = Mface_axis
            parts_N[axis_idx] = Nface_axis
            oi = [j for j in range(3) if j != axis_idx]
            parts_M[oi[0]] = axes[oi[0]]["m"].H
            parts_M[oi[1]] = axes[oi[1]]["m"].H
            parts_N[oi[0]] = axes[oi[0]]["m"].H
            parts_N[oi[1]] = axes[oi[1]]["m"].H
            M_total += _kron3(parts_M[2], parts_M[1], parts_M[0])
            N_total += _kron3(parts_N[2], parts_N[1], parts_N[0])
        return E.T @ M_total @ E, E.T @ N_total @ E

    Mb = np.zeros((E.shape[1], E.shape[1]))
    Nb = np.zeros((E.shape[1], E.shape[1]))
    for axis_idx in range(3):
        Mf, Nf = face_ops(axis_idx)
        Mb += Mf
        Nb += Nf

    A = K - Nb - Nb.T + beta * Mb
    A = 0.5 * (A + A.T)

    # Volume RHS  int f phi  by tensor contraction.
    fx, fy, fz = ax["x"], ay["x"], az["x"]
    Fgrid = f_fn(fx[:, None, None], fy[None, :, None], fz[None, None, :])
    # b_xyz[ix,iy,iz] = int f phi; the kron DOF order is x-fastest -> (z,y,x).
    b_xyz = np.einsum("ip,jq,kr,pqr->ijk", ax["V"] * ax["w"], ay["V"] * ay["w"], az["V"] * az["w"], Fgrid)
    rhs = E.T @ np.transpose(b_xyz, (2, 1, 0)).reshape(-1)

    # Boundary RHS  int_F g (beta phi - dn phi)  on each face, where g = u.
    # face_coef[i,j] = int_F g phi_i phi_j over the two in-face axes (global
    # order); axis_vec carries the (beta*trace - sign*dn-trace) factor on the
    # face axis.  The (x,y,z) tensor is then flattened with x fastest -> (z,y,x).
    def face_rhs(axis_idx: int) -> np.ndarray:
        oi = [j for j in range(3) if j != axis_idx]
        Vo0, wo0, xo0 = axes[oi[0]]["V"], axes[oi[0]]["w"], axes[oi[0]]["x"]
        Vo1, wo1, xo1 = axes[oi[1]]["V"], axes[oi[1]]["w"], axes[oi[1]]["x"]
        a = axes[axis_idx]
        out = np.zeros(E.shape[0])
        for trace, dtrace, sign, coord in (
            (a["tL"], a["dL"], -1.0, lefts[axis_idx]),
            (a["tR"], a["dR"], +1.0, 1.0),
        ):
            cg = [None, None, None]
            cg[axis_idx] = np.array([coord])
            cg[oi[0]] = xo0
            cg[oi[1]] = xo1
            G = u_fn(cg[0][:, None, None], cg[1][None, :, None], cg[2][None, None, :])
            G2 = np.squeeze(G, axis=axis_idx)  # -> (n_oi0, n_oi1) in global order
            face_coef = np.einsum("ip,jq,pq->ij", Vo0 * wo0, Vo1 * wo1, G2)
            axis_vec = beta * trace - sign * dtrace
            if axis_idx == 0:
                full_xyz = np.einsum("a,bc->abc", axis_vec, face_coef)
            elif axis_idx == 1:
                full_xyz = np.einsum("b,ac->abc", axis_vec, face_coef)
            else:
                full_xyz = np.einsum("c,ab->abc", axis_vec, face_coef)
            out += np.transpose(full_xyz, (2, 1, 0)).reshape(-1)
        return out

    for axis_idx in range(3):
        rhs += E.T @ face_rhs(axis_idx)

    coeff = np.linalg.solve(A, rhs)
    alpha = E @ coeff

    # Errors on the cut-domain tensor quadrature.
    Vx, Vy, Vz = ax["V"], ay["V"], az["V"]
    dVx, dVy, dVz = ax["dV"], ay["dV"], az["dV"]
    A3 = alpha.reshape(basis.n_basis, basis.n_basis, basis.n_basis)  # (z,y,x) order
    # uh(xq,yq,zq):  sum_ijk alpha[z,y,x] Vx[x,qx] Vy[y,qy] Vz[z,qz]
    uh = np.einsum("zyx,xp,yq,zr->pqr", A3, Vx, Vy, Vz)
    uhx = np.einsum("zyx,xp,yq,zr->pqr", A3, dVx, Vy, Vz)
    uhy = np.einsum("zyx,xp,yq,zr->pqr", A3, Vx, dVy, Vz)
    uhz = np.einsum("zyx,xp,yq,zr->pqr", A3, Vx, Vy, dVz)
    W = np.einsum("p,q,r->pqr", ax["w"], ay["w"], az["w"])
    Xg = fx[:, None, None]; Yg = fy[None, :, None]; Zg = fz[None, None, :]
    ue = u_fn(Xg, Yg, Zg)
    import math

    k = math.pi
    ux_e = 0.6 * np.exp(0.6 * Xg) * np.cos(k * Yg) * np.sin(k * Zg) + 0.0 * (Yg + Zg)
    uy_e = -k * np.exp(0.6 * Xg) * np.sin(k * Yg) * np.sin(k * Zg)
    uz_e = k * np.exp(0.6 * Xg) * np.cos(k * Yg) * np.cos(k * Zg)
    l2 = float(np.sqrt(np.sum(W * (uh - ue) ** 2)))
    l2_ref = float(np.sqrt(np.sum(W * ue**2)))
    h1 = float(np.sqrt(np.sum(W * ((uhx - ux_e) ** 2 + (uhy - uy_e) ** 2 + (uhz - uz_e) ** 2))))
    h1_ref = float(np.sqrt(np.sum(W * (ux_e**2 + uy_e**2 + uz_e**2))))
    # Relative maximum (L-infinity) error over the fluid quadrature nodes: the
    # pointwise worst case on the cut domain, normalized by the analytic peak.
    linf = float(np.max(np.abs(uh - ue)))
    linf_ref = float(np.max(np.abs(ue)))
    eigs = np.linalg.eigvalsh(A)
    return {
        "n_elements": n_elements,
        "degree": degree,
        "h": h,
        "n_phys": int(E.shape[1]),
        "relative_l2_error": l2 / max(l2_ref, 1.0e-300),
        "relative_h1_error": h1 / max(h1_ref, 1.0e-300),
        "relative_linf_error": linf / max(linf_ref, 1.0e-300),
        "spd_min_eig": float(eigs[0]),
    }


def run_poisson_3d_convergence(
    degree: int = 3,
    elements: Sequence[int] = (4, 6, 8),
    cut_fractions: Sequence[float] = (0.6, 0.6, 0.6),
    write: bool = True,
) -> dict:
    """Degree-p convergence of the 3D cut-box Poisson solve."""
    rows = [solve_poisson_3d(n_elements=n, degree=degree, cut_fractions=cut_fractions) for n in elements]
    hs = np.array([r["h"] for r in rows])
    l2 = np.array([r["relative_l2_error"] for r in rows])
    h1 = np.array([r["relative_h1_error"] for r in rows])

    def slope(err):
        lo = np.log(hs)
        return float(np.polyfit(lo, np.log(err), 1)[0])

    summary = {
        "degree": degree,
        "elements": list(elements),
        "cut_fractions": list(cut_fractions),
        "finest_relative_l2_error": float(rows[-1]["relative_l2_error"]),
        "finest_relative_h1_error": float(rows[-1]["relative_h1_error"]),
        "min_spd_eig": min(float(r["spd_min_eig"]) for r in rows),
        "l2_slope": slope(l2),
        "h1_slope": slope(h1),
    }
    if write:
        import json

        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "poisson_3d_solve.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        (RESULTS / "poisson_3d_solve_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"rows": rows, "summary": summary}


def write_3d_tables(n_elements: int = 8) -> None:
    """Emit the 3D small-cell relief, frontier, and Poisson convergence tables."""
    paper = ROOT / "paper"
    paper.mkdir(parents=True, exist_ok=True)

    # Small-cell relief table: corner sliver, degree 3, sweeping the volume ratio.
    rel_lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"corner sliver & naive $\lambda_{\min}(H)/h^3$ & pulled $\lambda_{\min}(H)/h^3$ & "
        r"relief & Green resid. & poly. resid. \\",
        r"\midrule",
    ]
    for frac in (0.5, 0.9, 0.99, 0.999):
        ops = assemble_cut_3d(n_elements=n_elements, degree=3, cut_fractions=(frac, frac, frac))
        rx, ry, rz = green_residuals_3d(ops)
        poly = polynomial_reproduction_residual_3d(n_elements, 3, (frac, frac, frac))
        pulled = scaled_mass_min_eig(ops)
        naive = ops.naive_scaled_mass_min_eig
        rel_lines.append(
            f"$(1{{-}}{frac:g})^3={ops.volume_fraction:.0e}$ & {naive:.1e} & {pulled:.2e} & "
            f"{pulled / max(naive, 1e-300):.1e} & {max(rx, ry, rz):.1e} & {poly:.1e} \\\\"
        )
    rel_lines += [r"\bottomrule", r"\end{tabular}", ""]
    (paper / "generated_cut_3d_small_cell_table.tex").write_text("\n".join(rel_lines), encoding="utf-8")

    # High-degree extreme-cut frontier table.
    fr = run_high_degree_frontier(degrees=(2, 3, 4, 5, 6), n_elements=n_elements, write=True)
    fr_lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"degree $p$ & 2 & 3 & 4 & 5 & 6 \\",
        r"\midrule",
    ]
    ratios = [fr["frontier"][p]["min_reproduced_sliver_ratio"] for p in (2, 3, 4, 5, 6)]
    fr_lines.append("min reproduced sliver ratio & " + " & ".join(f"{r:.0e}" for r in ratios) + r" \\")
    fr_lines += [r"\bottomrule", r"\end{tabular}", ""]
    (paper / "generated_cut_3d_frontier_table.tex").write_text("\n".join(fr_lines), encoding="utf-8")

    # 3D Poisson convergence table.
    p3 = run_poisson_3d_convergence(degree=3, elements=(4, 6, 8), write=True)
    pv_lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"$h$ & rel. $L^2$ error & rel. $H^1$ error & $\lambda_{\min}(A)$ \\",
        r"\midrule",
    ]
    for r in p3["rows"]:
        pv_lines.append(
            f"$1/{r['n_elements']}$ & {r['relative_l2_error']:.3e} & "
            f"{r['relative_h1_error']:.3e} & {r['spd_min_eig']:.2e} \\\\"
        )
    pv_lines.append(r"\midrule")
    pv_lines.append(
        f"slope & {p3['summary']['l2_slope']:.2f} & {p3['summary']['h1_slope']:.2f} & \\\\"
    )
    pv_lines += [r"\bottomrule", r"\end{tabular}", ""]
    (paper / "generated_poisson_3d_table.tex").write_text("\n".join(pv_lines), encoding="utf-8")


def write_3d_figure(n_elements: int = 8) -> None:
    """Three-panel 3D evidence figure: mass-floor relief, frontier, convergence."""
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    FIG = ROOT / "figures"
    FIG.mkdir(parents=True, exist_ok=True)
    blue, red, teal, green = "#22517a", "#a83232", "#1f6f6f", "#1c7c34"

    fracs = [0.5, 0.9, 0.99, 0.999, 0.9999]
    naive, pulled, vols = [], [], []
    for frac in fracs:
        ops = assemble_cut_3d(n_elements=n_elements, degree=3, cut_fractions=(frac, frac, frac))
        naive.append(max(ops.naive_scaled_mass_min_eig, 1e-300))
        pulled.append(scaled_mass_min_eig(ops))
        vols.append(ops.volume_fraction)

    fr = run_high_degree_frontier(degrees=(2, 3, 4, 5, 6), n_elements=n_elements, write=False)
    degs = [2, 3, 4, 5, 6]
    ratios = [fr["frontier"][p]["min_reproduced_sliver_ratio"] for p in degs]

    p3 = run_poisson_3d_convergence(degree=3, elements=(4, 6, 8), write=False)
    hs = [r["h"] for r in p3["rows"]]
    l2 = [r["relative_l2_error"] for r in p3["rows"]]
    h1 = [r["relative_h1_error"] for r in p3["rows"]]

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.4))
    a, b, c = axes
    a.loglog(vols, naive, "o-", color=red, label="naive active floor")
    a.loglog(vols, pulled, "s-", color=blue, label="pulled-back floor")
    a.set_xlabel("corner sliver volume ratio")
    a.set_ylabel(r"$\lambda_{\min}(H)/h^3$")
    a.set_title("(a) 3D small-cell mass-floor relief", fontsize=10)
    a.legend(frameon=False, fontsize=8)
    a.invert_xaxis()

    b.semilogy(degs, ratios, "D-", color=teal)
    b.set_xlabel("degree $p$")
    b.set_ylabel("min reproduced sliver ratio")
    b.set_title("(b) high-degree cut frontier", fontsize=10)
    b.set_xticks(degs)

    c.loglog(hs, l2, "o-", color=blue, label=fr"$L^2$ (slope {p3['summary']['l2_slope']:.2f})")
    c.loglog(hs, h1, "s-", color=green, label=fr"$H^1$ (slope {p3['summary']['h1_slope']:.2f})")
    c.set_xlabel("$h$")
    c.set_ylabel("relative error")
    c.set_title("(c) 3D cut-box Poisson convergence", fontsize=10)
    c.legend(frameon=False, fontsize=8)
    c.invert_xaxis()

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(FIG / f"fig_cut_3d_evidence.{suffix}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    import json

    out = run_small_cell_3d_sweep()
    print("3D small-cell sweep:")
    print(json.dumps(out["summary"], indent=2))
    frontier = run_high_degree_frontier()
    print("high-degree extreme-cut frontier:")
    print(json.dumps(frontier, indent=2))
    poisson = run_poisson_3d_convergence()
    print("3D Poisson convergence:")
    print(json.dumps(poisson["summary"], indent=2))
    write_3d_tables()
    write_3d_figure()
    print("3D tables and figure written")
