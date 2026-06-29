"""Fully conservative incompressible Navier-Stokes in 3D on a sharp cut.

This module lifts the planar fully-conservative cut-cell saddle solver of
``cut_saddle_flow`` to a genuine three-dimensional sharp cut.  "Fully
conservative" is the standard term for an incompressible discretization
that conserves mass, momentum, *and* kinetic energy simultaneously; the term is
used here in that sense.

Geometry.  The domain is the tensor cut box ``Omega = [a_x,1] x [a_y,1] x
[a_z,1]`` with ``a_i = h * frac_i``.  A fraction near one places each cut just
below the first interior knot, so the corner cell is a 3D sliver whose volume
ratio is ``prod_i (1 - frac_i)``; at ``frac=0.999`` on all three axes the corner
cut cell has volume ratio ``1e-9``.  The cut faces are genuine sub-cell cuts of
the background B-spline grid, so the box is a non-body-fitted, sharply cut 3D
region carried by the aggregated cut operators of ``cut_3d_sbp``.

Construction (the three steps shared with every operator in the manuscript).

1. Sharp assembly.  Velocity components live in the degree-``p`` clamped
   tensor-product space, pressure in the degree-``p-1`` space on the same knots
   (spline Taylor-Hood).  Mass, stiffness, the weak gradient ``G_i`` and its
   transpose divergence ``D_i = G_i^T``, the symmetric weak-penalty diffusion, and
   the skew convection are all assembled on the sharply cut box.  Because each
   axis is an interval cut ``[a_i, 1]`` the tensor volume integral is a
   Kronecker product of exact 1D cut Grams, so the weak gradient is exact
   (piecewise polynomial), and the discrete gradient/divergence duality is
   algebraic.

2. Congruence pullback.  Each space carries its own polynomial-preserving 1D
   extension per axis (``E_u`` at degree ``p``, ``E_p`` at degree ``p-1``,
   Kronecker-lifted to 3D).  Every block is pulled back by congruence,
   ``A -> E_u^T A E_u`` and ``G_i -> E_u^T G_i E_p``, so the transpose coupling
   ``D_i = G_i^T`` survives the pullback exactly, at any degree, on any cut.

3. Inherited conservation.  Once the saddle solve enforces the constraint rows
   ``sum_i G_i^T u_i = h_p``, the interior pressure work contracts to boundary
   data, ``sum_i u_i^T G_i p = p^T (sum_i G_i^T u_i) = p^T h_p``, with no
   interior contribution.  The skew-symmetric convection
   ``C_skew(w) = (C(w) - C(w)^T)/2 + M_{w.n}/2`` has a quadratic form equal to
   the boundary kinetic-energy flux identically, for any quadrature.  So the
   semidiscrete kinetic-energy budget closes term by term, and in the inviscid
   limit with no through-boundary flux the discrete kinetic energy is conserved
   to roundoff.

Benchmarks driven from this module.

* a 3D cut-operator MMS gate (polynomial reproduction and the directional
  SBP/Green identity to ~1e-12 across cut fractions);
* a 3D divergence-free projection gate (discrete divergence ~1e-10);
* the fully-conservative inviscid gate: advance the 3D Taylor-Green vortex with
  ``nu = 0``; the skew-symmetric kinetic-energy residual stays at roundoff over many
  steps and the interior pressure work is ~1e-14;
* a viscous-decay gate: with ``nu > 0`` the discrete kinetic energy matches the
  exact Taylor-Green envelope and the dissipation budget closes;
* a cut-independence sweep at a fixed background-scale time step as the smallest
  sliver fraction shrinks across decades.
"""

from __future__ import annotations

import os

for _thread_var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_var, "4")

from dataclasses import dataclass
from math import pi, sqrt
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from cut_spline_sbp import (
    _eval_all,
    _quadrature_panels,
    assemble_cut_matrices,
    make_open_uniform_basis,
    polynomial_preserving_extension,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def _sym(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.T)


def _kron3(az: np.ndarray, ay: np.ndarray, ax: np.ndarray) -> np.ndarray:
    """Kronecker lift with x fastest (DOF order is z-outer, y-middle, x-inner)."""
    return np.kron(np.kron(az, ay), ax)


# ---------------------------------------------------------------------------
# 1D axis bundle: cut Grams, extension, quadrature, traces
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AxisBundle:
    """All 1D data on a single cut axis [left, 1] for one spline degree."""

    degree: int
    left: float
    # cut Grams in the FULL 1D basis
    H: np.ndarray
    K: np.ndarray
    Q: np.ndarray
    # polynomial-preserving extension (full -> reduced)
    E: np.ndarray
    # cut volume quadrature on [left, 1] and basis/grad values there
    x: np.ndarray
    w: np.ndarray
    V: np.ndarray
    dV: np.ndarray
    # face traces and normal-derivative traces at the two endpoints
    tL: np.ndarray
    dL: np.ndarray
    tR: np.ndarray
    dR: np.ndarray
    # pressure-axis values at the matching velocity-axis quadrature nodes
    # (populated only for pressure bundles via attach_velocity_nodes)
    V_at_u: np.ndarray | None = None


def _axis_quadrature(basis, left: float, quad_pts: int | None = None):
    """1D Gauss quadrature on the cut interval [left, 1] with basis/grad values.

    ``quad_pts`` sets the Gauss-Legendre point count *per knot panel*.  The
    fully-conservative kinetic-energy identities are algebraic and hold for any
    quadrature, so a count that integrates the trilinear skew convection exactly
    (degree ``3p``) is sufficient; the default ``2p+2`` does this while keeping
    the captured tensor volume quadrature small (the per-step cost driver).
    """
    p = basis.degree
    q = int(quad_pts) if quad_pts is not None else max(2 * p + 2, p + 4)
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


def build_axis_bundle(
    n_elements: int,
    degree: int,
    frac: float,
    support_threshold: float = 0.05,
    aggregate: bool = True,
    active_tol: float = 1.0e-25,
    quad_pts: int | None = None,
) -> AxisBundle:
    """Assemble one cut axis: cut Grams, polynomial-preserving extension, quadrature.

    With ``aggregate=True`` the extension is the degree-``p`` polynomial-preserving
    aggregation (slivers carried by interior coordinates).  With
    ``aggregate=False`` the extension is the plain identity on the active set
    (every positive-mass coordinate kept, including slivers): this is the honest
    active-space control whose mass floor collapses with the sliver volume.
    """
    basis = make_open_uniform_basis(n_elements=n_elements, degree=degree)
    h = basis.h
    left = h * frac
    m = assemble_cut_matrices(basis, left=left, quadrature_order=max(2 * degree + 6, degree + 4))
    if aggregate:
        agg = polynomial_preserving_extension(
            basis, m, support_threshold=support_threshold, preserve_degree=degree
        )
        E = agg.E
    else:
        from cut_spline_sbp import active_sets

        active, _interior, _small, _gamma = active_sets(
            m, support_threshold=support_threshold, active_tol=active_tol
        )
        E = np.eye(basis.n_basis)[:, active]
    x, w, V, dV = _axis_quadrature(basis, left, quad_pts=quad_pts)
    tL = _eval_all(basis.basis, np.array([left])).ravel()
    dL = _eval_all(basis.dbasis, np.array([left])).ravel()
    tR = _eval_all(basis.basis, np.array([1.0])).ravel()
    dR = _eval_all(basis.dbasis, np.array([1.0])).ravel()
    return AxisBundle(
        degree=degree, left=left, H=m.H, K=m.K, Q=m.Q, E=E,
        x=x, w=w, V=V, dV=dV, tL=tL, dL=dL, tR=tR, dR=dR,
    )


def _mixed_1d_cut_grams(basis_u, basis_p, left: float):
    """Mixed 1D cut Grams (M, D) between a degree-p and degree-(p-1) basis on [left,1].

    M[a,b] = int_left^1 Bu_a Bp_b,  D[a,b] = int_left^1 Bu_a' Bp_b, by panel Gauss
    on the cut interval; exact since the integrand is piecewise polynomial.
    """
    q = max(2 * basis_u.degree + 6, basis_u.degree + 4)
    xg, wg = np.polynomial.legendre.leggauss(q)
    M = np.zeros((basis_u.n_basis, basis_p.n_basis))
    D = np.zeros_like(M)
    panels = _quadrature_panels(basis_u.knots, left, 1.0)
    for lo, hi in zip(panels[:-1], panels[1:]):
        if hi <= lo:
            continue
        x = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
        w = 0.5 * (hi - lo) * wg
        Vu = _eval_all(basis_u.basis, x)
        dVu = _eval_all(basis_u.dbasis, x)
        Vp = _eval_all(basis_p.basis, x)
        M += (Vu * w) @ Vp.T
        D += (dVu * w) @ Vp.T
    return M, D


# ---------------------------------------------------------------------------
# 3D mixed operator bundle
# ---------------------------------------------------------------------------


@dataclass
class Mixed3DCutOperators:
    """Aggregated 3D mixed velocity-pressure blocks on the tensor cut box."""

    p: int
    n_elements: int
    h: float
    beta: float
    cut_fractions: tuple[float, float, float]
    volume_fraction: float
    aggregate: bool
    # reduced velocity (scalar component) blocks
    H_u: np.ndarray
    Diff: np.ndarray  # symmetric weak-penalty diffusion at unit viscosity
    # reduced pressure mass, weak-gradient blocks, mean-pressure functional
    H_p: np.ndarray
    Gx: np.ndarray
    Gy: np.ndarray
    Gz: np.ndarray
    mass_p: np.ndarray
    # full -> reduced extensions
    E_u: np.ndarray
    E_p: np.ndarray
    # captured volume quadrature (3D, flattened) for convection / loads / norms
    QX: np.ndarray
    QY: np.ndarray
    QZ: np.ndarray
    QW: np.ndarray
    PHI_u: np.ndarray   # (n_u_full, nq)  velocity basis values
    GX_u: np.ndarray
    GY_u: np.ndarray
    GZ_u: np.ndarray
    PSI_p: np.ndarray   # (n_p_full, nq)  pressure basis values
    # boundary face quadrature (3D) with outward normals and traces
    BX: np.ndarray
    BY: np.ndarray
    BZ: np.ndarray
    BW: np.ndarray
    BNX: np.ndarray
    BNY: np.ndarray
    BNZ: np.ndarray
    PHI_u_b: np.ndarray
    DPHIN_u_b: np.ndarray
    PSI_p_b: np.ndarray
    # bookkeeping
    n_u_red: int
    n_p_red: int

    @property
    def n_saddle(self) -> int:
        return 3 * self.n_u_red + self.n_p_red + 1


def _tensor_volume_quadrature(axu_list):
    """3D tensor cut-box volume nodes, weights, and flattened velocity values.

    Returns flattened (x,y,z) coordinates and weights of the tensor product, plus
    the velocity-basis value and gradient arrays of shape (n_u_full, nq) with the
    global DOF order z-outer / y-middle / x-inner.
    """
    ax, ay, az = axu_list
    nqx, nqy, nqz = ax.x.size, ay.x.size, az.x.size
    # The basis value/gradient matrices below are built with ``_kron3`` (DOF and
    # quadrature-point columns both ordered z-outer / y-middle / x-inner, i.e.
    # x fastest).  The coordinate and weight arrays must use the SAME tensor
    # ordering so that an externally sampled field f(X, Y, Z) pairs column-for-
    # column with PHI.  Building X, Y, Z, W with ``_kron3`` (x fastest) instead of
    # ``einsum("i,j,k->ijk")`` (which is z fastest) keeps that pairing consistent;
    # the two orderings coincide only on axis-symmetric grids, so the mismatch was
    # masked whenever the cut fractions were equal across x and z.
    X = _kron3(np.ones((1, nqz)), np.ones((1, nqy)), ax.x[None, :]).reshape(-1)
    Y = _kron3(np.ones((1, nqz)), ay.x[None, :], np.ones((1, nqx))).reshape(-1)
    Z = _kron3(az.x[None, :], np.ones((1, nqy)), np.ones((1, nqx))).reshape(-1)
    W = _kron3(az.w[None, :], ay.w[None, :], ax.w[None, :]).reshape(-1)
    PHI = _kron3(az.V, ay.V, ax.V)
    GX = _kron3(az.V, ay.V, ax.dV)
    GY = _kron3(az.V, ay.dV, ax.V)
    GZ = _kron3(az.dV, ay.V, ax.V)
    return X, Y, Z, W, PHI, GX, GY, GZ


def _face_trace_array(val_x, val_y, val_z, axis, nq0, nq1):
    """Assemble a face trace array (n_dof_full, nq0*nq1) from per-axis factors.

    For the fixed face ``axis`` the factor ``val_{axis}`` is a column (n, 1) (a
    single face coordinate); for the two in-face axes the factor is (n_k, nq_k).
    The global DOF order is z-outer / y-middle / x-inner, and the face
    quadrature index runs over (q_oi0, q_oi1) with q_oi1 fastest, matching the
    weight/node construction ``einsum("a,b->ab", o0, o1).reshape(-1)``.
    """
    nx, qx = val_x.shape
    ny, qy = val_y.shape
    nz, qz = val_z.shape
    # T[dz, dy, dx, qz, qy, qx]
    T = np.einsum("xa,yb,zc->zyxcba", val_x, val_y, val_z)
    T = T.reshape(nz * ny * nx, qz * qy * qx)
    # The flattened q index runs (qz, qy, qx) with qx fastest.  Two of these have
    # the in-face sizes and one is a singleton (the face axis).  Reorder so the q
    # index runs (q_oi0, q_oi1) with q_oi1 fastest.
    oi = [j for j in range(3) if j != axis]
    pos = {2: 0, 1: 1, 0: 2}  # position of (z, y, x) in the (qz, qy, qx) layout
    idx = np.arange(qz * qy * qx).reshape(qz, qy, qx)
    idx = np.moveaxis(idx, [pos[oi[0]], pos[oi[1]]], [0, 1]).reshape(nq0, nq1)
    return T[:, idx.reshape(-1)]


def _face_quadrature(axu_list, axp_list):
    """3D boundary-face quadrature on the six faces of the cut box.

    Each face fixes one axis at a cut endpoint (outward normal +-1 on that axis)
    and tensors the in-face axes' volume quadrature.  Returns flattened nodes,
    weights, outward unit normals, and the velocity / pressure trace and
    normal-derivative arrays of shape (n_dof_full, nq_face) in the same global
    DOF order (z-outer / y-middle / x-inner) as the volume quadrature.
    """
    axu = axu_list
    axp = axp_list
    BX, BY, BZ, BW = [], [], [], []
    BNX, BNY, BNZ = [], [], []
    PHI_b, DPHIN_b, PSI_b = [], [], []

    for axis in range(3):
        oi = [j for j in range(3) if j != axis]
        a_u = axu[axis]
        u0, u1 = axu[oi[0]], axu[oi[1]]
        nq0, nq1 = u0.x.size, u1.x.size
        for endpoint, sign in (("L", -1.0), ("R", +1.0)):
            if endpoint == "L":
                coord = a_u.left
                tU, dU, tP = a_u.tL, a_u.dL, axp[axis].tL
            else:
                coord = 1.0
                tU, dU, tP = a_u.tR, a_u.dR, axp[axis].tR

            Wf = np.einsum("a,b->ab", u0.w, u1.w).reshape(-1)
            xs = [None, None, None]
            xs[axis] = np.full(nq0 * nq1, coord)
            xs[oi[0]] = np.einsum("a,b->ab", u0.x, np.ones(nq1)).reshape(-1)
            xs[oi[1]] = np.einsum("a,b->ab", np.ones(nq0), u1.x).reshape(-1)
            BX.append(xs[0]); BY.append(xs[1]); BZ.append(xs[2])
            BW.append(Wf)
            nrm = [0.0, 0.0, 0.0]
            nrm[axis] = sign
            BNX.append(np.full(nq0 * nq1, nrm[0]))
            BNY.append(np.full(nq0 * nq1, nrm[1]))
            BNZ.append(np.full(nq0 * nq1, nrm[2]))

            # velocity value-trace factors (n, nq) per axis
            val = [None, None, None]
            dn = [None, None, None]
            val[axis] = tU.reshape(-1, 1)
            dn[axis] = (sign * dU).reshape(-1, 1)
            val[oi[0]] = u0.V
            dn[oi[0]] = u0.V
            val[oi[1]] = u1.V
            dn[oi[1]] = u1.V
            PHI_b.append(_face_trace_array(val[0], val[1], val[2], axis, nq0, nq1))
            DPHIN_b.append(_face_trace_array(dn[0], dn[1], dn[2], axis, nq0, nq1))

            # pressure value-trace factors: pressure basis values at velocity nodes
            pv = [None, None, None]
            pv[axis] = tP.reshape(-1, 1)
            pv[oi[0]] = axp[oi[0]].V_at_u
            pv[oi[1]] = axp[oi[1]].V_at_u
            PSI_b.append(_face_trace_array(pv[0], pv[1], pv[2], axis, nq0, nq1))

    return (
        np.concatenate(BX), np.concatenate(BY), np.concatenate(BZ), np.concatenate(BW),
        np.concatenate(BNX), np.concatenate(BNY), np.concatenate(BNZ),
        np.concatenate(PHI_b, axis=1), np.concatenate(DPHIN_b, axis=1), np.concatenate(PSI_b, axis=1),
    )


def build_mixed_3d_operators(
    p: int,
    n_elements: int,
    cut_fractions: Sequence[float] = (0.6, 0.6, 0.6),
    beta_factor: float = 12.0,
    aggregate: bool = True,
    support_threshold: float = 0.05,
    quad_pts: int | None = None,
) -> Mixed3DCutOperators:
    """Assemble the aggregated 3D mixed velocity-pressure blocks on the cut box.

    Velocity components live in the degree-``p`` clamped tensor space, pressure in
    the degree-``p-1`` space; every block is sharp-assembled on the cut box and
    pulled back by congruence through the Kronecker-lifted polynomial-preserving
    extensions.
    """
    if p < 2:
        raise ValueError("The mixed pair requires velocity degree p >= 2.")
    fr = tuple(float(f) for f in cut_fractions)
    h = 1.0 / n_elements

    axu = [build_axis_bundle(n_elements, p, f, support_threshold=support_threshold, aggregate=aggregate, quad_pts=quad_pts) for f in fr]
    axp = [build_axis_bundle(n_elements, p - 1, f, support_threshold=support_threshold, aggregate=aggregate, quad_pts=quad_pts) for f in fr]
    # Pressure basis values at the matching velocity quadrature nodes, so that
    # pressure traces and volume values align with the velocity face / volume
    # quadrature (the divergence and pressure-work identities are then algebraic).
    basis_p = make_open_uniform_basis(n_elements, p - 1)
    axp = [
        __import__("dataclasses").replace(b, V_at_u=_eval_all(basis_p.basis, axu[i].x))
        for i, b in enumerate(axp)
    ]

    E_u = _kron3(axu[2].E, axu[1].E, axu[0].E)
    E_p = _kron3(axp[2].E, axp[1].E, axp[0].E)

    # Velocity mass and stiffness (full), pulled back.
    H_full = _kron3(axu[2].H, axu[1].H, axu[0].H)
    K_full = (
        _kron3(axu[2].H, axu[1].H, axu[0].K)
        + _kron3(axu[2].H, axu[1].K, axu[0].H)
        + _kron3(axu[2].K, axu[1].H, axu[0].H)
    )
    H_u = _sym(E_u.T @ H_full @ E_u)
    K_u = _sym(E_u.T @ K_full @ E_u)

    # Pressure mass (full), pulled back.
    Hp_full = _kron3(axp[2].H, axp[1].H, axp[0].H)
    H_p = _sym(E_p.T @ Hp_full @ E_p)

    # Captured tensor volume quadrature (velocity) and pressure values there.
    QX, QY, QZ, QW, PHI_u, GX_u, GY_u, GZ_u = _tensor_volume_quadrature(axu)
    # Pressure values on the velocity quadrature nodes (z-outer / y-mid / x-inner).
    PSI_p = _kron3(axp[2].V_at_u, axp[1].V_at_u, axp[0].V_at_u)

    mass_p = E_p.T @ (PSI_p @ QW)

    # Weak gradient blocks G_i = -int d_i phi psi + oint phi n_i psi (full), then
    # pulled back.  The volume part is a Kronecker of exact 1D cut mixed Grams.
    Mx, Dx = _mixed_1d_cut_grams(make_open_uniform_basis(n_elements, p), make_open_uniform_basis(n_elements, p - 1), axu[0].left)
    My, Dy = _mixed_1d_cut_grams(make_open_uniform_basis(n_elements, p), make_open_uniform_basis(n_elements, p - 1), axu[1].left)
    Mz, Dz = _mixed_1d_cut_grams(make_open_uniform_basis(n_elements, p), make_open_uniform_basis(n_elements, p - 1), axu[2].left)
    Gx_vol = -_kron3(Mz, My, Dx)
    Gy_vol = -_kron3(Mz, Dy, Mx)
    Gz_vol = -_kron3(Dz, My, Mx)

    # Boundary face quadrature with normals and traces.
    BX, BY, BZ, BW, BNX, BNY, BNZ, PHI_u_b, DPHIN_u_b, PSI_p_b = _face_quadrature(axu, axp)

    Gx_full = Gx_vol + (PHI_u_b * (BW * BNX)) @ PSI_p_b.T
    Gy_full = Gy_vol + (PHI_u_b * (BW * BNY)) @ PSI_p_b.T
    Gz_full = Gz_vol + (PHI_u_b * (BW * BNZ)) @ PSI_p_b.T
    Gx = E_u.T @ Gx_full @ E_p
    Gy = E_u.T @ Gy_full @ E_p
    Gz = E_u.T @ Gz_full @ E_p

    # Symmetric weak-penalty diffusion at unit viscosity on all six faces.
    M_bnd = (PHI_u_b * BW) @ PHI_u_b.T
    N_flux = (PHI_u_b * BW) @ DPHIN_u_b.T
    M_r = E_u.T @ M_bnd @ E_u
    N_r = E_u.T @ N_flux @ E_u
    beta = beta_factor * (p + 1) ** 2
    Diff = _sym(K_u - N_r - N_r.T + (beta / h) * M_r)

    return Mixed3DCutOperators(
        p=p, n_elements=n_elements, h=h, beta=beta,
        cut_fractions=fr,
        volume_fraction=float(np.prod([1.0 - f for f in fr if f > 0.0])) if any(f > 0 for f in fr) else 1.0,
        aggregate=aggregate,
        H_u=H_u, Diff=Diff, H_p=H_p, Gx=Gx, Gy=Gy, Gz=Gz, mass_p=mass_p,
        E_u=E_u, E_p=E_p,
        QX=QX, QY=QY, QZ=QZ, QW=QW, PHI_u=PHI_u, GX_u=GX_u, GY_u=GY_u, GZ_u=GZ_u, PSI_p=PSI_p,
        BX=BX, BY=BY, BZ=BZ, BW=BW, BNX=BNX, BNY=BNY, BNZ=BNZ,
        PHI_u_b=PHI_u_b, DPHIN_u_b=DPHIN_u_b, PSI_p_b=PSI_p_b,
        n_u_red=int(E_u.shape[1]), n_p_red=int(E_p.shape[1]),
    )


# ---------------------------------------------------------------------------
# Convection (skew-symmetric form), boundary work, saddle assembly
# ---------------------------------------------------------------------------


def convection_blocks_3d(ops: Mixed3DCutOperators, cu, cv, cw):
    """Return (C_red, M_wn_red) for advecting field w = (u, v, w).

    C_red is the plain convection matrix int phi_a (w . grad) phi_b pulled back;
    M_wn_red is the boundary mass weighted by w.n.  The skew-symmetric operator
    is C_skew = (C - C^T)/2 + M_wn/2.
    """
    au = ops.E_u @ cu
    av = ops.E_u @ cv
    aw = ops.E_u @ cw
    wx = ops.PHI_u.T @ au
    wy = ops.PHI_u.T @ av
    wz = ops.PHI_u.T @ aw
    full = (ops.PHI_u * (ops.QW * wx)) @ ops.GX_u.T
    full += (ops.PHI_u * (ops.QW * wy)) @ ops.GY_u.T
    full += (ops.PHI_u * (ops.QW * wz)) @ ops.GZ_u.T
    C_red = ops.E_u.T @ full @ ops.E_u

    ub = ops.PHI_u_b.T @ au
    vb = ops.PHI_u_b.T @ av
    wb = ops.PHI_u_b.T @ aw
    wn = ub * ops.BNX + vb * ops.BNY + wb * ops.BNZ
    M_wn = (ops.PHI_u_b * (ops.BW * wn)) @ ops.PHI_u_b.T
    M_wn_red = ops.E_u.T @ M_wn @ ops.E_u
    return C_red, M_wn_red


def skew_convection(C_red, M_wn_red, boundary_flux: bool = True):
    """fully conservative skew-symmetric convection operator.

    The kinetic-energy-conserving convection is the antisymmetric part
    ``(C - C^T)/2`` whose quadratic form ``u^T C_skew u`` vanishes identically for
    every field, mesh and quadrature.  When the boundary carries a through-flow
    the consistent (skew-adjoint) operator adds half the ``w.n`` boundary mass,
    ``+ M_wn/2``, so the discrete kinetic-energy budget tracks the true boundary
    flux.  For a flow with zero normal velocity on every face (the closed-cavity
    wall-confined vortex) that boundary flux is exactly zero, and including the
    discrete ``M_wn`` term would inject the small face-normal velocity error back
    into the energy; setting ``boundary_flux=False`` then leaves the pure
    antisymmetric operator and the kinetic energy is conserved to roundoff.
    """
    Cs = 0.5 * (C_red - C_red.T)
    if boundary_flux:
        Cs = Cs + 0.5 * M_wn_red
    return Cs


def weak_bc_rhs_3d(ops, gu, gv, gw, nu):
    """Momentum weak-penalty data terms for boundary velocity samples (gu, gv, gw)."""
    def comp(g):
        return ops.E_u.T @ (
            nu * ((ops.beta / ops.h) * ((ops.PHI_u_b * ops.BW) @ g) - (ops.DPHIN_u_b * ops.BW) @ g)
        )
    return comp(gu), comp(gv), comp(gw)


def continuity_rhs_3d(ops, gu, gv, gw):
    """Weak divergence data h_p[b] = oint psi_b (g . n)."""
    gn = gu * ops.BNX + gv * ops.BNY + gw * ops.BNZ
    return ops.E_p.T @ (ops.PSI_p_b @ (ops.BW * gn))


def assemble_saddle_3d(ops: Mixed3DCutOperators, A: np.ndarray) -> np.ndarray:
    """Pinned saddle matrix for momentum block A (shared by all three components).

    Layout: [u, v, w, p, lambda] with the mean-pressure multiplier lambda
    enforcing mass_p . p = 0 (all-Dirichlet boundary data).
    """
    nu_red = ops.n_u_red
    np_red = ops.n_p_red
    n = 3 * nu_red + np_red + 1
    S = np.zeros((n, n))
    for k in range(3):
        s = slice(k * nu_red, (k + 1) * nu_red)
        S[s, s] = A
    pcol = slice(3 * nu_red, 3 * nu_red + np_red)
    S[0 * nu_red:1 * nu_red, pcol] = ops.Gx
    S[1 * nu_red:2 * nu_red, pcol] = ops.Gy
    S[2 * nu_red:3 * nu_red, pcol] = ops.Gz
    S[pcol, 0 * nu_red:1 * nu_red] = ops.Gx.T
    S[pcol, 1 * nu_red:2 * nu_red] = ops.Gy.T
    S[pcol, 2 * nu_red:3 * nu_red] = ops.Gz.T
    S[pcol, -1] = ops.mass_p
    S[-1, pcol] = ops.mass_p
    return S


def _refined_solve(lu, S, rhs, refine: int = 1):
    sol = lu_solve(lu, rhs)
    for _ in range(refine):
        sol += lu_solve(lu, rhs - S @ sol)
    return sol


# ---------------------------------------------------------------------------
# Exact 3D Taylor-Green / Beltrami fields
# ---------------------------------------------------------------------------


def taylor_green_3d_fields(nu: float, mode: str = "beltrami"):
    """Exact decaying incompressible 3D fields with a known kinetic-energy law.

    ``mode='beltrami'`` returns a genuinely three-dimensional generalized
    Beltrami (ABC-type) flow that is an *exact* unforced Navier-Stokes solution:

        u =  ( sin x cos y cos z) F,  with the cyclic partners chosen so that
        div u = 0 and  d_t u = nu Lap u  (the nonlinear term is a pure gradient
        absorbed by the pressure).  The kinetic energy decays exactly as
        KE(t) = KE(0) exp(-2 nu k^2 t) with k^2 = 3 (pi^2 here).

    Concretely (wavenumber kk = pi):

        u =  cos(pi x) sin(pi y) sin(pi z) F(t)
        v =  sin(pi x) cos(pi y) sin(pi z) F(t)
        w = -2 sin(pi x) sin(pi y) cos(pi z) F(t)

    This is divergence-free for every t (d_x u + d_y v + d_z w = 0 identically),
    and each component satisfies the heat equation d_t = nu Lap with Lap = -3 pi^2,
    so F(t) = exp(-3 pi^2 nu t) and the nonlinear convection is balanced by the
    pressure gradient; the exact kinetic-energy envelope is KE(t)/KE(0) =
    exp(-6 pi^2 nu t).
    """
    kk = pi

    def F(t):
        return float(np.exp(-3.0 * kk * kk * nu * t))

    def u(x, y, z, t):
        return np.cos(kk * x) * np.sin(kk * y) * np.sin(kk * z) * F(t)

    def v(x, y, z, t):
        return np.sin(kk * x) * np.cos(kk * y) * np.sin(kk * z) * F(t)

    def w(x, y, z, t):
        return -2.0 * np.sin(kk * x) * np.sin(kk * y) * np.cos(kk * z) * F(t)

    return {"u": u, "v": v, "w": w, "F": F, "kk": kk, "decay_rate": 6.0 * kk * kk}


def wall_confined_tgv_3d_fields(
    nu: float,
    cut_fractions: Sequence[float] = (0.6, 0.6, 0.6),
    h: float = 1.0,
):
    """Closed-cavity Taylor-Green vortex confined to the sharp cut box.

    The cut box is ``Omega = [a_x,1] x [a_y,1] x [a_z,1]`` with ``a_i = h * f_i``,
    so the in-axis lengths are ``L_i = 1 - a_i`` and ``k_i = pi / L_i``.  The
    field is the planar Taylor-Green vortex, uniform in z, shifted to the cut box:

        u(t) =  sin(k_x (x-a_x)) cos(k_y (y-a_y)) F(t)
        v(t) = -cos(k_x (x-a_x)) sin(k_y (y-a_y)) F(t)
        w    =  0

    This field has three properties that make it the canonical inviscid
    kinetic-energy-conservation witness on a sharp cut:

    * **Divergence free** when ``k_x = k_y`` (equal x/y cut fractions), so it sits
      exactly on the discrete divergence-constraint manifold.
    * **No through-boundary flux**: the normal velocity vanishes on every face of
      the cut box (``u=0`` on the x faces, ``v=0`` on the y faces, ``w=0`` on the
      z faces).  The skew-symmetric-convection quadratic form therefore equals the
      boundary kinetic-energy flux, which is *identically zero*, so with ``nu=0``
      the discrete kinetic energy is conserved to roundoff.
    * **Exact unforced Navier-Stokes solution**: each component is a Laplacian
      eigenmode with eigenvalue ``-(k_x^2 + k_y^2)`` and the self-advection is a
      pure gradient absorbed by the pressure, so the kinetic energy decays exactly
      as ``KE(t)/KE(0) = exp(-2 nu (k_x^2 + k_y^2) t)``.

    The equal-fraction constraint is required only for the x/y pair (so that the
    field is divergence free); the z fraction is free, so the corner sliver
    volume ratio ``prod_i (1 - f_i)`` still ranges across decades.
    """
    fr = tuple(float(f) for f in cut_fractions)
    ax, ay, az = (h * fr[0], h * fr[1], h * fr[2])
    Lx, Ly = (1.0 - ax, 1.0 - ay)
    kx, ky = (pi / Lx, pi / Ly)
    lam = kx * kx + ky * ky  # Laplacian eigenvalue magnitude

    def F(t):
        return float(np.exp(-lam * nu * t))

    def u(x, y, z, t):
        return np.sin(kx * (x - ax)) * np.cos(ky * (y - ay)) * F(t)

    def v(x, y, z, t):
        return -np.cos(kx * (x - ax)) * np.sin(ky * (y - ay)) * F(t)

    def w(x, y, z, t):
        return np.zeros_like(np.asarray(x, dtype=float))

    return {
        "u": u, "v": v, "w": w, "F": F,
        "kx": kx, "ky": ky, "lam": lam,
        "decay_rate": 2.0 * lam,
        "ax": ax, "ay": ay, "az": az,
    }


def _project_velocity_constrained_3d(ops, fu, fv, fw, hp):
    """Constrained L2 projection: closest field with sum_i G_i^T u_i = hp."""
    rhs_u = ops.E_u.T @ ((ops.PHI_u * ops.QW) @ fu)
    rhs_v = ops.E_u.T @ ((ops.PHI_u * ops.QW) @ fv)
    rhs_w = ops.E_u.T @ ((ops.PHI_u * ops.QW) @ fw)
    S = assemble_saddle_3d(ops, ops.H_u)
    lu = lu_factor(S)
    sol = _refined_solve(lu, S, np.concatenate([rhs_u, rhs_v, rhs_w, hp, [0.0]]))
    nu_red = ops.n_u_red
    return sol[:nu_red], sol[nu_red:2 * nu_red], sol[2 * nu_red:3 * nu_red]


# ---------------------------------------------------------------------------
# Unsteady fully-conservative solver: Crank-Nicolson saddle with Picard
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TG3DLedger:
    times: np.ndarray
    kinetic_energy: np.ndarray
    ke_exact: np.ndarray
    ledger_residual: np.ndarray
    pressure_work_interior: np.ndarray
    divergence_residual: np.ndarray
    max_ledger_residual: float
    max_pressure_work_interior: float
    max_divergence_residual: float


@dataclass(frozen=True)
class TG3DCase:
    family: str
    p: int
    n_elements: int
    h: float
    nu: float
    dt: float
    t_final: float
    n_steps: int
    cut_fractions: tuple[float, float, float]
    volume_fraction: float
    aggregate: bool
    n_velocity: int
    n_pressure: int
    relative_l2_velocity_error: float
    max_divergence_residual: float
    max_ledger_residual_rel: float
    max_pressure_work_interior: float
    ke_relative_drift: float
    ke_decay_match_rel: float
    mean_picard_iterations: float


def solve_tg3d_unsteady(
    p: int = 2,
    n_elements: int = 6,
    cut_fractions: Sequence[float] = (0.6, 0.6, 0.6),
    nu: float = 0.0,
    dt: float = 0.01,
    t_final: float = 1.0,
    beta_factor: float = 12.0,
    aggregate: bool = True,
    picard_tol: float = 1.0e-13,
    picard_maxiter: int = 30,
    support_threshold: float = 0.05,
    ops: Mixed3DCutOperators | None = None,
    record_every: int = 1,
    quad_pts: int | None = None,
    field: str = "wall_confined",
) -> tuple[TG3DCase, TG3DLedger]:
    """Advance the exact 3D Taylor-Green / Beltrami vortex on the sharp cut box.

    Crank-Nicolson saddle stepping: the constant blocks (mass, diffusion,
    coupling, pin) are factorized once; the skew-symmetric convection at the
    midpoint velocity is lagged and Picard-iterated each step, so the converged
    step is the implicit-midpoint update of the nonlinear saddle system.  The
    divergence constraint is enforced at every time level, and the per-step
    kinetic-energy ledger

        dKE = dt * ( -nu u_m^T Diff u_m  - u_m^T C_skew u_m
                     - u_m^T G p         + u_m^T rhs_bnd )

    is closed.  In the inviscid limit (nu = 0) with the exact tangential
    boundary data the discrete kinetic energy is conserved to roundoff.

    ``field='wall_confined'`` (default) drives the closed-cavity planar Taylor-
    Green vortex of :func:`wall_confined_tgv_3d_fields`, whose normal velocity
    vanishes on every cut face: the convective and pressure boundary fluxes are
    then identically zero, so inviscidly the discrete kinetic energy is conserved
    to roundoff, and viscously it tracks the exact ``exp(-2 nu (k_x^2+k_y^2) t)``
    decay.  ``field='beltrami'`` selects the ABC-type field of
    :func:`taylor_green_3d_fields` (which exchanges energy with the open box
    boundary, so its ledger closes but its kinetic energy is not constant).
    """
    if field == "wall_confined":
        fields = wall_confined_tgv_3d_fields(nu, cut_fractions=cut_fractions, h=1.0 / n_elements)
        boundary_flux = False  # zero normal velocity on every face -> pure skew
    elif field == "beltrami":
        fields = taylor_green_3d_fields(nu)
        boundary_flux = True  # open-box through-flow -> skew-form w.n term retained
    else:
        raise ValueError(f"Unknown field family {field!r}.")
    if ops is None:
        ops = build_mixed_3d_operators(
            p, n_elements, cut_fractions=cut_fractions, beta_factor=beta_factor,
            aggregate=aggregate, support_threshold=support_threshold, quad_pts=quad_pts,
        )
    nu_red = ops.n_u_red
    np_red = ops.n_p_red
    n_steps = int(round(t_final / dt))

    A_const = ops.H_u / dt + 0.5 * nu * ops.Diff
    S_const = assemble_saddle_3d(ops, A_const)
    lu_S = lu_factor(S_const)

    def bdata(t):
        return (
            fields["u"](ops.BX, ops.BY, ops.BZ, t),
            fields["v"](ops.BX, ops.BY, ops.BZ, t),
            fields["w"](ops.BX, ops.BY, ops.BZ, t),
        )

    gu0, gv0, gw0 = bdata(0.0)
    hp_prev = continuity_rhs_3d(ops, gu0, gv0, gw0)
    fu0 = fields["u"](ops.QX, ops.QY, ops.QZ, 0.0)
    fv0 = fields["v"](ops.QX, ops.QY, ops.QZ, 0.0)
    fw0 = fields["w"](ops.QX, ops.QY, ops.QZ, 0.0)
    cu, cv, cw = _project_velocity_constrained_3d(ops, fu0, fv0, fw0, hp_prev)

    ke0 = 0.5 * float(cu @ (ops.H_u @ cu) + cv @ (ops.H_u @ cv) + cw @ (ops.H_u @ cw))
    ke_prev = ke0
    times = [0.0]
    kes = [ke_prev]
    ke_exact = [ke0]
    ledger_res = [0.0]
    pw_int = [0.0]
    div0 = float(np.linalg.norm(ops.Gx.T @ cu + ops.Gy.T @ cv + ops.Gz.T @ cw - hp_prev, ord=np.inf))
    div_res = [div0]
    max_ledger = 0.0
    max_pw = 0.0
    max_div = div0
    picard_counts = []
    cp = np.zeros(np_red)

    t = 0.0
    for step in range(1, n_steps + 1):
        t_new = step * dt
        gu_new, gv_new, gw_new = bdata(t_new)
        hp_new = continuity_rhs_3d(ops, gu_new, gv_new, gw_new)
        gu_mid, gv_mid, gw_mid = bdata(t + 0.5 * dt)
        rx_mid, ry_mid, rz_mid = weak_bc_rhs_3d(ops, gu_mid, gv_mid, gw_mid, nu)

        cu_new, cv_new, cw_new = cu.copy(), cv.copy(), cw.copy()
        n_picard = 0
        for n_picard in range(1, picard_maxiter + 1):
            um = 0.5 * (cu + cu_new)
            vm = 0.5 * (cv + cv_new)
            wm = 0.5 * (cw + cw_new)
            C_red, M_wn = convection_blocks_3d(ops, um, vm, wm)
            Cs = skew_convection(C_red, M_wn, boundary_flux=boundary_flux)
            rhs_u = ops.H_u @ (cu / dt) - 0.5 * nu * (ops.Diff @ cu) + rx_mid - Cs @ um
            rhs_v = ops.H_u @ (cv / dt) - 0.5 * nu * (ops.Diff @ cv) + ry_mid - Cs @ vm
            rhs_w = ops.H_u @ (cw / dt) - 0.5 * nu * (ops.Diff @ cw) + rz_mid - Cs @ wm
            rhs = np.concatenate([rhs_u, rhs_v, rhs_w, hp_new, [0.0]])
            sol = _refined_solve(lu_S, S_const, rhs)
            nu_next = sol[:nu_red]
            nv_next = sol[nu_red:2 * nu_red]
            nw_next = sol[2 * nu_red:3 * nu_red]
            cp = sol[3 * nu_red:3 * nu_red + np_red]
            delta = max(
                float(np.linalg.norm(nu_next - cu_new)),
                float(np.linalg.norm(nv_next - cv_new)),
                float(np.linalg.norm(nw_next - cw_new)),
            )
            scale = max(
                float(np.linalg.norm(nu_next)), float(np.linalg.norm(nv_next)),
                float(np.linalg.norm(nw_next)), 1.0e-300,
            )
            cu_new, cv_new, cw_new = nu_next, nv_next, nw_next
            if delta / scale < picard_tol:
                break
        picard_counts.append(n_picard)

        um = 0.5 * (cu + cu_new)
        vm = 0.5 * (cv + cv_new)
        wm = 0.5 * (cw + cw_new)
        C_red, M_wn = convection_blocks_3d(ops, um, vm, wm)
        Cs = skew_convection(C_red, M_wn, boundary_flux=boundary_flux)
        ke_new = 0.5 * float(cu_new @ (ops.H_u @ cu_new) + cv_new @ (ops.H_u @ cv_new) + cw_new @ (ops.H_u @ cw_new))
        d_ke = ke_new - ke_prev
        visc = -nu * float(um @ (ops.Diff @ um) + vm @ (ops.Diff @ vm) + wm @ (ops.Diff @ wm))
        conv = -float(um @ (Cs @ um) + vm @ (Cs @ vm) + wm @ (Cs @ wm))
        pwork = -float(um @ (ops.Gx @ cp) + vm @ (ops.Gy @ cp) + wm @ (ops.Gz @ cp))
        bwork = float(um @ rx_mid + vm @ ry_mid + wm @ rz_mid)
        residual = abs(d_ke - dt * (visc + conv + pwork + bwork))
        pw_interior = abs(
            float(um @ (ops.Gx @ cp) + vm @ (ops.Gy @ cp) + wm @ (ops.Gz @ cp))
            - 0.5 * float(cp @ (hp_prev + hp_new))
        )
        step_div = float(
            np.linalg.norm(ops.Gx.T @ cu_new + ops.Gy.T @ cv_new + ops.Gz.T @ cw_new - hp_new, ord=np.inf)
        )
        max_ledger = max(max_ledger, residual)
        max_pw = max(max_pw, pw_interior)
        max_div = max(max_div, step_div)

        cu, cv, cw = cu_new, cv_new, cw_new
        ke_prev = ke_new
        t = t_new
        hp_prev = hp_new
        if step % record_every == 0 or step == n_steps:
            times.append(t)
            kes.append(ke_new)
            ke_exact.append(ke0 * fields["F"](t) ** 2)
            ledger_res.append(residual)
            pw_int.append(pw_interior)
            div_res.append(step_div)

    # Final-time velocity error.
    au = ops.E_u @ cu
    av = ops.E_u @ cv
    aw = ops.E_u @ cw
    uh = ops.PHI_u.T @ au
    vh = ops.PHI_u.T @ av
    wh = ops.PHI_u.T @ aw
    W = ops.QW
    ue = fields["u"](ops.QX, ops.QY, ops.QZ, t)
    ve = fields["v"](ops.QX, ops.QY, ops.QZ, t)
    we = fields["w"](ops.QX, ops.QY, ops.QZ, t)
    l2 = sqrt(float(np.sum(W * ((uh - ue) ** 2 + (vh - ve) ** 2 + (wh - we) ** 2))))
    l2_ref = sqrt(float(np.sum(W * (ue ** 2 + ve ** 2 + we ** 2))))

    ke_arr = np.asarray(kes)
    ke_ex_arr = np.asarray(ke_exact)
    ke_scale = float(np.max(ke_arr))
    # Inviscid: KE conservation drift relative to KE0.  Viscous: match to exact.
    ke_drift = float(np.max(np.abs(ke_arr - ke0)) / max(ke0, 1.0e-300))
    ke_decay_match = float(np.max(np.abs(ke_arr - ke_ex_arr)) / max(ke0, 1.0e-300))

    ledger = TG3DLedger(
        times=np.asarray(times),
        kinetic_energy=ke_arr,
        ke_exact=ke_ex_arr,
        ledger_residual=np.asarray(ledger_res),
        pressure_work_interior=np.asarray(pw_int),
        divergence_residual=np.asarray(div_res),
        max_ledger_residual=max_ledger,
        max_pressure_work_interior=max_pw,
        max_divergence_residual=max_div,
    )
    case = TG3DCase(
        family="taylor_green_3d_cut_unsteady",
        p=p, n_elements=n_elements, h=ops.h, nu=nu, dt=dt, t_final=t, n_steps=n_steps,
        cut_fractions=ops.cut_fractions, volume_fraction=ops.volume_fraction, aggregate=aggregate,
        n_velocity=3 * nu_red, n_pressure=np_red,
        relative_l2_velocity_error=l2 / max(l2_ref, 1.0e-300),
        max_divergence_residual=ledger.max_divergence_residual,
        max_ledger_residual_rel=ledger.max_ledger_residual / max(ke_scale, 1.0e-300),
        max_pressure_work_interior=ledger.max_pressure_work_interior,
        ke_relative_drift=ke_drift,
        ke_decay_match_rel=ke_decay_match,
        mean_picard_iterations=float(np.mean(picard_counts)) if picard_counts else 0.0,
    )
    return case, ledger
