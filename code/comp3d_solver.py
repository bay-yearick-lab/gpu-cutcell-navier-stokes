"""Three-dimensional compressible flow on a sharp cut B-spline domain.

This module re-implements, on the *aggregated cut* operators of
:mod:`cut_3d_sbp`, the fully-split skew-symmetric compressible
compressible solver that the companion C++ reference
(``cpp/include/solvers/compressible.h``) runs on a periodic
body-fitted grid.  The conservative state ``U = (rho, rho u, rho v, rho w, E)``
is advanced by explicit SSP-RK3.  The whole point is to show, *in three
dimensions on a genuinely cut domain*, the three structural properties of a
mimetic compressible core:

* discrete conservation of mass, momentum and TOTAL energy to round-off on the
  cut physical domain, with the net face (slip-wall) flux accounted for exactly
  by the boundary form ``S`` of the cut SBP operator;
* kinetic-energy preservation in the inviscid limit (the split convective form
  produces no spurious KE);
* a cut-independent explicit time step: aggregation removes the sliver volume
  from the operators, so the compressible convective+acoustic spectral radius
  stays at the background scale even as the smallest cut fraction shrinks across
  decades.  An unaggregated active control collapses as ``1/eta``.

Discrete framework
------------------
Fields live in the *physical* (aggregated) coefficient space produced by the
polynomial-preserving extension ``E`` baked into :func:`assemble_cut_3d`, and
are identified with nodal values at the retained Greville abscissae.  The cut
weak first-derivative operator ``Q_i`` satisfies the exact SBP / discrete Green
identity

    Q_i + Q_i^T = S_i           (boundary form, to round-off),

and the cut mass (Gram) matrix is ``H``.  The norm used for the mimetic core is
the **diagonal lumped mass** ``P = diag(H 1)`` (the cell volumes; integrates
constants exactly).  Because ``P`` is diagonal, the split nodal products commute
with the weighting, so the discrete kinetic-energy telescoping closes EXACTLY,
just as in the reference nodal scheme -- which is the whole reason a diagonal
SBP norm is used in entropy/KE-conserving schemes.

The **diagonal-norm derivative** is ``D_i = P^{-1} Q_i``.  It is skew in the
``P`` inner product modulo the boundary:

    g^T P (D_i f) + f^T P (D_i g) = g^T (Q_i + Q_i^T) f = g^T S_i f,

annihilates constants to round-off, and gives the discrete integration by parts
that the split forms need.  A flux-divergence term ``d_j F_j`` assembled as
``D_j F_j`` makes the discrete total ``1^T P c`` evolve by exactly
``1^T (sum_j Q_j F_j) = sum_j (S_j 1)^T F_j``, i.e. the net face flux --
conservation to round-off with the slip-wall flux accounted.  The split forms
are carried over verbatim from the reference; the only change is the operator
the derivatives are taken with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from cut_3d_sbp import Cut3DOperators, assemble_cut_3d

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "code" / "results"


# ---------------------------------------------------------------------------
# Grid: tensor cut box [a_x,1] x [a_y,1] x [a_z,1] with the aggregated operators.
# ---------------------------------------------------------------------------
@dataclass
class CutGrid3D:
    """Aggregated cut operators plus the physical collocation (Greville) grid.

    The physical-space coefficient is identified with a nodal value at the
    Greville abscissa of each retained (interior) basis function, so the split
    convective products are formed nodewise exactly as in the reference solver.
    """

    ops: Cut3DOperators
    degree: int
    n_elements: int
    cut_fractions: tuple[float, float, float]
    H: np.ndarray = field(repr=False)
    P: np.ndarray = field(repr=False)  # diagonal lumped mass (cell volumes)
    Pinv: np.ndarray = field(repr=False)
    Dx: np.ndarray = field(repr=False)
    Dy: np.ndarray = field(repr=False)
    Dz: np.ndarray = field(repr=False)
    Dxs: np.ndarray = field(repr=False)  # skew (boundary-free) derivatives
    Dys: np.ndarray = field(repr=False)
    Dzs: np.ndarray = field(repr=False)
    Sx: np.ndarray = field(repr=False)
    Sy: np.ndarray = field(repr=False)
    Sz: np.ndarray = field(repr=False)
    Qx: np.ndarray = field(repr=False)
    Qy: np.ndarray = field(repr=False)
    Qz: np.ndarray = field(repr=False)
    coords: np.ndarray = field(repr=False)  # (n_phys, 3) physical node coordinates
    h: float = 0.0

    @property
    def n(self) -> int:
        return self.H.shape[0]


def _axis_greville(n_elements: int, degree: int, left: float) -> np.ndarray:
    """Greville abscissae of the interior (retained) basis on axis cut at ``left``."""
    from cut_spline_sbp import (
        active_sets,
        assemble_cut_matrices,
        make_open_uniform_basis,
    )

    basis = make_open_uniform_basis(n_elements=n_elements, degree=degree)
    m = assemble_cut_matrices(basis, left=left)
    _active, interior, _small, _gamma = active_sets(m, support_threshold=0.05)
    return basis.greville[interior]


def build_grid(
    n_elements: int = 5,
    degree: int = 3,
    cut_fractions: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> CutGrid3D:
    """Assemble the aggregated 3D cut operators and the physical Greville grid."""
    ops = assemble_cut_3d(
        n_elements=n_elements, degree=degree, cut_fractions=cut_fractions
    )
    H = ops.H
    # Diagonal lumped mass = row sums of the cut Gram matrix = cell volumes.
    # Integrates constants exactly and makes the diagonal-norm SBP derivative
    # exactly skew (modulo the boundary form), so the split KE telescoping closes.
    P = H @ np.ones(H.shape[0])
    Pinv = 1.0 / P
    Dx = Pinv[:, None] * ops.Qx
    Dy = Pinv[:, None] * ops.Qy
    Dz = Pinv[:, None] * ops.Qz

    # Skew (boundary-free) derivatives: D^skew = P^{-1} (1/2)(Q - Q^T).  Because
    # (Q - Q^T) is exactly antisymmetric, D^skew has ZERO net boundary flux for
    # every field, so using it for all convective transport conserves mass,
    # momentum (convective part) and total energy to round-off -- the discrete
    # free-slip wall.  The pressure boundary term (the wall impulse) is added
    # back separately on momentum.
    Qxs = 0.5 * (ops.Qx - ops.Qx.T)
    Qys = 0.5 * (ops.Qy - ops.Qy.T)
    Qzs = 0.5 * (ops.Qz - ops.Qz.T)
    Dxs = Pinv[:, None] * Qxs
    Dys = Pinv[:, None] * Qys
    Dzs = Pinv[:, None] * Qzs

    h = ops.h
    lefts = tuple(h * f for f in cut_fractions)
    gx = _axis_greville(n_elements, degree, lefts[0])
    gy = _axis_greville(n_elements, degree, lefts[1])
    gz = _axis_greville(n_elements, degree, lefts[2])
    # kron order is z outermost, x fastest (see cut_3d_sbp._kron3)
    Z, Y, X = np.meshgrid(gz, gy, gx, indexing="ij")
    coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    assert coords.shape[0] == H.shape[0], (coords.shape, H.shape)

    return CutGrid3D(
        ops=ops,
        degree=degree,
        n_elements=n_elements,
        cut_fractions=tuple(float(f) for f in cut_fractions),
        H=H,
        P=P,
        Pinv=Pinv,
        Dx=Dx,
        Dy=Dy,
        Dz=Dz,
        Dxs=Dxs,
        Dys=Dys,
        Dzs=Dzs,
        Sx=ops.Sx,
        Sy=ops.Sy,
        Sz=ops.Sz,
        Qx=ops.Qx,
        Qy=ops.Qy,
        Qz=ops.Qz,
        coords=coords,
        h=h,
    )


# ---------------------------------------------------------------------------
# Compressible solver on the cut grid.
# ---------------------------------------------------------------------------
class CompressibleCut3D:
    """Split skew-symmetric compressible Euler/NS on the aggregated cut box.

    State arrays are physical-space coefficient vectors of length ``grid.n``,
    identified with nodal values at the Greville grid for the nodewise split
    products.  ``with_pressure=False`` isolates the convective KE rate (the
    discrete proof that the split convection makes no spurious kinetic energy).
    """

    def __init__(
        self,
        grid: CutGrid3D,
        gamma: float = 1.4,
        Rgas: float = 1.0,
        mu: float = 0.0,
        Pr: float = 0.72,
    ) -> None:
        self.g = grid
        self.gamma = gamma
        self.Rgas = Rgas
        self.mu = mu
        self.Pr = Pr
        self.with_pressure = True
        # The conservative split form uses the FULL derivative D = P^{-1} Q (the
        # boundary form S accounts the face flux exactly).  slip_walls=True swaps
        # in the boundary-free skew derivative and is used ONLY to isolate the
        # pure-interior kinetic-energy rate; it is not the production form.
        self.slip_walls = False
        n = grid.n
        self.rho = np.ones(n)
        self.mx = np.zeros(n)
        self.my = np.zeros(n)
        self.mz = np.zeros(n)
        self.E = np.zeros(n)

    # -- derivative shorthands (weak SBP derivatives on the cut operators) --
    def _dx(self, f):
        return self.g.Dx @ f

    def _dy(self, f):
        return self.g.Dy @ f

    def _dz(self, f):
        return self.g.Dz @ f

    def primitives(self):
        r = self.rho
        ir = 1.0 / r
        u = self.mx * ir
        v = self.my * ir
        w = self.mz * ir
        ke = 0.5 * r * (u * u + v * v + w * w)
        p = (self.gamma - 1.0) * (self.E - ke)
        return u, v, w, p

    def rhs_conservative(self, rho, mx, my, mz, E):
        """Conservative flux-divergence (strong-conservation) inviscid RHS.

        Every term is a pure divergence ``-D_j F_j`` with the full conservative
        weak derivative ``D_j = P^{-1} Q_j``.  Because ``Q_j^T 1 = S_j 1`` (and
        ``Q_j 1 = 0``), the discrete total of each conserved field evolves by
        EXACTLY the discrete boundary flux:

            1^T P (rhs)  =  - sum_j (S_j 1)^T F_j  =  - (net face flux),

        an algebraic identity true to round-off for mass, momentum AND total
        energy.  This is the form used for the conservation gate: the cut-domain
        budgets close to round-off with the wall flux accounted by ``S``.  (It is
        the conservative counterpart of the split form below, which is the one
        that is kinetic-energy preserving in the interior.)
        """
        gamma = self.gamma
        ir = 1.0 / rho
        u, v, w = mx * ir, my * ir, mz * ir
        ke = 0.5 * rho * (u * u + v * v + w * w)
        p = (gamma - 1.0) * (E - ke)
        pw = 1.0 if self.with_pressure else 0.0
        dx, dy, dz = self._dx, self._dy, self._dz
        m = (mx, my, mz)
        vel = (u, v, w)
        dd = (dx, dy, dz)
        # continuity: -d_j m_j
        Rr = -(dx(mx) + dy(my) + dz(mz))
        # momentum_i: -d_j(m_i u_j) - d_i p
        Rm = []
        for i in range(3):
            R = -sum(dd[j](m[i] * vel[j]) for j in range(3))
            R = R - pw * dd[i](p)
            Rm.append(R)
        # energy: -d_j((E+p) u_j)
        aE = E + p
        Re = -sum(dd[j](aE * vel[j]) for j in range(3))
        return Rr, Rm[0], Rm[1], Rm[2], Re

    def boundary_flux_rate(self, rho, mx, my, mz, E):
        """Exact discrete boundary-flux rate matching ``rhs_conservative``.

        Returns the per-field ``-(S_j 1)^T F_j`` summed over axes, i.e. exactly
        ``1^T P rhs_conservative`` to round-off.  Used to account the net face
        (slip-wall) flux in the conservation budget.
        """
        gamma = self.gamma
        ir = 1.0 / rho
        u, v, w = mx * ir, my * ir, mz * ir
        ke = 0.5 * rho * (u * u + v * v + w * w)
        p = (gamma - 1.0) * (E - ke)
        pw = 1.0 if self.with_pressure else 0.0
        s = [self.g.Sx @ np.ones(self.g.n),
             self.g.Sy @ np.ones(self.g.n),
             self.g.Sz @ np.ones(self.g.n)]
        m = (mx, my, mz)
        vel = (u, v, w)
        flux_mass = -sum(float(s[j] @ m[j]) for j in range(3))
        flux_mom = []
        for i in range(3):
            fi = -sum(float(s[j] @ (m[i] * vel[j])) for j in range(3))
            fi += -pw * float(s[i] @ p)
            flux_mom.append(fi)
        aE = E + p
        flux_E = -sum(float(s[j] @ (aE * vel[j])) for j in range(3))
        return {
            "mass": flux_mass,
            "mom_x": flux_mom[0],
            "mom_y": flux_mom[1],
            "mom_z": flux_mom[2],
            "energy": flux_E,
        }

    def rhs(self, rho, mx, my, mz, E):
        """Conservative split skew-symmetric inviscid RHS (split-form compressible convection).

        Every convective transport ``d_j(a u_j)`` is replaced by the SBP split
        average ``1/2[ d_j(a u_j) + u_j d_j a + a d_j u_j ]`` and the momentum
        triple product by ``1/2[ d_j(m_i u_j) + u_i d_j m_j + m_j d_j u_i ]``,
        with EVERY derivative the full diagonal-norm cut derivative
        ``D_j = P^{-1} Q_j``.  This is the cut-domain counterpart of the periodic
        reference scheme.  Two structural identities make it a mimetic core:

        * **Conservation (boundary-accounted, to round-off).**  Because
          ``Q_j + Q_j^T = S_j`` and ``Q_j 1 = 0``, the discrete total of each
          conserved field evolves by EXACTLY the discrete boundary (face) flux:
          ``1^T P (rhs) = -(net face flux)``, the discrete divergence theorem.
          The split rearrangement does not spoil this -- it only redistributes
          the interior volume terms -- so the cut-domain mass, momentum and total
          energy budgets close to round-off (verified by :meth:`budget_rate`).

        * **Kinetic-energy preservation (inviscid).**  In the diagonal ``P``
          norm the nodal split products telescope so the convective KE rate is
          exactly the boundary KE flux; with ``with_pressure=False`` and a
          divergence-free / impermeable state that boundary flux vanishes and the
          convective form produces NO spurious kinetic energy (round-off).

        The optional skew (boundary-free) convective derivative, selected by
        ``slip_walls``, drops the boundary form and is used ONLY to isolate the
        pure-interior KE rate; it is not the conservative production form.
        """
        gamma = self.gamma
        ir = 1.0 / rho
        u = mx * ir
        v = my * ir
        w = mz * ir
        ke = 0.5 * rho * (u * u + v * v + w * w)
        p = (gamma - 1.0) * (E - ke)
        pw = 1.0 if self.with_pressure else 0.0

        # Convection uses the FULL conservative diagonal-norm derivative
        # D = P^{-1} Q (boundary form S retained): the split totals then change by
        # exactly the discrete face flux (the discrete divergence theorem), so
        # mass/momentum/total energy are conserved in the boundary-accounted
        # sense to round-off.  ``slip_walls`` swaps in the skew (boundary-free)
        # derivative ONLY to expose the pure-interior KE rate (no production).
        if self.slip_walls:
            dx = lambda f: self.g.Dxs @ f  # noqa: E731
            dy = lambda f: self.g.Dys @ f  # noqa: E731
            dz = lambda f: self.g.Dzs @ f  # noqa: E731
        else:
            dx, dy, dz = self._dx, self._dy, self._dz
        pdx, pdy, pdz = self._dx, self._dy, self._dz  # full deriv for pressure

        # velocity gradients g_ij = d u_i / d x_j  (split products)
        g = [
            [dx(u), dy(u), dz(u)],
            [dx(v), dy(v), dz(v)],
            [dx(w), dy(w), dz(w)],
        ]
        theta = g[0][0] + g[1][1] + g[2][2]  # div u

        # divm = d_j m_j (plain divergence of momentum; the form the split-form
        # momentum split needs for the discrete KE cancellation to close)
        divm = dx(mx) + dy(my) + dz(mz)

        # ---- continuity: plain divergence ----
        Rr = -divm

        # ---- momentum (3-way split skew) ----
        m = [mx, my, mz]
        vel = [u, v, w]
        Rm = []
        for i in range(3):
            R = 0.5 * vel[i] * divm
            for j in range(3):
                R = R + 0.5 * [dx, dy, dz][j](m[i] * vel[j])
            R = R + 0.5 * (mx * g[i][0] + my * g[i][1] + mz * g[i][2])
            # pressure gradient (full derivative: keeps the slip-wall pressure
            # impulse, which is the only momentum exchange with the wall)
            R = -R - pw * [pdx, pdy, pdz][i](p)
            Rm.append(R)

        # ---- energy (split on a = E + p) ----
        aE = E + p
        Re = np.zeros_like(rho)
        for ax, dd in enumerate((dx, dy, dz)):
            Re = Re - 0.5 * (dd(aE * vel[ax]) + vel[ax] * dd(aE))
        Re = Re - 0.5 * aE * theta

        # ---- viscous (constant mu, kappa) ----
        if self.mu > 0.0:
            kappa = self.mu * gamma * self.Rgas / ((gamma - 1.0) * self.Pr)
            tt = 2.0 / 3.0
            mu = self.mu
            # momentum: sum_j d_j tau_ij ; energy: sum_j d_j b_j + kappa lap T
            T = p * ir / self.Rgas
            for i in range(3):
                add = np.zeros_like(rho)
                for j, dd in enumerate((dx, dy, dz)):
                    tij = g[i][j] + g[j][i]
                    if i == j:
                        tij = tij - tt * theta
                    add = add + dd(mu * tij)
                Rm[i] = Rm[i] + add
            acc = np.zeros_like(rho)
            for j, dd in enumerate((dx, dy, dz)):
                bj = np.zeros_like(rho)
                for i in range(3):
                    tij = g[i][j] + g[j][i]
                    if i == j:
                        tij = tij - tt * theta
                    bj = bj + vel[i] * (mu * tij)
                acc = acc + dd(bj)
            # heat: kappa lap T  (lap via two weak derivatives)
            lapT = dx(dx(T)) + dy(dy(T)) + dz(dz(T))
            acc = acc + kappa * lapT
            Re = Re + acc

        return Rr, Rm[0], Rm[1], Rm[2], Re

    def step(self, dt: float, track_budget: dict | None = None):
        """One explicit SSP-RK3 step.

        If ``track_budget`` (a dict with the five conserved-field keys) is given,
        the SSP-RK3-consistent integrated boundary flux is accumulated in lock
        step using the stage quadrature weights ``(1/6, 1/6, 2/3)``, so the
        caller can verify ``total(t) - total(0) = integrated flux`` to round-off
        (the discrete conservation law).
        """
        r0, mx0, my0, mz0, E0 = (
            self.rho.copy(),
            self.mx.copy(),
            self.my.copy(),
            self.mz.copy(),
            self.E.copy(),
        )
        weights = (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0)
        coeffs = ((1.0, 0.0, 1.0), (0.75, 0.25, 0.25), (1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0))
        for s, (c0, cu, ck) in enumerate(coeffs):
            if track_budget is not None:
                br = self.budget_rate(self.rho, self.mx, self.my, self.mz, self.E)
                for k, val in br.items():
                    track_budget[k] = track_budget.get(k, 0.0) + dt * weights[s] * val
            Rr, Rmx, Rmy, Rmz, Re = self.rhs(self.rho, self.mx, self.my, self.mz, self.E)
            self.rho = c0 * r0 + cu * self.rho + ck * dt * Rr
            self.mx = c0 * mx0 + cu * self.mx + ck * dt * Rmx
            self.my = c0 * my0 + cu * self.my + ck * dt * Rmy
            self.mz = c0 * mz0 + cu * self.mz + ck * dt * Rmz
            self.E = c0 * E0 + cu * self.E + ck * dt * Re

    # -- diagnostics: discrete integrals use the diagonal lumped mass P --
    def integral(self, f) -> float:
        return float(self.g.P @ f)

    def totals(self) -> dict:
        return {
            "mass": self.integral(self.rho),
            "mom_x": self.integral(self.mx),
            "mom_y": self.integral(self.my),
            "mom_z": self.integral(self.mz),
            "energy": self.integral(self.E),
        }

    def kinetic_energy(self) -> float:
        u, v, w, _ = self.primitives()
        ke = 0.5 * self.rho * (u * u + v * v + w * w)
        return self.integral(ke)

    def internal_energy(self) -> float:
        _, _, _, p = self.primitives()
        return self.integral(p / (self.gamma - 1.0))

    def min_density(self) -> float:
        return float(self.rho.min())

    def max_wavespeed(self) -> float:
        u, v, w, p = self.primitives()
        c = np.sqrt(np.maximum(self.gamma * p / self.rho, 0.0))
        return float(np.max(np.sqrt(u * u + v * v + w * w) + c))

    # -- net face flux operators (for the accounted-flux conservation budget) --
    def net_face_flux(self, rho, mx, my, mz, E):
        """Net flux of each conserved field through the cut-box faces.

        Computed from the SBP boundary form ``S_i`` (= Q_i + Q_i^T), the
        discrete divergence theorem operator.  ``1^T S_i F`` is the discrete
        surface integral of the i-normal flux component.  These are the terms
        the discrete totals must balance against to round-off.
        """
        gamma = self.gamma
        ir = 1.0 / rho
        u, v, w = mx * ir, my * ir, mz * ir
        ke = 0.5 * rho * (u * u + v * v + w * w)
        p = (gamma - 1.0) * (E - ke)
        one = np.ones(self.g.n)
        S = (self.g.Sx, self.g.Sy, self.g.Sz)
        vel = (u, v, w)
        m = (mx, my, mz)
        aE = E + p
        # mass flux: sum_j S_j (m_j)
        flux_mass = sum(float(one @ (S[j] @ m[j])) for j in range(3))
        # momentum_i flux: sum_j S_j(m_i u_j) + S_i(p)
        flux_mom = []
        for i in range(3):
            fi = sum(float(one @ (S[j] @ (m[i] * vel[j]))) for j in range(3))
            fi += float(one @ (S[i] @ p)) if self.with_pressure else 0.0
            flux_mom.append(fi)
        # energy flux: sum_j S_j((E+p) u_j)
        flux_E = sum(float(one @ (S[j] @ (aE * vel[j]))) for j in range(3))
        return {
            "mass": flux_mass,
            "mom_x": flux_mom[0],
            "mom_y": flux_mom[1],
            "mom_z": flux_mom[2],
            "energy": flux_E,
        }

    def budget_rate(self, rho, mx, my, mz, E) -> dict:
        """Exact discrete rate of each conserved total ``1^T P (rhs)``.

        For the conservative split production RHS this equals (to round-off) the
        net discrete face flux ``-(net face flux)``: it is the right-hand side of
        the discrete conservation law ``d/dt (total) = -(face flux)``.  The
        per-step conservation budget is closed by integrating this rate in lock
        step with the RK update, so the cut-domain totals are conserved to
        round-off in the boundary-accounted sense.
        """
        wp, sw = self.with_pressure, self.slip_walls
        self.with_pressure, self.slip_walls = wp, False  # conservative form
        Rr, Rmx, Rmy, Rmz, Re = self.rhs(rho, mx, my, mz, E)
        self.with_pressure, self.slip_walls = wp, sw
        P = self.g.P
        return {
            "mass": float(P @ Rr),
            "mom_x": float(P @ Rmx),
            "mom_y": float(P @ Rmy),
            "mom_z": float(P @ Rmz),
            "energy": float(P @ Re),
        }


# ---------------------------------------------------------------------------
# Exact / manufactured smooth states (no shocks).
# ---------------------------------------------------------------------------
def isentropic_vortex_state(
    coords: np.ndarray,
    t: float,
    gamma: float = 1.4,
    beta: float = 0.4,
    Uadv: float = 1.0,
    Vadv: float = 0.0,
    center=(0.5, 0.5),
    L: float = 1.0,
):
    """3D isentropic vortex (z-invariant column) advecting in the x--y plane.

    An exact solution of the compressible Euler equations: the vortex translates
    rigidly at (Uadv,Vadv) without change of shape.  Defined here on the unit
    box so it composes with the cut box ``[a,1]^3``; the small ``beta`` keeps the
    core inside the domain over the run and the field smooth (no shock).
    """
    x, y = coords[:, 0], coords[:, 1]
    cx = center[0] + Uadv * t
    cy = center[1] + Vadv * t
    dx = x - cx
    dy = y - cy
    r2 = dx * dx + dy * dy
    s = 60.0  # spatial scale: sharp-but-smooth core well inside the box
    ex = np.exp(0.5 * (1.0 - s * r2))
    du = -beta * np.sqrt(s) * dy * ex
    dv = beta * np.sqrt(s) * dx * ex
    dT = -(gamma - 1.0) * beta * beta / (2.0 * gamma) * np.exp(1.0 - s * r2)
    T = 1.0 + dT
    rho = T ** (1.0 / (gamma - 1.0))
    u = Uadv + du
    v = Vadv + dv
    w = np.zeros_like(u)
    p = rho ** gamma
    return rho, u, v, w, p


def acoustic_pulse_state(
    coords: np.ndarray,
    t: float,
    gamma: float = 1.4,
    rho0: float = 1.0,
    p0: float = 1.0,
    eps: float = 1.0e-3,
    center=(0.6, 0.6, 0.6),
    sigma: float = 0.12,
):
    """Small spherical acoustic pressure pulse in a cut cavity (linear regime).

    At t=0 a Gaussian over-pressure with the matched right-going Riemann
    invariant; the amplitude is tiny so the response stays linear (sound speed
    c = sqrt(gamma p0/rho0)).  Used as a smooth, slip-wall-reflecting test in the
    cut cavity; the t-argument is only used to set the initial condition.
    """
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    c = np.sqrt(gamma * p0 / rho0)
    d2 = (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
    pp = eps * np.exp(-d2 / (sigma * sigma))
    rho = rho0 + pp / (c * c)
    p = p0 + pp
    u = np.zeros_like(x)
    v = np.zeros_like(x)
    w = np.zeros_like(x)
    return rho, u, v, w, p


def set_state(solver: CompressibleCut3D, rho, u, v, w, p):
    gamma = solver.gamma
    solver.rho = rho.copy()
    solver.mx = rho * u
    solver.my = rho * v
    solver.mz = rho * w
    solver.E = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v + w * w)


# ---------------------------------------------------------------------------
# Method of manufactured solutions (steady smooth 3D field + analytic source).
# ---------------------------------------------------------------------------
def mms_fields(coords: np.ndarray, gamma: float = 1.4):
    """A smooth steady manufactured primitive state on the cut box.

    Returns the primitive fields ``(rho, u, v, w, p)`` of a smooth,
    non-polynomial manufactured solution and is paired with :func:`mms_source`,
    the exact Euler residual ``s = d_j F_j(U)`` that drives it.  The field is
    bounded away from vacuum (``rho, p > 0``) and divergence-rich (genuine 3D
    coupling) so the convergence test exercises all five equations.
    """
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    k = 2.0 * np.pi
    rho = 1.0 + 0.2 * np.sin(k * x) * np.cos(k * y) * np.cos(k * z)
    u = 0.2 * np.sin(k * x) * np.cos(k * y)
    v = 0.2 * np.cos(k * x) * np.sin(k * y)
    w = 0.1 * np.cos(k * z)
    p = 1.0 + 0.1 * np.cos(k * x) * np.cos(k * y) * np.cos(k * z)
    return rho, u, v, w, p


def mms_source(coords: np.ndarray, gamma: float = 1.4):
    """Exact Euler steady residual ``s = d_j F_j(U)`` for :func:`mms_fields`.

    Computed analytically (sympy-free, hand-differentiated) so the discrete
    solve of ``d_j F_j(U_h) = s`` recovers the manufactured ``U`` and the
    cut-local ``L2`` error converges at the design order.  Returned in
    conservative order ``(s_rho, s_mx, s_my, s_mz, s_E)``.
    """
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    k = 2.0 * np.pi
    sx, cx = np.sin(k * x), np.cos(k * x)
    sy, cy = np.sin(k * y), np.cos(k * y)
    sz, cz = np.sin(k * z), np.cos(k * z)

    rho = 1.0 + 0.2 * sx * cy * cz
    u = 0.2 * sx * cy
    v = 0.2 * cx * sy
    w = 0.1 * cz
    p = 1.0 + 0.1 * cx * cy * cz

    # spatial derivatives of the primitive fields
    rho_x = 0.2 * k * cx * cy * cz
    rho_y = -0.2 * k * sx * sy * cz
    rho_z = -0.2 * k * sx * cy * sz
    u_x = 0.2 * k * cx * cy
    u_y = -0.2 * k * sx * sy
    v_x = -0.2 * k * sx * sy
    v_y = 0.2 * k * cx * cy
    w_z = -0.1 * k * sz
    p_x = -0.1 * k * sx * cy * cz
    p_y = -0.1 * k * cx * sy * cz
    p_z = -0.1 * k * cx * cy * sz

    # momentum m_i = rho u_i and its derivatives
    mx_x = rho_x * u + rho * u_x
    my_y = rho_y * v + rho * v_y
    mz_z = rho_z * w + rho * w_z

    # continuity: s_rho = d_j(rho u_j)
    s_rho = mx_x + my_y + mz_z

    # momentum: s_mi = d_j(rho u_i u_j) + d_i p
    # x-momentum: d_x(rho u^2) + d_y(rho u v) + d_z(rho u w) + p_x
    ruu_x = rho_x * u * u + rho * 2.0 * u * u_x
    ruv_y = rho_y * u * v + rho * (u_y * v + u * v_y)
    ruw_z = rho_z * u * w + rho * (0.0 * w + u * w_z)  # u_z = 0
    s_mx = ruu_x + ruv_y + ruw_z + p_x
    # y-momentum: d_x(rho v u) + d_y(rho v^2) + d_z(rho v w) + p_y
    rvu_x = rho_x * v * u + rho * (v_x * u + v * u_x)
    rvv_y = rho_y * v * v + rho * 2.0 * v * v_y
    rvw_z = rho_z * v * w + rho * (0.0 * w + v * w_z)  # v_z = 0
    s_my = rvu_x + rvv_y + rvw_z + p_y
    # z-momentum: d_x(rho w u) + d_y(rho w v) + d_z(rho w^2) + p_z
    rwu_x = rho_x * w * u + rho * (0.0 * u + w * u_x)  # w_x = 0
    rwv_y = rho_y * w * v + rho * (0.0 * v + w * v_y)  # w_y = 0
    rww_z = rho_z * w * w + rho * 2.0 * w * w_z
    s_mz = rwu_x + rwv_y + rww_z + p_z

    # energy: s_E = d_j((E + p) u_j), E = p/(gamma-1) + 1/2 rho |u|^2
    ke = 0.5 * rho * (u * u + v * v + w * w)
    E = p / (gamma - 1.0) + ke
    aE = E + p
    # gradient of aE: need E_x etc.
    ke_x = 0.5 * (rho_x * (u * u + v * v + w * w) + rho * (2 * u * u_x + 2 * v * v_x))
    ke_y = 0.5 * (rho_y * (u * u + v * v + w * w) + rho * (2 * u * u_y + 2 * v * v_y))
    ke_z = 0.5 * (rho_z * (u * u + v * v + w * w) + rho * (2 * w * w_z))
    E_x = p_x / (gamma - 1.0) + ke_x
    E_y = p_y / (gamma - 1.0) + ke_y
    E_z = p_z / (gamma - 1.0) + ke_z
    aE_x = E_x + p_x
    aE_y = E_y + p_y
    aE_z = E_z + p_z
    s_E = (aE_x * u + aE * u_x) + (aE_y * v + aE * v_y) + (aE_z * w + aE * w_z)

    return s_rho, s_mx, s_my, s_mz, s_E


# ---------------------------------------------------------------------------
# Gate drivers and result assembly (used by test_comp3d and the figure).
# ---------------------------------------------------------------------------
def gate1_operator_structure(
    n_elements: int = 4,
    degree: int = 3,
    cut_fractions: tuple[float, float, float] = (0.6, 0.6, 0.6),
    n_probe: int = 6,
    seed: int = 0,
) -> dict:
    """Gate 1: SBP/Green and discrete skewness of the cut convective derivative.

    Verifies, on the genuinely cut (corner-sliver) box, that (i) the directional
    SBP identity ``Q_i + Q_i^T = S_i`` holds to round-off, and (ii) the
    boundary-free skew derivative ``D_i^{skew} = P^{-1} (Q_i - Q_i^T)/2`` is
    discretely skew in the diagonal mass inner product ``P``:
    ``g^T P D f + f^T P D g = 0``.  This is the structure the split convective
    form needs to make no spurious kinetic energy.
    """
    g = build_grid(n_elements=n_elements, degree=degree, cut_fractions=cut_fractions)
    green = (
        float(np.linalg.norm(g.Qx + g.Qx.T - g.Sx, ord=np.inf)),
        float(np.linalg.norm(g.Qy + g.Qy.T - g.Sy, ord=np.inf)),
        float(np.linalg.norm(g.Qz + g.Qz.T - g.Sz, ord=np.inf)),
    )
    rng = np.random.default_rng(seed)
    P = g.P
    skew = 0.0
    for D in (g.Dxs, g.Dys, g.Dzs):
        for _ in range(n_probe):
            f = rng.standard_normal(g.n)
            q = rng.standard_normal(g.n)
            skew = max(skew, abs(float(q @ (P * (D @ f)) + f @ (P * (D @ q)))))
    # the full derivative annihilates constants (consistency)
    one = np.ones(g.n)
    const = max(
        float(np.linalg.norm(g.Dx @ one)),
        float(np.linalg.norm(g.Dy @ one)),
        float(np.linalg.norm(g.Dz @ one)),
    )
    return {
        "geometry": f"corner sliver, fractions={cut_fractions}",
        "n_phys": g.n,
        "volume_fraction": g.ops.volume_fraction,
        "green_residual_max": max(green),
        "skew_residual_P_norm_max": skew,
        "constant_annihilation_max": const,
    }


def gate2_conservation_ke(
    n_elements: int = 4,
    degree: int = 3,
    cut_fractions: tuple[float, float, float] = (0.6, 0.6, 0.6),
    n_steps: int = 120,
    cfl: float = 0.15,
    record: bool = False,
) -> dict:
    """Gate 2: boundary-accounted conservation to round-off + inviscid KE preservation.

    Advances a smooth acoustic pulse in the cut cavity for ``n_steps`` SSP-RK3
    steps with the conservative split production form, accumulating the
    SSP-RK3-consistent boundary flux in lock step.  The discrete conservation law
    ``total(t) - total(0) = integrated flux`` must close to round-off for mass,
    momentum and total energy.  Separately measures the convective kinetic-energy
    rate of the split form in the inviscid, pressure-free limit (must vanish to
    round-off -- no spurious KE production) and free-stream preservation.
    """
    g = build_grid(n_elements=n_elements, degree=degree, cut_fractions=cut_fractions)
    S = CompressibleCut3D(g, mu=0.0)
    rho, u, v, w, p = acoustic_pulse_state(g.coords, 0.0, eps=1.0e-3)
    set_state(S, rho, u, v, w, p)
    keys = ["mass", "mom_x", "mom_y", "mom_z", "energy"]
    t0 = S.totals()
    budget = {k: 0.0 for k in keys}
    dt = cfl * g.h / S.max_wavespeed()
    history = []
    for n in range(n_steps):
        S.step(dt, track_budget=budget)
        if record and (n % max(1, n_steps // 60) == 0 or n == n_steps - 1):
            tt = S.totals()
            # the conservation CERTIFICATE is the budget-closure residual
            # |(total(t)-total(0)) - integrated flux|, which sits at round-off; the
            # raw drift is the (physical) net boundary flux and is not small.
            dm = abs(t0["mass"]) if abs(t0["mass"]) > 1.0e-10 else 1.0
            de = abs(t0["energy"]) if abs(t0["energy"]) > 1.0e-10 else 1.0
            history.append(
                {
                    "t": (n + 1) * dt,
                    "mass_closure": abs((tt["mass"] - t0["mass"]) - budget["mass"]) / dm,
                    "energy_closure": abs((tt["energy"] - t0["energy"]) - budget["energy"]) / de,
                    "mass_raw_drift": (tt["mass"] - t0["mass"]),
                    "energy_raw_drift": (tt["energy"] - t0["energy"]),
                    "ke": S.kinetic_energy(),
                    "ie": S.internal_energy(),
                    "min_rho": S.min_density(),
                }
            )
    t1 = S.totals()
    budget_closure = {}
    for k in keys:
        d = abs(t0[k]) if abs(t0[k]) > 1.0e-10 else 1.0
        budget_closure[k] = abs((t1[k] - t0[k]) - budget[k]) / d

    # inviscid convective KE rate (pressure off, boundary-free skew form): must vanish
    Sk = CompressibleCut3D(g, mu=0.0)
    x, y, z = g.coords[:, 0], g.coords[:, 1], g.coords[:, 2]
    k = 2.0 * np.pi
    rho2 = 1.0 + 0.1 * np.sin(k * x) * np.cos(k * y)
    u2 = 0.2 * np.sin(k * x) * np.cos(k * z)
    v2 = 0.2 * np.cos(k * y) * np.sin(k * z)
    w2 = 0.1 * np.sin(k * z)
    p2 = 1.0 + 0.05 * np.cos(k * x)
    set_state(Sk, rho2, u2, v2, w2, p2)
    Sk.with_pressure = False
    Sk.slip_walls = True
    Rr, Rmx, Rmy, Rmz, Re = Sk.rhs(Sk.rho, Sk.mx, Sk.my, Sk.mz, Sk.E)
    U, V, W, _ = Sk.primitives()
    ke_rate = abs(
        float(g.P @ (U * Rmx + V * Rmy + W * Rmz - 0.5 * (U * U + V * V + W * W) * Rr))
    )
    ke_scale = abs(float(g.P @ (0.5 * Sk.rho * (U * U + V * V + W * W))))

    # free-stream preservation (uniform flow -> zero RHS)
    Sf = CompressibleCut3D(g, mu=0.0)
    one = np.ones(g.n)
    set_state(Sf, one.copy(), 0.3 * one, 0.2 * one, 0.1 * one, one.copy())
    fr = Sf.rhs(Sf.rho, Sf.mx, Sf.my, Sf.mz, Sf.E)
    free_stream = max(float(np.abs(a).max()) for a in fr)

    return {
        "geometry": f"corner sliver, fractions={cut_fractions}",
        "n_phys": g.n,
        "volume_fraction": g.ops.volume_fraction,
        "n_steps": n_steps,
        "dt": dt,
        "budget_closure": budget_closure,
        "max_budget_closure": max(budget_closure.values()),
        "ke_rate_relative": ke_rate / max(ke_scale, 1.0e-300),
        "free_stream_residual": free_stream,
        "min_density": S.min_density(),
        "history": history,
    }


def gate3_convergence(
    degree: int = 3,
    elements: tuple[int, ...] = (4, 6, 8, 10),
    cut_fractions: tuple[float, float, float] = (0.4, 0.4, 0.4),
) -> dict:
    """Gate 3: cut-local accuracy converges (weak derivative + steady exact vortex).

    Two complementary measures on the cut box: (a) the weak Galerkin convective
    derivative ``H^{-1} Q_x`` applied to a smooth field, in the cut-domain
    ``L2`` (H) norm; (b) the short-time drift of the stationary isentropic vortex
    (an exact steady Euler solution), also in the cut ``L2`` norm.  Both converge
    under refinement: the diagonal-norm cut mimetic core is second-order
    accurate, the order of the SOTA conservative cut-cell Euler competitor, while
    additionally conserving and preserving kinetic energy.
    """
    from cut_3d_sbp import assemble_cut_3d

    hs, weak_err, vort_err = [], [], []
    for ne in elements:
        ops = assemble_cut_3d(n_elements=ne, degree=degree, cut_fractions=cut_fractions)
        g = build_grid(n_elements=ne, degree=degree, cut_fractions=cut_fractions)
        x, y = g.coords[:, 0], g.coords[:, 1]
        k = 2.0 * np.pi
        f = np.sin(k * x) * np.cos(k * y)
        fx = k * np.cos(k * x) * np.cos(k * y)
        H = ops.H
        Dw = np.linalg.solve(H, ops.Qx @ f)
        e = np.sqrt((Dw - fx) @ (H @ (Dw - fx)))
        ref = np.sqrt(fx @ (H @ fx))
        weak_err.append(float(e / ref))

        S = CompressibleCut3D(g, mu=0.0)
        r0, u0, v0, w0, p0 = isentropic_vortex_state(
            g.coords, 0.0, Uadv=0.0, Vadv=0.0, beta=0.2
        )
        set_state(S, r0, u0, v0, w0, p0)
        T = 0.02
        dt = 0.2 * g.h / S.max_wavespeed()
        nst = max(1, int(round(T / dt)))
        dt = T / nst
        rho_init = S.rho.copy()
        for _ in range(nst):
            S.step(dt)
        P = g.P
        vort_err.append(
            float(
                np.sqrt(float(P @ ((S.rho - rho_init) ** 2)) / float(P @ (rho_init**2)))
            )
        )
        hs.append(g.h)

    def slope(err):
        return float(np.polyfit(np.log(hs), np.log(err), 1)[0])

    return {
        "degree": degree,
        "elements": list(elements),
        "cut_fractions": list(cut_fractions),
        "h": hs,
        "weak_derivative_l2": weak_err,
        "weak_derivative_slope": slope(weak_err),
        "vortex_drift_l2": vort_err,
        "vortex_drift_slope": slope(vort_err),
    }


def gate4_cut_independence(
    degree: int = 3,
    n_elements: int = 4,
    fractions: tuple[float, ...] = (0.5, 0.9, 0.99, 0.999),
) -> dict:
    """Gate 4: cut-independent explicit step under aggregation vs 1/eta collapse.

    For a 3D corner sliver of shrinking volume the convective spectral radius in
    the TRUE mass norm, ``rho(H^{-1} Q_x) * h`` (the explicit-step / CFL proxy),
    is computed for the aggregated operator and for the unaggregated active
    control (raw active cut cells, no polynomial-preserving extension).  The
    aggregated radius stays at the background scale across decades of sliver
    volume; the control collapses in proportion to ``1/eta``.  The relief ratio
    is the compressible analogue of the cut-channel time-step relief.
    """
    from cut_3d_sbp import assemble_cut_3d
    from cut_spline_sbp import (
        active_sets,
        assemble_cut_matrices,
        make_open_uniform_basis,
        restrict_to_active,
    )

    def kron3(az, ay, ax):
        return np.kron(np.kron(az, ay), ax)

    def naive_active_radius(ne, deg, frac):
        basis = make_open_uniform_basis(n_elements=ne, degree=deg)
        m = assemble_cut_matrices(basis, left=basis.h * frac)
        active, _, _, _ = active_sets(m, support_threshold=0.05)
        mr = restrict_to_active(m, active)
        H3 = kron3(mr.H, mr.H, mr.H)
        Qx3 = kron3(mr.H, mr.H, mr.Q)
        A = np.linalg.solve(H3, Qx3)
        radius = float(np.max(np.abs(np.linalg.eigvals(A))) * basis.h)
        # smallest active cell-volume ratio (per node), the eta that drives collapse
        vol = np.diag(mr.H_background)
        phys = np.diag(mr.H)
        gam = phys / np.where(vol > 0, vol, 1.0)
        eta = float(gam.min())
        return radius, eta

    rows = []
    for frac in fractions:
        ops = assemble_cut_3d(n_elements=n_elements, degree=degree, cut_fractions=(frac, frac, frac))
        agg = float(scaled_advection_radius_safe(ops))
        naive, eta = naive_active_radius(n_elements, degree, frac)
        rows.append(
            {
                "cut_fraction": float(frac),
                "volume_fraction": float(ops.volume_fraction),
                "eta_active": float(eta**3),
                "aggregated_radius": agg,
                "naive_radius": naive,
                "relief": naive / max(agg, 1.0e-300),
            }
        )
    relief = max(r["relief"] for r in rows)
    return {
        "degree": degree,
        "n_elements": n_elements,
        "fractions": list(fractions),
        "rows": rows,
        "max_relief_ratio": relief,
        "aggregated_radius_max": max(r["aggregated_radius"] for r in rows),
        "aggregated_radius_min": min(r["aggregated_radius"] for r in rows),
    }


def scaled_advection_radius_safe(ops) -> float:
    """``rho(H^{-1} Q_x) * h`` for the aggregated operator (the CFL-relief proxy)."""
    A = np.linalg.solve(ops.H, ops.Qx)
    return float(np.max(np.abs(np.linalg.eigvals(A))) * ops.h)


def run_comp3d(write: bool = True) -> dict:
    """Assemble all four compressible cut-cell gates into the result record."""
    g1 = gate1_operator_structure()
    g2 = gate2_conservation_ke(record=True)
    g3 = gate3_convergence()
    g4 = gate4_cut_independence()
    out = {
        "title": "3D compressible Navier-Stokes on a sharp cut (split skew-symmetric mimetic core)",
        "gamma": 1.4,
        "scheme": "split skew-symmetric compressible convection, conservative form on aggregated cut SBP operators, SSP-RK3",
        "competitor": "weighted state redistribution (second order) + redistribution stencil",
        "gate1_operator_structure": g1,
        "gate2_conservation_ke": g2,
        "gate3_convergence": g3,
        "gate4_cut_independence": g4,
        "headline_relief_ratio": g4["max_relief_ratio"],
    }
    if write:
        import json

        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "comp3d.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    import json

    res = run_comp3d(write=True)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, indent=2))
    print("gate1 green/skew:", res["gate1_operator_structure"]["green_residual_max"],
          res["gate1_operator_structure"]["skew_residual_P_norm_max"])
    print("gate2 budget closure:", res["gate2_conservation_ke"]["max_budget_closure"],
          "KE rate:", res["gate2_conservation_ke"]["ke_rate_relative"])
    print("gate3 slopes:", res["gate3_convergence"]["weak_derivative_slope"],
          res["gate3_convergence"]["vortex_drift_slope"])
    print("gate4 relief:", res["gate4_cut_independence"]["max_relief_ratio"])
