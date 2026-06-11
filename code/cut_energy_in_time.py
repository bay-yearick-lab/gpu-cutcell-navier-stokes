"""Energy-in-time demonstrations on a sharp oblique cut.

The static mass-spectrum sweep (``s6_mass_spectrum``) shows that the active
assembly mass matrix loses positivity as the cut sliver shrinks
(``lambda_min(H_active) -> 0``) while the aggregated mass matrix keeps an
order-one floor, and symmetrically that the active derivative spectral radius
grows like the full/small cell ratio while the aggregated radius stays bounded.
Those are *static* spectral facts.  This module turns them into *dynamical*
facts, which is the form practitioners cite.

Two witnesses are provided.

Cut-position-independent time step (``wave_cfl_contrast``).
    The scalar wave system ``M p'' = -K p`` is built on the same oblique cut for
    the active space and the aggregated space, and both are advanced with the
    explicit Stormer-Verlet (leapfrog) step from the same initial mode at a
    *single, fixed* time step chosen from the aggregated Courant limit.  The
    aggregated energy stays bounded; the active energy blows up at the same step
    because its stability limit collapses like ``sqrt(epsilon)`` at the small
    cell.  An implicit integrator would hide the effect by being unconditionally
    stable, so the contrast is deliberately explicit: it is the dynamical
    realization of the cut-position-independent time step that the static
    spectral radius only predicts.

Structure preservation in time (``advection_ke_witness``).
    On the aggregated space the skew-symmetric summation-by-parts advection
    operator ``C = 1/2 (Q - Q^T)`` is advanced with the implicit-midpoint rule,
    which preserves the quadratic invariant ``1/2 u^T H u`` exactly for a system
    whose generator is ``H``-skew.  The discrete kinetic energy is therefore held
    at roundoff for all time: the dynamical face of the algebraic
    summation-by-parts identity ``Q + Q^T = S``.

Matrices are assembled on a single oblique cut at low resolution, so every
routine here runs in well under a second.
"""

from __future__ import annotations

import os

# Pin the linear-algebra backends to a single thread before NumPy loads so any
# parallel driver that imports this module cannot oversubscribe the machine.
for _thread_var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

from dataclasses import dataclass
from math import cos, sin, sqrt

import numpy as np
from scipy.linalg import eigvalsh

from cut_spline_sbp import make_open_uniform_basis
from oblique_cut_sbp import (
    HalfPlaneCut,
    assemble_oblique_cut_matrices,
    polynomial_preserving_oblique_extension,
    pullback_oblique_2d,
    restrict_oblique_2d,
)

# A blow-up cap for the explicit active run: once the relative energy crosses
# this the trajectory has unambiguously diverged and integration can stop.
_BLOWUP_CAP = 1.0e12


def _sym(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.T)


@dataclass(frozen=True)
class CutEnergyPair:
    """Active and aggregated operator blocks on one oblique cut plus geometry.

    ``p0_active`` and ``p0_aggregated`` are the L2 projections of the same smooth
    manufactured field onto the two spaces.  The active projection necessarily
    carries the small-support coordinate (the small cell overlaps the field), so
    it is the honest initial state that exposes the active assembly instability,
    while the aggregated projection is the smooth state that stays bounded.
    """

    active: object
    aggregated: object
    p0_active: np.ndarray
    p0_aggregated: np.ndarray
    full_to_small_cell_ratio: float
    min_cell_volume_fraction: float
    n_active: int
    n_aggregated: int
    degree: int
    epsilon: float
    n_elements: int


def _min_cell_fraction(volume_quadrature, n_elements: int) -> float:
    areas: dict[int, float] = {}
    for cell_id, weight in zip(volume_quadrature.cell_ids, volume_quadrature.weights):
        areas[int(cell_id)] = areas.get(int(cell_id), 0.0) + float(weight)
    full_area = (1.0 / n_elements) ** 2
    fractions = [
        area / full_area
        for area in areas.values()
        if area > 1.0e-22 and area / full_area < 1.0 - 1.0e-12
    ]
    return float(min(fractions) if fractions else 1.0)


def build_cut_pair(
    degree: int = 4,
    epsilon: float = 2.0e-3,
    n_elements: int = 8,
    angle: float = 0.71,
    support_threshold: float = 0.05,
) -> CutEnergyPair:
    """Assemble one oblique cut and return active and aggregated operator blocks.

    The active set keeps every basis row with any positive mass (``active_tol``
    effectively zero), so it is the genuine unaggregated assembly space whose
    small-support row carries the vanishing mass eigenvalue.  The aggregated
    blocks follow from the polynomial-preserving extension by the congruence
    ``E^T(.)E``.
    """
    quadrature_order = max(28, 2 * degree + 18)
    basis_x = make_open_uniform_basis(n_elements=n_elements, degree=degree)
    basis_y = make_open_uniform_basis(n_elements=n_elements, degree=degree)
    normal = np.array([cos(angle), sin(angle)], dtype=float)
    normal = normal / np.linalg.norm(normal)
    h = 1.0 / n_elements
    offset = float((normal[0] + normal[1]) * h * (1.0 - epsilon))
    cut = HalfPlaneCut(normal=normal, offset=offset)
    matrices = assemble_oblique_cut_matrices(
        basis_x, basis_y, cut, quadrature_order=quadrature_order,
        capture_volume_quadrature=True,
    )
    agg = polynomial_preserving_oblique_extension(
        basis_x, basis_y, matrices,
        support_threshold=support_threshold,
        preserve_degree=degree,
        active_tol=1.0e-100,
    )
    active = restrict_oblique_2d(matrices, agg.active)
    aggregated = pullback_oblique_2d(matrices, agg.E)
    min_frac = _min_cell_fraction(matrices.volume_quadrature, n_elements)

    # L2 projection of a smooth manufactured field onto each space.  The load is
    # b_i = integral phi_i g over the cut domain; the projection is H^{-1} b.
    vq = matrices.volume_quadrature
    g = np.sin(np.pi * vq.x) * np.sin(np.pi * vq.y)
    b_full = vq.basis_values @ (vq.weights * g)
    p0_active = np.linalg.solve(_sym(active.H), b_full[agg.active])
    p0_aggregated = np.linalg.solve(_sym(aggregated.H), agg.E.T @ b_full)

    return CutEnergyPair(
        active=active,
        aggregated=aggregated,
        p0_active=p0_active,
        p0_aggregated=p0_aggregated,
        full_to_small_cell_ratio=float(1.0 / min_frac),
        min_cell_volume_fraction=float(min_frac),
        n_active=int(len(agg.active)),
        n_aggregated=int(aggregated.H.shape[0]),
        degree=int(degree),
        epsilon=float(epsilon),
        n_elements=int(n_elements),
    )


def _wave_operators(m) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric wave mass M and stiffness K from an operator block."""
    return _sym(m.H), _sym(m.K)


def _omega_max(M: np.ndarray, K: np.ndarray) -> float:
    """Largest natural frequency of M p'' = -K p (generalized symmetric eig)."""
    lam = eigvalsh(K, M)
    return float(sqrt(max(lam[-1], 0.0)))


def _verlet_energy(M, K, p0, dt, n_steps):
    """Stormer-Verlet trajectory of M p'' = -K p; return relative energy in time.

    The trajectory starts from rest (p = p0, p' = 0).  Integration halts early if
    the relative energy crosses the blow-up cap, which only happens when the step
    exceeds the stability limit of the operator being integrated.
    """
    MinvK = np.linalg.solve(M, K)
    p = p0.astype(float).copy()
    v = np.zeros_like(p)
    acc = -MinvK @ p

    def energy(pp, vv):
        return 0.5 * (vv @ (M @ vv) + pp @ (K @ pp))

    e0 = float(energy(p, v))
    rel = [1.0]
    blew_up = False
    for _ in range(n_steps):
        p = p + dt * v + 0.5 * dt * dt * acc
        acc_next = -MinvK @ p
        v = v + 0.5 * dt * (acc + acc_next)
        acc = acc_next
        ek = float(energy(p, v))
        r = abs(ek) / max(abs(e0), 1.0e-300)
        rel.append(r)
        if not np.isfinite(r) or r > _BLOWUP_CAP:
            blew_up = True
            break
    return np.array(rel), bool(blew_up), float(e0)


def wave_cfl_contrast(
    degree: int = 4,
    epsilon: float = 2.0e-3,
    n_elements: int = 8,
    cfl_safety: float = 0.9,
    n_steps: int = 4000,
    pair: CutEnergyPair | None = None,
) -> dict:
    """Dynamical cut-position-independent time step: active blow-up vs bounded.

    Both spaces are advanced with the same explicit Stormer-Verlet step, sized to
    ``cfl_safety`` times the aggregated stability limit.  The aggregated energy
    stays bounded; the active energy diverges because the same step is far past
    its own (sliver-collapsed) stability limit.
    """
    if pair is None:
        pair = build_cut_pair(degree=degree, epsilon=epsilon, n_elements=n_elements)
    Ma, Ka = _wave_operators(pair.active)
    Mr, Kr = _wave_operators(pair.aggregated)

    omega_active = _omega_max(Ma, Ka)
    omega_aggregated = _omega_max(Mr, Kr)
    dt_max_active = 2.0 / max(omega_active, 1.0e-300)
    dt_max_aggregated = 2.0 / max(omega_aggregated, 1.0e-300)
    dt = cfl_safety * dt_max_aggregated

    # Both runs start from the L2 projection of the same smooth manufactured
    # field, so the contrast is purely the integrator stability at the shared
    # background time step, not the choice of data.
    rel_aggregated, blew_aggregated, _ = _verlet_energy(Mr, Kr, pair.p0_aggregated, dt, n_steps)
    rel_active, blew_active, _ = _verlet_energy(Ma, Ka, pair.p0_active, dt, n_steps)

    return {
        "dt": float(dt),
        "dt_max_active": float(dt_max_active),
        "dt_max_aggregated": float(dt_max_aggregated),
        "cfl_number_active": float(dt / dt_max_active),
        "cfl_number_aggregated": float(dt / dt_max_aggregated),
        "omega_active": float(omega_active),
        "omega_aggregated": float(omega_aggregated),
        "rel_energy_active": rel_active,
        "rel_energy_aggregated": rel_aggregated,
        "active_blew_up": bool(blew_active),
        "aggregated_blew_up": bool(blew_aggregated),
        "max_rel_energy_active": float(np.max(rel_active)),
        "max_rel_energy_aggregated": float(np.max(rel_aggregated)),
        "full_to_small_cell_ratio": float(pair.full_to_small_cell_ratio),
        "degree": int(pair.degree),
        "epsilon": float(pair.epsilon),
        "n_elements": int(pair.n_elements),
    }


def cfl_ratio_sweep(
    degree: int = 4,
    epsilons: tuple[float, ...] = (1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4),
    n_elements: int = 8,
) -> list[dict]:
    """Active and aggregated stable time step versus full/small cell ratio.

    The aggregated stable step is bounded; the active stable step collapses with
    the small cell, so the achievable time step ratio is the dynamical analog of
    the static spectral-radius relief.
    """
    rows = []
    for eps in epsilons:
        pair = build_cut_pair(degree=degree, epsilon=eps, n_elements=n_elements)
        Ma, Ka = _wave_operators(pair.active)
        Mr, Kr = _wave_operators(pair.aggregated)
        dt_active = 2.0 / max(_omega_max(Ma, Ka), 1.0e-300)
        dt_aggregated = 2.0 / max(_omega_max(Mr, Kr), 1.0e-300)
        rows.append({
            "epsilon": float(eps),
            "full_to_small_cell_ratio": float(pair.full_to_small_cell_ratio),
            "dt_max_active": float(dt_active),
            "dt_max_aggregated": float(dt_aggregated),
            "time_step_relief": float(dt_aggregated / dt_active),
        })
    return rows


def wave_energy_witness(
    degree: int = 4,
    epsilon: float = 2.0e-3,
    n_elements: int = 8,
    n_periods: float = 20.0,
    steps_per_period: int = 200,
    pair: CutEnergyPair | None = None,
) -> dict:
    """Wave energy in time on the aggregated space under implicit midpoint.

    The homogeneous wave system ``M p'' = -K p`` is written in first-order form
    and advanced with implicit midpoint, which conserves the quadratic energy
    ``1/2 (v^T M v + p^T K p)`` exactly for the linear system.  The recorded drift
    is the dynamical realization of the algebraic semidiscrete energy identity on
    the aggregated cut space.
    """
    if pair is None:
        pair = build_cut_pair(degree=degree, epsilon=epsilon, n_elements=n_elements)
    M, K = _wave_operators(pair.aggregated)
    r = M.shape[0]
    MinvK = np.linalg.solve(M, K)
    omega_max = _omega_max(M, K)
    period = 2.0 * np.pi / max(omega_max, 1.0e-12)
    dt = period / steps_per_period
    n_steps = int(np.ceil(n_periods * period / dt))

    J = np.zeros((2 * r, 2 * r))
    J[:r, r:] = np.eye(r)
    J[r:, :r] = -MinvK
    I2 = np.eye(2 * r)
    step = np.linalg.solve(I2 - 0.5 * dt * J, I2 + 0.5 * dt * J)

    y = np.concatenate([pair.p0_aggregated, np.zeros(r)])

    def energy(state):
        pp, vv = state[:r], state[r:]
        return 0.5 * (vv @ (M @ vv) + pp @ (K @ pp))

    e0 = float(energy(y))
    rel = [1.0]
    for _ in range(n_steps):
        y = step @ y
        rel.append(float(energy(y)) / max(abs(e0), 1.0e-300))
    rel = np.array(rel)
    return {
        "rel_energy": rel,
        "drift": float(np.max(np.abs(rel - 1.0))),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "period": float(period),
        "n_periods": float(n_periods),
        "full_to_small_cell_ratio": float(pair.full_to_small_cell_ratio),
    }


def advection_ke_witness(
    degree: int = 4,
    epsilon: float = 2.0e-3,
    n_elements: int = 8,
    advection_angle: float = 0.4,
    n_steps: int = 4000,
    n_periods: float = 20.0,
    pair: CutEnergyPair | None = None,
) -> dict:
    """Discrete kinetic energy in time under skew SBP advection (aggregated).

    The skew-symmetric operator ``C = 1/2 (Q_a - Q_a^T)`` generates a system whose
    quadratic invariant ``1/2 u^T H u`` is preserved exactly by implicit midpoint.
    The recorded kinetic-energy drift is the dynamical face of the algebraic
    summation-by-parts identity ``Q + Q^T = S``.
    """
    if pair is None:
        pair = build_cut_pair(degree=degree, epsilon=epsilon, n_elements=n_elements)
    m = pair.aggregated
    H = _sym(m.H)
    ax, ay = cos(advection_angle), sin(advection_angle)
    Q = ax * m.Qx + ay * m.Qy
    C = 0.5 * (Q - Q.T)  # exactly skew-symmetric advection generator

    # The smooth aggregated state; kinetic-energy conservation is a property of
    # the operator pair (H, C), and the smooth initial data keeps it physical.
    u = pair.p0_aggregated.copy()

    # Time step from the advective Courant limit on the aggregated space.
    Hinv = np.linalg.inv(H)
    rho = float(np.max(np.abs(np.linalg.eigvals(Hinv @ C))))
    period = 2.0 * np.pi / max(rho, 1.0e-12)
    dt = n_periods * period / n_steps

    # Implicit midpoint on H u' = -C u:  (H + dt/2 C) u_{n+1} = (H - dt/2 C) u_n.
    lhs = H + 0.5 * dt * C
    rhs = H - 0.5 * dt * C
    step = np.linalg.solve(lhs, rhs)

    def ke(uu):
        return 0.5 * float(uu @ (H @ uu))

    ke0 = ke(u)
    rel = [1.0]
    for _ in range(n_steps):
        u = step @ u
        rel.append(ke(u) / max(ke0, 1.0e-300))
    rel = np.array(rel)
    drift = float(np.max(np.abs(rel - 1.0)))
    return {
        "rel_kinetic_energy": rel,
        "drift": drift,
        "dt": float(dt),
        "n_steps": int(n_steps),
        "skew_residual": float(np.linalg.norm(C + C.T, ord=np.inf)),
        "full_to_small_cell_ratio": float(pair.full_to_small_cell_ratio),
        "degree": int(pair.degree),
        "epsilon": float(pair.epsilon),
        "n_elements": int(pair.n_elements),
    }


if __name__ == "__main__":
    import json

    contrast = wave_cfl_contrast()
    ke = advection_ke_witness()
    sweep = cfl_ratio_sweep()
    out = {
        "wave_cfl_contrast": {
            k: v for k, v in contrast.items()
            if not isinstance(v, np.ndarray)
        },
        "advection_ke_drift": ke["drift"],
        "advection_skew_residual": ke["skew_residual"],
        "cfl_ratio_sweep": sweep,
    }
    print(json.dumps(out, indent=2))
