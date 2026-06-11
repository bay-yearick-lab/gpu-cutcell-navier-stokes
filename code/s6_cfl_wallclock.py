"""Section 6 A3.  Wall-clock CFL relief on the 1D heat-on-sliver problem.

The spectral-radius witness (already in ``s6_mass_spectrum.py``) shows that the
no-aggregation active assembly space has unbounded spectral radius as the
small-cell volume fraction shrinks, while the BMG^cut and BMF^cut physical
space keeps ``rho`` at the background scale.  This driver converts that
spectral claim into a *wall-clock* claim that no reader can dismiss.

Problem.  Heat equation ``u_t = kappa u_xx`` on the cut interval
``[0, 1 - sliver_width]`` with Dirichlet BCs and a smooth Gaussian
initial profile.  Wall-clock is the time to advance to ``T_final`` using
explicit RK4 with the largest stable step ``Delta t``.  ``Delta t_stable``
is set from the spectral radius of the second-derivative operator on the
chosen active or physical space.

Two rows on the same axes:

  1. BMG^cut == BMF^cut (the two routes coincide in 1D on the protected
     polynomial sector; the figure labels them as a single curve in this
     panel with an explicit caption note);
  2. No-aggregation active assembly space (the baseline that collapses).

The driver writes ``code/results/s6_cfl_wallclock_summary.json`` and the
figure ``figures/fig_s6_cfl_wallclock.pdf``.  Wall-clock budget is well
under one minute per panel.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import plot_style
from cut_spline_sbp import (
    SplineBasis1D,
    assemble_cut_matrices,
    make_open_uniform_basis,
    polynomial_preserving_extension,
    pullback,
    restrict_to_active,
    active_sets,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "code" / "results"
FIGURES = ROOT / "figures"


# ---------------------------------------------------------------------------
# Numerics: spectral radius and stable Delta t for RK4 on -L (Laplacian)
# ---------------------------------------------------------------------------


def _stable_dt_rk4(L: np.ndarray) -> float:
    """Return the largest stable Delta t for explicit RK4 on u' = -L u."""
    eigs = np.linalg.eigvals(L)
    # For RK4 stability on stiff diffusion, the negative real axis stability
    # boundary is about 2.785.  Use 2.78 / rho_real for safety.
    rho_real = float(np.max(np.abs(eigs.real)))
    if rho_real <= 0.0:
        return float("inf")
    return 2.78 / rho_real


def _rk4_step(L: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    k1 = -L @ u
    k2 = -L @ (u + 0.5 * dt * k1)
    k3 = -L @ (u + 0.5 * dt * k2)
    k4 = -L @ (u + dt * k3)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _run_to(L: np.ndarray, u0: np.ndarray, dt: float, t_final: float, max_steps: int = 200_000) -> tuple[np.ndarray, float, int]:
    """RK4-march u' = -L u to t_final.  Returns (u, wall_clock, n_steps)."""
    n_steps = int(np.ceil(t_final / dt))
    if n_steps > max_steps:
        return u0.copy(), float("inf"), n_steps
    u = u0.copy()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        u = _rk4_step(L, u, dt)
    wall = time.perf_counter() - t0
    return u, wall, n_steps


# ---------------------------------------------------------------------------
# Driver: 1D heat-on-sliver
# ---------------------------------------------------------------------------


def _build_routes(n_elements: int, degree: int, sliver_eps: float) -> dict:
    """Build active and physical Laplacians on [0, 1 - sliver_eps * h].

    No boundary projection is applied: the small-support active basis
    functions remain in the system so the spectrum collapses as the
    sliver shrinks.  Homogeneous Neumann is the implicit BC at both ends
    (the stiffness matrix K = int phi' psi' has natural Neumann).
    """
    basis = make_open_uniform_basis(n_elements=n_elements, degree=degree)
    h = basis.h
    # The sliver lives at the right end with retained width sliver_eps * h
    # (so sliver_eps -> 0 shrinks the cell-inside-domain).  In the original
    # cell [(n-1)/n, 1] the right cut sits at (n-1)/n + sliver_eps * h.
    right = (n_elements - 1) * h + sliver_eps * h
    matrices = assemble_cut_matrices(basis, left=0.0, right=right)
    # Use ultra-loose active tolerance so that every basis function with any
    # nonzero physical mass stays in the active set across the full eps sweep
    # (matches the s6_mass_spectrum.py convention).  Otherwise vanishingly small
    # cells silently drop out below double precision and the active spectrum
    # rebounds.
    active, interior, small, gamma = active_sets(matrices, active_tol=1.0e-100)

    Ha = matrices.H[np.ix_(active, active)]
    Ka = matrices.K[np.ix_(active, active)]
    L_active = np.linalg.solve(Ha, Ka)

    extension = polynomial_preserving_extension(basis, matrices)
    pulled = pullback(matrices, extension.E)
    Hp = pulled.H
    Kp = pulled.K
    L_phys = np.linalg.solve(Hp, Kp)

    # Smooth initial condition on the original (full) basis greville.
    u0_full = np.exp(-((basis.greville - 0.5) / 0.12) ** 2)
    u0_active = u0_full[active]
    # Physical IC: E carries the full basis to the physical (interior)
    # coordinates; the natural projection is E^T @ u0 over the diagonal
    # column scale of E.  We use a Galerkin L2 projection by computing
    # u_phys = H_phys^{-1} E^T H u_active_full.
    H_full = matrices.H
    rhs_phys = extension.E.T @ (H_full @ u0_full)
    u0_phys = np.linalg.solve(Hp, rhs_phys)

    return {
        "basis": basis,
        "h": h,
        "L_active": L_active,
        "L_phys": L_phys,
        "u0_active": u0_active,
        "u0_phys": u0_phys,
        "n_active": int(L_active.shape[0]),
        "n_phys": int(L_phys.shape[0]),
        "extension_residual": extension.polynomial_residual,
        "min_gamma_active": float(np.min(gamma[active])) if len(active) > 0 else float("nan"),
    }


# Largest explicit step count actually timed.  Beyond this the active solve is
# infeasible to run to completion, so its wall-clock is projected (not timed)
# from the measured per-step cost.  The gate and the run cap use this one value,
# so no sliver falls into a dead zone that returns an infinite wall-clock.
MAX_MEASURED_STEPS = 200_000


def _diffuse_compare(eps_list: list[float], degree: int, n_elements: int, t_final: float, kappa: float = 0.1) -> list[dict]:
    rows: list[dict] = []
    measured_cost_per_step: float | None = None
    for sliver_eps in eps_list:
        info = _build_routes(n_elements, degree, sliver_eps)
        # Scale L by kappa to apply the physical diffusivity.
        L_active = kappa * info["L_active"]
        L_phys = kappa * info["L_phys"]

        # Stable Delta t for each method.
        dt_active = _stable_dt_rk4(L_active)
        dt_phys = _stable_dt_rk4(L_phys)

        _, wall_phys, steps_phys = _run_to(L_phys, info["u0_phys"], dt_phys, t_final,
                                           max_steps=MAX_MEASURED_STEPS)

        steps_active = int(np.ceil(t_final / max(dt_active, 1.0e-30)))
        feasible = dt_active >= 1.0e-12 and steps_active <= MAX_MEASURED_STEPS
        if feasible:
            # Time the active solve directly and calibrate the per-step cost.
            _, wall_active, steps_active = _run_to(L_active, info["u0_active"], dt_active, t_final,
                                                   max_steps=MAX_MEASURED_STEPS)
            measured = True
            if steps_active > 0 and np.isfinite(wall_active):
                measured_cost_per_step = wall_active / steps_active
        else:
            # Project the wall-clock from the measured per-step cost so the curve
            # is continuous; the active solve is infeasible to time at this
            # sliver width (the stable step forces n_steps to explode).
            cps = measured_cost_per_step
            if cps is None:
                cps = max(wall_phys / max(steps_phys, 1), 1.0e-7) * (info["n_active"] / max(info["n_phys"], 1)) ** 2
            wall_active = steps_active * cps
            measured = False

        rho_active = 1.0 / dt_active * 2.78 if np.isfinite(dt_active) else float("inf")
        rho_phys = 1.0 / dt_phys * 2.78 if np.isfinite(dt_phys) else float("inf")

        rows.append({
            "sliver_eps": sliver_eps,
            "h": info["h"],
            "n_active": info["n_active"],
            "n_phys": info["n_phys"],
            "dt_active": dt_active,
            "dt_phys": dt_phys,
            "rho_active": rho_active,
            "rho_phys": rho_phys,
            "wall_active": wall_active,
            "wall_phys": wall_phys,
            "wall_active_measured": measured,
            "steps_active": steps_active,
            "steps_phys": steps_phys,
            "cfl_relief_dt": dt_phys / max(dt_active, 1.0e-30),
            "wall_relief": wall_active / max(wall_phys, 1.0e-30),
            "extension_residual": info["extension_residual"],
        })
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot(rows: list[dict], degree: int, t_final: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=plot_style.WIDE_SIZE)
    eps_arr = np.array([r["sliver_eps"] for r in rows])

    # Panel (a): largest stable Delta t vs sliver epsilon.
    ax = axes[0]
    dt_a = np.array([r["dt_active"] for r in rows])
    dt_p = np.array([r["dt_phys"] for r in rows])
    ax.loglog(eps_arr, dt_a, marker="o", lw=1.5, color=plot_style.NAIVE)
    ax.loglog(eps_arr, dt_p, marker="s", lw=1.7, color=plot_style.LOCAL_GRAPH)
    ax.set_xlabel(r"sliver cell width fraction $\epsilon$")
    ax.set_ylabel(r"largest stable $\Delta t$ (RK4 on $u_t = \kappa u_{xx}$)")
    ax.set_title(fr"(a) stable $\Delta t$ vs cut-cell width, $p={degree}$")
    ax.invert_xaxis()
    plot_style.style_axes(ax)

    # Panel (b): wall-clock to T = t_final.  The active curve is continuous;
    # filled markers are timed runs, open markers are projected from the
    # measured per-step cost where the active solve is infeasible to run.
    ax = axes[1]
    wa = np.array([r["wall_active"] for r in rows])
    wp = np.array([r["wall_phys"] for r in rows])
    measured = np.array([bool(r.get("wall_active_measured", True)) for r in rows])
    ax.loglog(eps_arr, wa, ls="-", lw=1.5, color=plot_style.NAIVE, zorder=2)
    ax.loglog(eps_arr[measured], wa[measured], ls="None", marker="o", ms=6.0,
              color=plot_style.NAIVE, zorder=3)
    ax.loglog(eps_arr[~measured], wa[~measured], ls="None", marker="o", ms=6.0,
              mfc="white", mec=plot_style.NAIVE, mew=1.2, zorder=3)
    ax.loglog(eps_arr, wp, marker="s", lw=1.7, color=plot_style.LOCAL_GRAPH)
    ax.set_xlabel(r"sliver cell width fraction $\epsilon$")
    ax.set_ylabel(fr"wall-clock to $T = {t_final:g}$  (s)")
    ax.set_title(fr"(b) wall-clock cost vs cut-cell width, $p={degree}$")
    ax.invert_xaxis()
    plot_style.style_axes(ax)

    handles = [
        Line2D([0], [0], color=plot_style.NAIVE, marker="o", lw=1.5,
               label=plot_style.LABEL_ACTIVE),
        Line2D([0], [0], color=plot_style.LOCAL_GRAPH, marker="s", lw=1.7,
               label=plot_style.LABEL_PHYSICAL + r" (BMG$^{\rm cut}$ = BMF$^{\rm cut}$)"),
        Line2D([0], [0], color=plot_style.NAIVE, marker="o", ls="None", ms=6.0,
               mfc="white", mec=plot_style.NAIVE, mew=1.2, label="projected (infeasible to time)"),
    ]
    plot_style.legend_top(fig, handles, ncol=3, y=1.0)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    plot_style.save_pdf_png(fig, FIGURES / "fig_s6_cfl_wallclock.pdf")
    plt.close(fig)


def replot_from_cache() -> None:
    """Regenerate the CFL wall-clock figure from cached summary JSON."""
    s = json.loads((RESULTS / "s6_cfl_wallclock_summary.json").read_text(encoding="utf-8"))
    _plot(s["rows"], int(s["degree"]), float(s["t_final"]))


def main() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    degree = 4
    n_elements = 8
    eps_list = [0.5, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 1.0e-5]
    t_final = 0.05  # short enough to be wall-clock comparable, long enough to be physical

    rows = _diffuse_compare(eps_list, degree, n_elements, t_final)
    _plot(rows, degree, t_final)

    summary = {
        "problem": "1D heat on cut interval [0, 1 - eps*h], RK4 to T_final",
        "degree": degree,
        "n_elements": n_elements,
        "t_final": t_final,
        "kappa": 0.1,
        "eps_list": eps_list,
        "max_dt_relief_phys_over_active": max(r["cfl_relief_dt"] for r in rows),
        "max_wall_relief": max(r["wall_relief"] for r in rows),
        "min_dt_phys": min(r["dt_phys"] for r in rows),
        "max_rho_active": max(r["rho_active"] for r in rows),
        "max_rho_phys": max(r["rho_phys"] for r in rows),
        "max_extension_residual": max(r["extension_residual"] for r in rows),
        "rows": rows,
    }
    (RESULTS / "s6_cfl_wallclock_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    s = main()
    print(f"[s6_cfl_wallclock] max Delta t relief (phys/active): {s['max_dt_relief_phys_over_active']:.3e}")
    print(f"[s6_cfl_wallclock] max wall-clock relief:            {s['max_wall_relief']:.3e}")
    print(f"[s6_cfl_wallclock] BMG/BMF physical-space rho max:   {s['max_rho_phys']:.3e}")
    print(f"[s6_cfl_wallclock] active rho max:                   {s['max_rho_active']:.3e}")
