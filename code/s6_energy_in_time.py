"""Section 6 energy-in-time demonstrations on a sharp oblique cut.

Two panels turn the static small-cell spectral facts and the algebraic
summation-by-parts identities into dynamical facts.

  (a) Cut-position-independent time step.  The scalar wave system is advanced on
      the active assembly space and on the aggregated space with the same
      explicit Stormer-Verlet step, sized to the aggregated Courant limit, from
      the L2 projection of one smooth field.  The aggregated energy is bounded;
      the active energy diverges because the same step is far past its own
      stability limit, and the divergence sets in sooner as the small-cell ratio
      grows.

  (b) Energy and kinetic energy conserved in time.  On the aggregated space the
      wave energy under implicit midpoint and the advective kinetic energy under
      the skew-symmetric summation-by-parts operator stay at the roundoff floor
      over a long integration: the dynamical face of the semidiscrete energy
      identity and of ``Q + Q^T = S``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_style
from cut_energy_in_time import (
    advection_ke_witness,
    build_cut_pair,
    cfl_ratio_sweep,
    wave_cfl_contrast,
    wave_energy_witness,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "code" / "results"
FIGURES = ROOT / "figures"

DEGREE = 4
N_ELEMENTS = 8
# epsilons chosen to land near small-cell ratios of roughly 35, 140, and 940
# (the last matches a 940:1 sliver used in the literature).
BLOWUP_EPSILONS = (0.12, 0.06, 0.023)
CFL_SAFETY = 0.5
BLOWUP_STEPS = 16
ENERGY_STEPS = 4000
SWEEP_EPSILONS = (1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5)


def _compute() -> dict:
    contrasts = []
    for eps in BLOWUP_EPSILONS:
        pair = build_cut_pair(degree=DEGREE, epsilon=eps, n_elements=N_ELEMENTS)
        c = wave_cfl_contrast(pair=pair, cfl_safety=CFL_SAFETY, n_steps=BLOWUP_STEPS)
        contrasts.append({
            "ratio": float(pair.full_to_small_cell_ratio),
            "cfl_number_active": float(c["cfl_number_active"]),
            "dt": float(c["dt"]),
            "dt_max_active": float(c["dt_max_active"]),
            "dt_max_aggregated": float(c["dt_max_aggregated"]),
            "rel_energy_active": c["rel_energy_active"].tolist(),
            "rel_energy_aggregated": c["rel_energy_aggregated"].tolist(),
        })

    headline = build_cut_pair(degree=DEGREE, epsilon=BLOWUP_EPSILONS[-1], n_elements=N_ELEMENTS)
    we = wave_energy_witness(pair=headline, n_periods=ENERGY_STEPS // 200, steps_per_period=200)
    ke = advection_ke_witness(pair=headline, n_steps=ENERGY_STEPS, n_periods=20.0)
    sweep = cfl_ratio_sweep(degree=DEGREE, epsilons=SWEEP_EPSILONS, n_elements=N_ELEMENTS)
    return {
        "contrasts": contrasts,
        "wave_energy": {
            "rel": we["rel_energy"].tolist(), "drift": float(we["drift"]),
            "dt": float(we["dt"]), "n_steps": int(we["n_steps"]),
        },
        "advection_ke": {
            "rel": ke["rel_kinetic_energy"].tolist(), "drift": float(ke["drift"]),
            "dt": float(ke["dt"]), "skew_residual": float(ke["skew_residual"]),
            "n_steps": int(ke["n_steps"]),
        },
        "cfl_ratio_sweep": sweep,
    }


def _active_shade(i: int, n: int):
    # Lightest red for the smallest ratio, deepening to the headline sliver.
    t = 0.45 + 0.55 * (i / max(n - 1, 1))
    base = np.array([0.659, 0.196, 0.196])  # SMALL_SUPPORT #a83232
    white = np.array([1.0, 1.0, 1.0])
    return tuple(white + t * (base - white))


def _panel_blowup(ax, contrasts) -> None:
    n = len(contrasts)
    # Aggregated energy from the headline sliver: one bounded curve.
    head = contrasts[-1]
    agg = np.array(head["rel_energy_aggregated"])
    ax.semilogy(np.arange(agg.size), agg, color=plot_style.LOCAL_GRAPH, lw=2.0,
                marker="o", ms=4.0, zorder=4, label=plot_style.LABEL_PHYSICAL)
    for i, c in enumerate(contrasts):
        act = np.array(c["rel_energy_active"])
        color = _active_shade(i, n)
        is_head = i == n - 1
        ax.semilogy(np.arange(act.size), act, color=color,
                    lw=1.9 if is_head else 1.3, marker="^", ms=4.5 if is_head else 3.4,
                    zorder=3, label=plot_style.LABEL_ACTIVE if is_head else None)
    ax.set_xlim(-0.4, BLOWUP_STEPS + 3.0)
    ax.set_ylim(0.3, 1.0e17)
    ax.set_xlabel("explicit time step")
    ax.set_ylabel(r"energy $E_n/E_0$")
    ax.set_title("(a) shared explicit step: active blow-up, aggregated bounded")
    plot_style.style_axes(ax)
    ax.legend(loc="lower right", fontsize=8.6, frameon=False)


def _panel_conservation(ax, wave_energy, advection_ke) -> None:
    we = np.array(wave_energy["rel"])
    ke = np.array(advection_ke["rel"])
    nw = np.arange(we.size)
    nk = np.arange(ke.size)
    ax.semilogy(nw, np.abs(we - 1.0) + 1.0e-18, color=plot_style.PULLBACK, lw=1.5,
                label="wave energy (implicit midpoint)")
    ax.semilogy(nk, np.abs(ke - 1.0) + 1.0e-18, color=plot_style.BOUNDARY, lw=1.5,
                label="advective kinetic energy (skew SBP)")
    ax.axhline(1.0e-10, ls=(0, (1, 3)), color=plot_style.REFERENCE, lw=0.9,
               label=r"$10^{-10}$ audit tolerance")
    ax.set_ylim(1.0e-16, 1.0e-7)
    ax.set_xlim(0, max(we.size, ke.size))
    ax.set_xlabel("time step")
    ax.set_ylabel(r"relative drift $|Q_n/Q_0-1|$")
    ax.set_title("(b) energy and kinetic energy held at roundoff in time")
    plot_style.style_axes(ax)
    ax.legend(loc="upper left", fontsize=8.6, frameon=False)


def _write_figure(data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=plot_style.WIDE_SIZE)
    _panel_blowup(axes[0], data["contrasts"])
    _panel_conservation(axes[1], data["wave_energy"], data["advection_ke"])
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    plot_style.save_pdf_png(fig, FIGURES / "fig_s6_energy_in_time.pdf")
    plt.close(fig)


def run(write: bool = True) -> dict:
    cache = RESULTS / "s6_energy_in_time.json"
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
    else:
        data = _compute()
        if write:
            RESULTS.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(data, indent=2), encoding="utf-8")
    if write:
        FIGURES.mkdir(parents=True, exist_ok=True)
        _write_figure(data)

    headline = data["contrasts"][-1]
    relief = [r for r in data["cfl_ratio_sweep"]]
    summary = {
        "headline_ratio": float(headline["ratio"]),
        "headline_active_cfl_number": float(headline["cfl_number_active"]),
        "headline_active_max_energy": float(max(headline["rel_energy_active"])),
        "headline_aggregated_max_energy": float(max(headline["rel_energy_aggregated"])),
        "wave_energy_drift": float(data["wave_energy"]["drift"]),
        "advection_ke_drift": float(data["advection_ke"]["drift"]),
        "advection_skew_residual": float(data["advection_ke"]["skew_residual"]),
        "max_time_step_relief": float(max(r["time_step_relief"] for r in relief)),
        "max_ratio_in_sweep": float(max(r["full_to_small_cell_ratio"] for r in relief)),
    }
    if write:
        (RESULTS / "s6_energy_in_time_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def replot_from_cache() -> None:
    data = json.loads((RESULTS / "s6_energy_in_time.json").read_text(encoding="utf-8"))
    _write_figure(data)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
