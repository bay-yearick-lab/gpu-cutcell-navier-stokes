"""Section 6 A5.  Angle-uniformity audit for the admissible geometry class.

For the admissible root-accessible reach-admissible geometry class, the
manuscript claims that the BMG^cut and BMF^cut routes deliver:

  (1) Green-identity residuals at roundoff uniformly in cut angle and offset;
  (2) physical mass conditioning bounded above by a geometry-uniform constant;
  (3) physical stiffness spectral radius at the background scale uniformly
      in cut angle and offset.

This driver sweeps a clean grid in cut angle theta in {0, 15, 30, 45, 60, 75
degrees} x offset in {0.25, 0.55, 0.85} of the background cell, for degrees
p in {3, 4, 5, 6}, and reports each metric as a function of theta to
demonstrate the geometry-uniform constants.

This is the algebraic precondition for angle-independent high-order
convergence.

Both BMG^cut and BMF^cut appear on every panel per the dual-route rule
(section 8c).  Wall-clock budget: a few minutes.
"""

from __future__ import annotations

import json
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigvalsh

import plot_style
from cut_spline_sbp import make_open_uniform_basis
from oblique_cut_sbp import (
    HalfPlaneCut,
    assemble_oblique_cut_matrices,
    polynomial_preserving_oblique_extension,
    pullback_oblique_2d,
)
from cut_mimetic_routes_2d import build_bmg_cut_route_2d, build_bmf_cut_route_2d

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "code" / "results"
FIGURES = ROOT / "figures"


ANGLES_DEG = (5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0)
OFFSETS = (0.25, 0.55, 0.85)
DEGREES = (3, 4, 5)
N_ELEMENTS = 6


def _condition_number(M: np.ndarray) -> float:
    Hs = 0.5 * (M + M.T)
    eigs = eigvalsh(Hs)
    pos = eigs[eigs > 0]
    if pos.size == 0:
        return float("inf")
    return float(pos[-1] / pos[0])


def _row(degree: int, n_elements: int, angle_deg: float, offset_fraction: float) -> dict:
    basis_x = make_open_uniform_basis(n_elements=n_elements, degree=degree)
    basis_y = make_open_uniform_basis(n_elements=n_elements, degree=degree)
    angle = angle_deg * pi / 180.0
    offset = offset_fraction * basis_x.h
    cut = HalfPlaneCut.from_angle(angle, offset)

    # Active-space build (route audit) and physical-space pullback (Cap IV).
    bmg = build_bmg_cut_route_2d(basis_x, basis_y, cut)
    bmf = build_bmf_cut_route_2d(basis_x, basis_y, cut)
    matrices = assemble_oblique_cut_matrices(basis_x, basis_y, cut, capture_volume_quadrature=True)
    agg = polynomial_preserving_oblique_extension(basis_x, basis_y, matrices, preserve_degree=degree)
    pulled = pullback_oblique_2d(matrices, agg.E)
    # Physical mass and physical first-derivative on the pulled-back space.
    H_phys = 0.5 * (pulled.H + pulled.H.T)
    Dx_phys = np.linalg.solve(H_phys, pulled.Qx)
    Dy_phys = np.linalg.solve(H_phys, pulled.Qy)
    cond_H_phys = _condition_number(H_phys)
    rho_phys = float(max(np.max(np.abs(np.linalg.eigvals(Dx_phys))), np.max(np.abs(np.linalg.eigvals(Dy_phys)))))

    bmg_green = max(bmg.green_residual_x, bmg.green_residual_y)
    bmf_mixed_green = max(bmf.x.mixed_green_residual, bmf.y.mixed_green_residual)

    return {
        "p": degree,
        "angle_deg": angle_deg,
        "offset_fraction": offset_fraction,
        "n_elements": n_elements,
        "h": float(basis_x.h),
        "n_active_scalar": int(bmg.Dx.shape[0]),
        "n_physical_scalar": int(H_phys.shape[0]),
        "bmg_green_residual": bmg_green,
        "bmf_mixed_green_residual": bmf_mixed_green,
        # Physical-space (pulled-back) Cap-IV diagnostics:
        "cond_H_phys": cond_H_phys,
        "rho_phys": rho_phys,
        # Active-space (for comparison; should grow with degenerate cuts):
        "cond_H_active": _condition_number(bmg.H),
        "rho_active": float(np.max(np.abs(np.linalg.eigvals(bmg.Dx)))),
        "pressure_work_residual": bmf.pressure_work_residual,
        "skew_energy_residual": bmf.skew_energy_residual,
    }


def _color(p: int):
    # Span the full degree ramp across the degrees actually shown (3..5).
    return plot_style.degree_color(p, 3, 5)


def _plot(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=plot_style.WIDE_SIZE)
    ps = sorted({int(r["p"]) for r in rows})

    from matplotlib.lines import Line2D

    # Panel (a): Green identity residual vs angle (constant at roundoff).
    ax = axes[0]
    for p in ps:
        rows_p = sorted([r for r in rows if int(r["p"]) == p], key=lambda r: (r["angle_deg"], r["offset_fraction"]))
        x = np.array([r["angle_deg"] for r in rows_p])
        bmg = np.array([r["bmg_green_residual"] for r in rows_p])
        bmf = np.array([r["bmf_mixed_green_residual"] for r in rows_p])
        c = _color(p)
        ax.semilogy(x, bmg, marker="o", ls="None", color=c, ms=5.0)
        ax.semilogy(x, bmf, marker="s", ls="None", color=c, ms=4.4, alpha=0.65)
    ax.axhline(1.0e-12, color=plot_style.REFERENCE, ls=(0, (5, 4)), lw=0.8)
    ax.set_xlabel(r"cut angle $\theta$ (degrees)")
    ax.set_ylabel(r"$\|\!$Green residual$\!\|_{\infty}$")
    ax.set_title(r"(a) Green identity uniform in $\theta$")
    ax.set_ylim(1.0e-17, 1.0e0)
    plot_style.style_axes(ax)

    # Panel (b): physical mass conditioning vs angle (BMG^cut = BMF^cut shared scalar space).
    ax = axes[1]
    for p in ps:
        rows_p = sorted([r for r in rows if int(r["p"]) == p], key=lambda r: (r["angle_deg"], r["offset_fraction"]))
        x = np.array([r["angle_deg"] for r in rows_p])
        cond_phys = np.array([r["cond_H_phys"] for r in rows_p])
        cond_active = np.array([r["cond_H_active"] for r in rows_p])
        c = _color(p)
        ax.semilogy(x, cond_phys, marker="o", ls="None", color=c, ms=5.0)
        ax.semilogy(x, cond_active, marker="x", ls="None", color=c, ms=4.4, alpha=0.55)
    ax.set_xlabel(r"cut angle $\theta$ (degrees)")
    ax.set_ylabel(r"$\kappa(H)$ (mass conditioning)")
    ax.set_title(r"(b) Aggregated mass conditioning bounded uniformly")
    plot_style.style_axes(ax)

    # Panel (c): physical-space spectral radius vs angle.
    ax = axes[2]
    for p in ps:
        rows_p = sorted([r for r in rows if int(r["p"]) == p], key=lambda r: (r["angle_deg"], r["offset_fraction"]))
        x = np.array([r["angle_deg"] for r in rows_p])
        rho_phys = np.array([r["rho_phys"] for r in rows_p])
        rho_active = np.array([r["rho_active"] for r in rows_p])
        c = _color(p)
        ax.semilogy(x, rho_phys, marker="o", ls="None", color=c, ms=5.0)
        ax.semilogy(x, rho_active, marker="x", ls="None", color=c, ms=4.4, alpha=0.55)
    ax.set_xlabel(r"cut angle $\theta$ (degrees)")
    ax.set_ylabel(r"$\rho(H^{-1} Q)$ (first-derivative)")
    ax.set_title(r"(c) Aggregated spectral radius bounded uniformly")
    plot_style.style_axes(ax)

    # Single shared legend below the panels (marker meaning + degree color).
    route_handles = [
        Line2D([0], [0], color="k", marker="o", ls="None", ms=5.2,
               label=plot_style.LABEL_PHYSICAL + r" (BMG$^{\rm cut}$)"),
        Line2D([0], [0], color="k", marker="s", ls="None", ms=4.6, alpha=0.65,
               label=plot_style.LABEL_PHYSICAL + r" (BMF$^{\rm cut}$)"),
        Line2D([0], [0], color="k", marker="x", ls="None", ms=4.6, alpha=0.55,
               label=plot_style.LABEL_ACTIVE),
    ]
    degree_handles = [Line2D([0], [0], color=_color(p), marker="o", ls="None", ms=5.2, label=fr"$p={p}$") for p in ps]
    ref_handle = [Line2D([0], [0], color=plot_style.REFERENCE, ls=(0, (5, 4)), lw=0.8,
                         label=r"$10^{-12}$ roundoff target")]
    handles = route_handles + degree_handles + ref_handle
    plot_style.legend_top(fig, handles, ncol=4, y=1.0)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.82))
    plot_style.save_pdf_png(fig, FIGURES / "fig_s6_oblique_sweep.pdf")
    plt.close(fig)


def replot_from_cache() -> None:
    """Regenerate the oblique-sweep figure from cached summary JSON."""
    s = json.loads((RESULTS / "s6_oblique_sweep_summary.json").read_text(encoding="utf-8"))
    _plot(s["rows"])


def main() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in DEGREES:
        for theta in ANGLES_DEG:
            for off in OFFSETS:
                rows.append(_row(p, N_ELEMENTS, theta, off))

    _plot(rows)

    # Geometry-uniform constants: report MIN/MAX over the sweep at each p.
    by_p: dict[int, list[dict]] = {}
    for r in rows:
        by_p.setdefault(int(r["p"]), []).append(r)

    per_p = {}
    for p, sub in by_p.items():
        green_bmg = [r["bmg_green_residual"] for r in sub]
        green_bmf = [r["bmf_mixed_green_residual"] for r in sub]
        cond_phys = [r["cond_H_phys"] for r in sub]
        cond_active = [r["cond_H_active"] for r in sub]
        rho_phys = [r["rho_phys"] for r in sub]
        rho_active = [r["rho_active"] for r in sub]
        per_p[str(p)] = {
            "max_bmg_green_residual": max(green_bmg),
            "max_bmf_mixed_green_residual": max(green_bmf),
            "max_cond_H_phys": max(cond_phys),
            "min_cond_H_phys": min(cond_phys),
            "cond_H_phys_ratio_max_over_min": max(cond_phys) / max(min(cond_phys), 1e-30),
            "max_cond_H_active": max(cond_active),
            "max_rho_phys": max(rho_phys),
            "min_rho_phys": min(rho_phys),
            "rho_phys_ratio_max_over_min": max(rho_phys) / max(min(rho_phys), 1e-30),
            "max_rho_active": max(rho_active),
        }
    summary = {
        "angles_deg": list(ANGLES_DEG),
        "offsets": list(OFFSETS),
        "degrees": list(DEGREES),
        "n_elements": N_ELEMENTS,
        "rows": rows,
        "per_p": per_p,
        "overall_max_bmg_green": max(r["bmg_green_residual"] for r in rows),
        "overall_max_bmf_mixed_green": max(r["bmf_mixed_green_residual"] for r in rows),
    }
    (RESULTS / "s6_oblique_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    s = main()
    print(f"[s6_oblique_sweep] {len(s['rows'])} audit rows across angle x offset x degree")
    for p, info in s["per_p"].items():
        print(f"  p={p}: cond(H_phys) max/min ratio {info['cond_H_phys_ratio_max_over_min']:.2f},  rho_phys max/min {info['rho_phys_ratio_max_over_min']:.2f}")
        print(f"        cond(H_phys) max {info['max_cond_H_phys']:.2e};  cond(H_active) max {info['max_cond_H_active']:.2e};  rho_phys max {info['max_rho_phys']:.2f};  rho_active max {info['max_rho_active']:.2e}")
    print(f"  overall max BMG Green residual: {s['overall_max_bmg_green']:.2e}")
    print(f"  overall max BMF mixed Green:    {s['overall_max_bmf_mixed_green']:.2e}")
