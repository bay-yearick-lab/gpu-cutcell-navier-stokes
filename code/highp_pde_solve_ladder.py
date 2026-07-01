"""High-order PDE solve on a sharp curved cut: conditioned vs stock extension.

Companion to ``highp_degree_ladder`` (which measures the best-approximation order
of the cut SPACE by a weighted-L2 projection).  This driver runs the actual
weak-penalty-Dirichlet Poisson SOLVE on the self-similar curved corner cut and shows
that swapping the stock global-monomial extension for the locally conditioned
extension (``highp_cut_basis.well_conditioned_circular_extension``) lifts the
REALIZED solve order at high spline degree.

Story it tells (head to head, identical solver, identical geometry):
  * STOCK extension: the realized L2 order saturates near 4-5 for p >= 6 and the
    error floors near 1e-9..1e-10, because the global-monomial moment matrix is
    conditioned ~1e13 at high degree.
  * WELL-conditioned extension: the realized order climbs and the floor drops by
    orders of magnitude on the identical solve.

The residual gap to the ideal p+1 in this panel is the DOUBLE-PRECISION weak-penalty
stiffness conditioning (the solve here is float64); the projection ladder, run in
extended precision, shows the cut space itself supports the full p+1 through
degree 12, so that remaining gap is precision, not approximation power.

Self-similar geometry (radius = c*h) holds the corner sliver at a fixed fluid
fraction on every mesh, isolating the algebraic order from grazing artifacts.

Writes results/highp_pde_solve_ladder.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

from cut_pde_solve import solve_poisson_circular
from circular_corner_sbp import polynomial_preserving_circular_extension
from highp_cut_basis import well_conditioned_circular_extension

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "code" / "results"

RADIUS_OVER_H = 1.3

# Meshes per degree chosen to span the asymptotic window above the float64 solve
# floor (high p reaches the stiffness-conditioning floor on finer meshes).
DEGREE_PLAN: dict[int, tuple[int, ...]] = {
    4: (8, 12, 16, 24, 32),
    6: (6, 8, 12, 16, 24),
    8: (5, 6, 8, 12, 16),
    10: (4, 5, 6, 8, 12),
}

BUILDERS = {
    "stock": polynomial_preserving_circular_extension,
    "well": well_conditioned_circular_extension,
}


def _fit_slope(ns, errs):
    h = 1.0 / np.asarray(ns, dtype=float)
    return float(np.polyfit(np.log(h), np.log(np.asarray(errs, dtype=float)), 1)[0])


def _pairwise(ns, errs):
    out = []
    for i in range(1, len(ns)):
        out.append(round(float(np.log(errs[i - 1] / errs[i]) / np.log(ns[i] / ns[i - 1])), 3))
    return out


def run(plan=None, log=None, builders=None):
    plan = plan or DEGREE_PLAN
    builders = builders or BUILDERS
    rows = []
    summary = {}

    def _log(m):
        if log is not None:
            log.write(m + "\n"); log.flush()
        print(m, flush=True)

    for p in sorted(plan):
        ns = list(plan[p])
        summary[str(p)] = {"expected_order": p + 1, "radius_over_h": RADIUS_OVER_H, "meshes": ns}
        for tag, builder in builders.items():
            errs, linfs, conds = [], [], []
            for n in ns:
                t0 = time.time()
                radius = RADIUS_OVER_H / n
                case = solve_poisson_circular(p=p, n_elements=n, radius=radius,
                                              extension_builder=builder)
                errs.append(float(case.relative_l2_error))
                linfs.append(float(case.relative_linf_error))
                conds.append(float(case.cond_stiffness))
                rows.append({"p": p, "n": n, "extension": tag, "radius": radius,
                             "relative_l2_error": float(case.relative_l2_error),
                             "relative_linf_error": float(case.relative_linf_error),
                             "cond_stiffness": float(case.cond_stiffness),
                             "n_physical": int(case.n_physical), "seconds": time.time() - t0})
                _log(f"p={p} n={n:>2} [{tag:>5}]: relL2={case.relative_l2_error:.4e} "
                     f"relLinf={case.relative_linf_error:.4e} "
                     f"cond(A)={case.cond_stiffness:.2e} t={time.time()-t0:.0f}s")
            summary[str(p)][tag] = {
                "observed_l2_order": _fit_slope(ns, errs),
                "observed_linf_order": _fit_slope(ns, linfs),
                "pairwise_orders": _pairwise(ns, errs),
                "pairwise_linf_orders": _pairwise(ns, linfs),
                "deepest_error": float(min(errs)),
                "deepest_linf_error": float(min(linfs)),
                "cond_stiffness_max": float(max(conds)),
                "errors": errs,
                "linf_errors": linfs,
            }
            _log(f"==> p={p} [{tag:>5}] order L2={summary[str(p)][tag]['observed_l2_order']:.2f} "
                 f"Linf={summary[str(p)][tag]['observed_linf_order']:.2f} "
                 f"deepest Linf={summary[str(p)][tag]['deepest_linf_error']:.2e}")
    out = {"geometry": f"circular corner cut, self-similar radius={RADIUS_OVER_H} h",
           "note": "float64 weak-penalty solve; projection ladder (longdouble) shows space supports p+1",
           "degrees": summary, "rows": rows}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "highp_pde_solve_ladder.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--degrees", type=str, default=None, help="comma list, e.g. 6,8,10")
    ap.add_argument("--extension", type=str, default=None, help="stock|well (default both)")
    ap.add_argument("--log", type=str, default=None)
    args = ap.parse_args()
    plan = DEGREE_PLAN
    if args.degrees:
        keep = {int(d) for d in args.degrees.split(",")}
        plan = {p: m for p, m in DEGREE_PLAN.items() if p in keep}
    builders = BUILDERS
    if args.extension:
        builders = {args.extension: BUILDERS[args.extension]}
    log = open(args.log, "a", encoding="utf-8") if args.log else None
    try:
        out = run(plan=plan, log=log, builders=builders)
    finally:
        if log:
            log.close()
    print(json.dumps(out["degrees"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
