"""One-command CPU verification of the headline results.

Recomputes, from scratch and in a few minutes on a laptop:

1. Mass-matrix conditioning on a sharp 5-degree oblique cut (offset 0.85 h,
   6 x 6 elements) with and without polynomial-preserving aggregation, plus the
   cut-domain Green (SBP) identity residual, for p = 3, 4, 5.
2. Discrete conservation of compressible Navier-Stokes on a sharply cut 3D box:
   the mass, momentum and total-energy budgets close to round-off, and the
   inviscid kinetic-energy production of the split form vanishes.
3. Observed convergence order of a 3D cut-box Poisson solve (p = 3).

Run with ``make verify`` or ``uv run python code/verify.py``.  Exits nonzero if
any check misses its bound.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

from comp3d_solver import gate2_conservation_ke  # noqa: E402
from cut_3d_sbp import run_poisson_3d_convergence  # noqa: E402
from s6_oblique_sweep import _row  # noqa: E402


def main() -> int:
    t0 = time.time()
    ok = True

    print("1. Conditioning on a 5-degree oblique cut (6 x 6 elements, offset 0.85 h)")
    print("   p   cond(H) aggregated   cond(H) unaggregated   Green residual")
    for p in (3, 4, 5):
        r = _row(p, 6, 5.0, 0.85)
        agg, raw, green = r["cond_H_phys"], r["cond_H_active"], r["bmg_green_residual"]
        print(f"   {p}   {agg:18.3e}   {raw:20.3e}   {green:14.1e}")
        ok &= agg < 1.0e5 and raw / agg > 1.0e3 and green < 1.0e-10

    print("\n2. Compressible Navier-Stokes on a sharply cut 3D box (40 SSP-RK3 steps)")
    g = gate2_conservation_ke(n_steps=40)
    for k in ("mass", "mom_x", "mom_y", "mom_z", "energy"):
        print(f"   {k:7s} budget closure   {g['budget_closure'][k]:.2e}")
    print(f"   inviscid KE production   {g['ke_rate_relative']:.2e}")
    print(f"   free-stream residual     {g['free_stream_residual']:.2e}")
    ok &= g["max_budget_closure"] < 1.0e-13 and g["ke_rate_relative"] < 1.0e-14

    print("\n3. 3D cut-box Poisson solve, p = 3, 4/6/8 elements per axis")
    s = run_poisson_3d_convergence(degree=3, elements=(4, 6, 8), write=False)["summary"]
    print(f"   observed L2 order {s['l2_slope']:.2f} (formal 4), H1 order {s['h1_slope']:.2f} (formal 3)")
    ok &= s["l2_slope"] > 3.5 and s["h1_slope"] > 2.6

    print(f"\n{'all checks passed' if ok else 'CHECK FAILED'} in {time.time() - t0:.0f} s")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
