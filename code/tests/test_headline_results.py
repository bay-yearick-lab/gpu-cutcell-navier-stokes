"""Numerical validation of the headline claims.

Two kinds of checks:

* live recomputations that run in seconds: conservation of the compressible
  cut-cell scheme to round-off, the SBP (Green) identity on aggregated
  coordinates, and the conditioning gap between aggregated and unaggregated
  spaces;
* consistency checks on the committed result files that back every number in
  the README (convergence slopes, conservation ledgers, GPU vs CPU agreement).
"""

import csv
import json
from pathlib import Path

import pytest

from comp3d_solver import gate2_conservation_ke
from s6_oblique_sweep import _row

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "code" / "results"
BENCH = ROOT / "benchmarks" / "results"


def _load(name: str, base: Path = RESULTS) -> dict:
    return json.loads((base / name).read_text())


# ---------------------------------------------------------------- live checks


def test_compressible_conservation_to_roundoff():
    g = gate2_conservation_ke(n_steps=20)
    assert g["volume_fraction"] < 0.1  # genuinely cut corner sliver
    for k in ("mass", "mom_x", "mom_y", "mom_z", "energy"):
        assert g["budget_closure"][k] < 1.0e-13, k
    assert g["ke_rate_relative"] < 1.0e-15
    assert g["free_stream_residual"] < 1.0e-12


def test_aggregation_fixes_conditioning_and_keeps_green_identity():
    r = _row(3, 6, 5.0, 0.85)
    assert r["bmg_green_residual"] < 1.0e-13
    assert r["bmf_mixed_green_residual"] < 1.0e-12
    assert r["skew_energy_residual"] < 1.0e-13
    assert r["cond_H_phys"] < 1.0e4
    assert r["cond_H_active"] / r["cond_H_phys"] > 1.0e4


# ---------------------------------------------------------- committed results


@pytest.mark.parametrize("p,formal", [(4, 5), (6, 7), (8, 9)])
def test_curved_cut_poisson_order_reaches_p_plus_1(p, formal):
    deg = _load("highp_pde_solve_ladder.json")["degrees"][str(p)]
    assert deg["expected_order"] == formal
    assert deg["well"]["observed_l2_order"] > formal


def test_conditioning_uniform_across_72_cuts():
    s = _load("s6_oblique_sweep_summary.json")
    assert len(s["rows"]) == 72
    ratios = {p: v["cond_H_phys_ratio_max_over_min"] for p, v in s["per_p"].items()}
    assert max(ratios.values()) < 3.71
    assert s["per_p"]["5"]["max_cond_H_active"] > 1.0e11
    assert s["overall_max_bmg_green"] < 1.0e-10


def test_conservation_ledgers_close_at_roundoff():
    c = _load("comp3d.json")["gate2_conservation_ke"]
    assert c["max_budget_closure"] < 1.0e-14
    assert c["ke_rate_relative"] < 1.0e-17
    t = _load("tg3d.json")["inviscid"]
    assert t["max_ledger_residual_rel"] < 1.0e-15
    assert t["max_divergence_residual"] < 1.0e-16
    cyl = _load("cylinder_benchmark_p3_fine_summary.json")
    assert cyl["max_divergence_residual"] < 1.0e-16
    assert 1.0 / cyl["min_cell_volume_fraction"] > 1900


def test_small_cell_stress_never_fails():
    s = _load("fast_solver_stress_summary.json")
    assert s["n_cases"] == 25 and s["all_converged"]
    assert s["volume_ratio_range"][1] >= 0.99e5
    assert s["max_relative_residual"] < 1.0e-8


def test_gpu_and_cpu_pcg_agree():
    rows = list(csv.DictReader((BENCH / "gpu_solver_benchmark_3d.csv").open()))
    assert len(rows) == 4
    for r in rows:
        assert r["gpu_success"] == "True"
        assert abs(int(r["cpu_cg_iterations"]) - int(r["gpu_cg_iterations"])) <= 1
        assert float(r["cpu_cg_residual"]) < 1.0e-8 and float(r["gpu_cg_residual"]) < 1.0e-8
        assert float(r["gpu_speedup_vs_cpu_cg"]) > 8.0
