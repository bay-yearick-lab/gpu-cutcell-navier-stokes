"""Smoke test for the energy-in-time demonstrations on a sharp oblique cut.

Checks the dynamical claims of Pass 6 on cheap meshes: the active assembly
space blows up at the aggregated time step while the aggregated space stays
bounded, the wave energy and the advective kinetic energy are conserved to
roundoff in time, and the achievable time step relief grows with the small-cell
ratio.  All cases use a single low-resolution oblique cut, so the test runs in
well under the repository time budget.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from cut_energy_in_time import (
    advection_ke_witness,
    build_cut_pair,
    cfl_ratio_sweep,
    wave_cfl_contrast,
    wave_energy_witness,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"


class TestCutEnergyInTime(unittest.TestCase):
    def test_active_blows_up_aggregated_bounded(self) -> None:
        pair = build_cut_pair(degree=3, epsilon=0.05, n_elements=8)
        c = wave_cfl_contrast(pair=pair, cfl_safety=0.5, n_steps=24)
        # The same explicit step is comfortably stable on the aggregated space
        # and far past the active stability limit.
        self.assertLess(c["max_rel_energy_aggregated"], 1.5)
        self.assertFalse(c["aggregated_blew_up"])
        self.assertTrue(c["active_blew_up"])
        self.assertGreater(c["max_rel_energy_active"], 1.0e6)
        self.assertGreater(c["cfl_number_active"], 1.0)
        self.assertLess(c["cfl_number_aggregated"], 1.0)

    def test_wave_energy_conserved_in_time(self) -> None:
        w = wave_energy_witness(degree=3, epsilon=0.05, n_elements=8,
                                n_periods=10.0, steps_per_period=120)
        self.assertLess(w["drift"], 1.0e-10)

    def test_advective_kinetic_energy_conserved_and_skew(self) -> None:
        k = advection_ke_witness(degree=3, epsilon=0.05, n_elements=8, n_steps=1500)
        # The advection generator is exactly skew-symmetric (an algebraic fact),
        # so implicit midpoint holds the kinetic energy at roundoff.
        self.assertEqual(k["skew_residual"], 0.0)
        self.assertLess(k["drift"], 1.0e-10)

    def test_time_step_relief_grows_with_ratio(self) -> None:
        rows = cfl_ratio_sweep(degree=3, epsilons=(1.0e-1, 1.0e-2, 1.0e-3), n_elements=8)
        ratios = [r["full_to_small_cell_ratio"] for r in rows]
        relief = [r["time_step_relief"] for r in rows]
        self.assertEqual(ratios, sorted(ratios))
        self.assertEqual(relief, sorted(relief))
        # The aggregated stable step is essentially constant across the sweep.
        agg = [r["dt_max_aggregated"] for r in rows]
        self.assertLess((max(agg) - min(agg)) / max(agg), 0.05)
        # The active stable step collapses, so the relief is large at the tightest sliver.
        self.assertGreater(relief[-1], 100.0)

    def test_committed_summary_matches_claims(self) -> None:
        summary = RESULTS / "s6_energy_in_time_summary.json"
        if not summary.exists():
            self.skipTest("Run s6_energy_in_time.py to generate the summary.")
        data = json.loads(summary.read_text())
        self.assertEqual(data["advection_skew_residual"], 0.0)
        self.assertLess(data["wave_energy_drift"], 1.0e-10)
        self.assertLess(data["advection_ke_drift"], 1.0e-10)
        self.assertGreater(data["headline_active_max_energy"], 1.0e6)
        self.assertAlmostEqual(data["headline_aggregated_max_energy"], 1.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
