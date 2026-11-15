import tempfile
import unittest
from pathlib import Path

import numpy as np

from reachpath.catalog import Catalog
from reachpath.cases import canonical
from reachpath.paths import (
    AnalysisStore,
    DAG,
    _design,
    _empirical_mediator_pool,
    _fit_centers,
    _levels,
    analyze,
    assemble,
)
from reachpath.service import Application


class PathAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = Catalog("data")
        cls.rows, cls.timeline = assemble(cls.catalog)
        cls.levels = _levels(cls.rows)

    def test_join_is_one_to_one_and_timeline_ordered(self):
        self.assertEqual(len(self.rows), len(self.catalog.rows("assignment")))
        self.assertEqual(self.timeline["per_user_ordering_violations"], 0)
        self.assertEqual(self.timeline["rule"][0], "baseline.available_at")

    def test_dag_keeps_randomization_semantics(self):
        edges = {(edge["from"], edge["to"]) for edge in DAG["edges"]}
        self.assertIn(("Z", "M"), edges)
        self.assertIn(("Z", "R"), edges)
        self.assertIn(("R", "M"), edges)
        self.assertIn(("R", "Y"), edges)
        randomized = {node["id"] for node in DAG["nodes"] if node["randomized"]}
        self.assertEqual(randomized, {"Z"})

    def test_missing_impressions_never_become_zero(self):
        raw_missing = sum(record["impressions"] is None for record in self.catalog.rows("mediators"))
        self.assertGreater(raw_missing, 0)
        self.assertEqual(sum(row["m"] is None for row in self.rows), raw_missing)
        report = analyze(self.catalog, boot_reps=10, mc_draws=2)
        self.assertEqual(report["sample"]["missing_impressions"], raw_missing)
        self.assertFalse(report["positivity"]["zero_imputation_used"])

    def test_itt_matches_arm_mean_difference(self):
        report = analyze(self.catalog, boot_reps=10, mc_draws=2)
        rates = report["randomization_estimands"]["itt_purchase"]["arm_rates"]
        expected = rates["arm_1"]["purchase_rate"] - rates["arm_0"]["purchase_rate"]
        self.assertAlmostEqual(report["randomization_estimands"]["itt_purchase"]["estimate"],
                               expected, places=10)

    def test_g_formula_decomposition_adds_up(self):
        report = analyze(self.catalog, boot_reps=10, mc_draws=4)
        block = report["interventional_mediation"]
        for residual in block["decomposition_check"].values():
            self.assertAlmostEqual(residual, 0.0, places=9)

    def test_g_formula_total_effect_close_to_itt(self):
        report = analyze(self.catalog, boot_reps=20, mc_draws=4)
        itt = report["randomization_estimands"]["itt_purchase"]["estimate"]
        block = report["interventional_mediation"]
        self.assertLess(abs(block["total_effect"]["point_check"] - itt), 0.02)
        calibration = block["calibration"]["gformula_total_effect_vs_itt"]
        self.assertEqual(calibration["randomized_itt"], itt)
        for arm in ("control", "treated"):
            entry = block["calibration"]["natural_arm_predicted_vs_observed"][arm]
            self.assertLess(abs(entry[next(k for k in entry if k.startswith("predicted"))] -
                                entry["observed_purchase_rate"]), 0.03)

    def test_positivity_flags_lack_of_common_support(self):
        report = analyze(self.catalog, boot_reps=10, mc_draws=2)
        positivity = report["positivity"]
        self.assertGreater(positivity["treated_units_outside_control_support"], 0)
        self.assertGreater(positivity["control_units_outside_treated_support"], 0)
        self.assertIn("不补零", positivity["note"])

    def test_empirical_pool_contains_only_complete_cases(self):
        for arm in (0, 1):
            pool = _empirical_mediator_pool(self.rows, arm)
            expected = sum(row["z"] == arm and row["m"] is not None for row in self.rows)
            self.assertEqual(len(pool), expected)
            self.assertTrue(np.isfinite(pool).all())

    def test_cross_fitting_centers_come_from_training_only(self):
        rng = np.random.default_rng(0)
        folds = np.array_split(rng.permutation(len(self.rows)), 5)
        for fold in folds:
            held = [self.rows[i] for i in fold]
            train = [row for i, row in enumerate(self.rows) if i not in set(fold.tolist())]
            centers = _fit_centers(train)
            matrix = _design(held, centers, self.levels, include_z=True,
                             include_r=True, include_m=True)
            self.assertTrue(np.isfinite(matrix).all())
        # 留出折内不应出现训练折之外的用户
        self.assertEqual(len({row["customer_id"] for row in held}), len(held))

    def test_identification_status_includes_unidentified_targets(self):
        report = analyze(self.catalog, boot_reps=10, mc_draws=2)
        statuses = {item["estimand"]: item["status"] for item in report["identification"]}
        self.assertEqual(statuses["ITT 总效应（Z->Y）"], "已识别")
        self.assertEqual(statuses["自然直接/间接效应 NDE/NIE"], "现有数据不可识别")
        self.assertEqual(statuses["实际领券的因果效应 / CACE"], "现有数据不可识别")
        self.assertEqual(statuses["干预型直接/间接效应 IDE/IIE"], "依赖额外假设")
        self.assertFalse(report["natural_mediation"]["identified"])

    def test_sensitivity_grid_covers_full_rho_range(self):
        report = analyze(self.catalog, boot_reps=10, mc_draws=2)
        grid = report["natural_mediation"]["sensitivity"]["rho_grid"]
        self.assertAlmostEqual(grid[0], -0.9)
        self.assertAlmostEqual(grid[-1], 0.9)
        self.assertTrue(all("rho" in entry for entry in
                            report["natural_mediation"]["sensitivity"]["grid"]))

    def test_naive_regression_is_marked_as_counterexample(self):
        report = analyze(self.catalog, boot_reps=10, mc_draws=2)
        self.assertIn("coef_arm", report["naive_regression_control"])
        self.assertIn("不是因果直接效应", report["naive_regression_control"]["warning"])

    def test_claim_contrast_is_observational_only(self):
        report = analyze(self.catalog, boot_reps=10, mc_draws=2)
        claim = report["coupon_claim"]
        self.assertEqual(claim["claim_rates_by_arm"]["arm_0"]["claim_rate"], 0.0)
        self.assertIn("不是随机对比", claim["contrast_warning"])
        self.assertIn("不可识别", claim["identification_warning"])

    def test_analysis_is_deterministic_under_seed(self):
        first = analyze(self.catalog, seed=7, boot_reps=10, mc_draws=2)
        second = analyze(self.catalog, seed=7, boot_reps=10, mc_draws=2)
        self.assertEqual(first, second)

    def test_report_is_json_safe_and_finite(self):
        import json
        report = analyze(self.catalog, boot_reps=10, mc_draws=2)
        encoded = json.dumps(report, ensure_ascii=False, allow_nan=False)
        self.assertIn("interventional_mediation", encoded)

    def test_analysis_store_round_trip_and_tamper_detection(self):
        import hashlib
        import json
        with tempfile.TemporaryDirectory() as directory:
            store = AnalysisStore(Path(directory))
            report = analyze(self.catalog, boot_reps=10, mc_draws=2)
            record = store.create(report, "复核路径")
            self.assertEqual(store.read(record["id"]), record)
            target = Path(directory) / (record["id"] + ".json")
            envelope = json.loads(target.read_text())
            envelope["content"]["label"] = "被改动"
            target.write_bytes(canonical(envelope))
            with self.assertRaises(ValueError):
                store.read(record["id"])
            with self.assertRaises(KeyError):
                store.read("../etc")


class PathServiceTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.application = Application("data", Path(self.workspace.name) / "state")

    def test_paths_endpoint_and_persisted_record(self):
        status, result = self.application.dispatch(
            "GET", "/paths?boot_reps=8&mc_draws=2&no_sensitivity=1")
        self.assertEqual(status, 200)
        self.assertEqual(result["options"]["boot_reps"], 8)
        self.assertFalse(result["options"]["run_sensitivity"])
        self.assertIn("report", result)
        status, created = self.application.dispatch(
            "POST", "/paths?boot_reps=8&mc_draws=2&no_sensitivity=1",
            {"label": "财务复核"})
        self.assertEqual(status, 201)
        status, listing = self.application.dispatch("GET", "/paths/records")
        self.assertEqual(status, 200)
        self.assertEqual(len(listing["records"]), 1)
        status, restored = self.application.dispatch(
            "GET", "/paths/records/" + created["id"])
        self.assertEqual(status, 200)
        self.assertEqual(restored["report"], result["report"])

    def test_paths_rejects_unknown_fields_and_bad_params(self):
        self.assertEqual(self.application.dispatch(
            "POST", "/paths", {"label": "x", "extra": 1})[0], 400)
        self.assertEqual(self.application.dispatch(
            "GET", "/paths?boot_reps=0")[0], 400)
        self.assertEqual(self.application.dispatch(
            "GET", "/paths?folds=99")[0], 400)
        self.assertEqual(self.application.dispatch(
            "GET", "/paths?boot_reps=notanint")[0], 400)

    def test_unknown_record_returns_404(self):
        self.assertEqual(self.application.dispatch(
            "GET", "/paths/records/" + "0" * 32)[0], 404)


if __name__ == "__main__":
    unittest.main()
