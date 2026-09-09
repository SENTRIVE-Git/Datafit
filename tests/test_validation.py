import unittest

import numpy as np
import pandas as pd

from datafit.diagnostics import diagnose
from datafit.readiness import assess_readiness
from datafit.experiments import run_imbalance_experiment, run_synthetic_data_experiment, run_regression_baseline_experiment
from sklearn.datasets import load_breast_cancer


def base_frame(rows=1200, seed=7):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "feature_a": rng.normal(0, 1, rows),
        "feature_b": rng.normal(0, 1, rows),
        "category": np.tile(["a", "b", "c"], rows // 3 + 1)[:rows],
        "target": np.tile([0, 1], rows // 2 + 1)[:rows],
    })


class DetectorValidationTests(unittest.TestCase):
    def assert_has(self, report, check):
        self.assertTrue(any(issue.check == check for issue in report.issues), report.summary())

    def assert_not_has(self, report, check):
        self.assertFalse(any(issue.check == check for issue in report.issues), report.summary())

    def test_clean_dataset_has_no_problem_findings(self):
        report = diagnose(base_frame(), target_column="target", run_label_noise_check=False)
        for check in ("missing_values", "duplicates", "low_variance", "high_cardinality",
                      "distribution", "target_leakage", "multicollinearity"):
            self.assert_not_has(report, check)

    def test_severe_imbalance(self):
        frame = base_frame()
        frame["target"] = [1] * 12 + [0] * 1188
        report = diagnose(frame, target_column="target", run_label_noise_check=False)
        self.assert_has(report, "class_imbalance")
        issue = next(i for i in report.issues if i.check == "class_imbalance")
        self.assertEqual(issue.severity, "critical")

    def test_missing_values(self):
        frame = base_frame()
        frame.loc[:119, "feature_a"] = np.nan
        report = diagnose(frame, run_label_noise_check=False)
        self.assert_has(report, "missing_values")

    def test_duplicates(self):
        frame = base_frame()
        frame.loc[100:149] = frame.loc[0:49].to_numpy()
        self.assert_has(diagnose(frame, run_label_noise_check=False), "duplicates")

    def test_constant_column(self):
        frame = base_frame()
        frame["constant"] = 1
        self.assert_has(diagnose(frame, run_label_noise_check=False), "low_variance")

    def test_extreme_outliers(self):
        frame = base_frame()
        frame.loc[:19, "feature_a"] = 25
        self.assert_has(diagnose(frame, run_label_noise_check=False), "distribution")

    def test_high_cardinality_identifier(self):
        frame = base_frame()
        frame["record_id"] = [f"row-{index}" for index in range(len(frame))]
        self.assert_has(diagnose(frame, run_label_noise_check=False), "high_cardinality")

    def test_suspicious_linear_leakage(self):
        frame = base_frame()
        frame["leaky_feature"] = frame["target"] * 100 + 5
        report = diagnose(frame, target_column="target", run_label_noise_check=False)
        self.assert_has(report, "target_leakage")
        issue = next(i for i in report.issues if i.check == "target_leakage")
        self.assertIn("potential leakage indicator", issue.message)
        self.assertEqual(issue.severity, "warning")
        self.assertIn("cannot prove", issue.message)

    def test_split_drift(self):
        frame = base_frame()
        frame["split"] = ["train"] * 600 + ["test"] * 600
        frame.loc[600:, "feature_a"] += 10
        report = diagnose(frame, split_column="split", run_label_noise_check=False)
        self.assert_has(report, "split_drift")

    def test_label_noise_is_only_a_candidate_signal(self):
        frame = base_frame()
        frame["feature_a"] = np.arange(len(frame), dtype=float)
        frame["target"] = (frame["feature_a"] > 600).astype(int)
        flipped = [10, 20, 30, 40, 50, 650, 660, 670, 680, 690]
        frame.loc[flipped, "target"] = 1 - frame.loc[flipped, "target"]
        report = diagnose(frame, target_column="target")
        candidates = [i for i in report.issues if i.check == "label_noise"]
        self.assertTrue(candidates, report.summary())
        self.assertIn("potential label inconsistency", candidates[0].message)

    def test_edge_cases_are_analyzable_or_explicitly_rejected(self):
        one_column = pd.DataFrame({"value": np.arange(20, dtype=float)})
        self.assertTrue(diagnose(one_column, run_label_noise_check=False).analyzable)

        tiny = pd.DataFrame({"value": [1, 2, 3, 4]})
        tiny_report = diagnose(tiny, run_label_noise_check=False)
        self.assertFalse(tiny_report.analyzable)
        self.assertIn("At least", tiny_report.analyzability_reason)

        all_missing = base_frame()
        all_missing["empty_feature"] = np.nan
        self.assert_has(diagnose(all_missing, run_label_noise_check=False), "missing_values")

        single_class = base_frame()
        single_class["target"] = 0
        self.assert_has(diagnose(single_class, target_column="target", run_label_noise_check=False), "class_imbalance")

        categorical_only = pd.DataFrame({
            "color": np.tile(["red", "blue", "green"], 400),
            "target": np.tile(["yes", "no"], 600),
        })
        self.assertTrue(diagnose(categorical_only, target_column="target", run_label_noise_check=False).analyzable)

        regression = base_frame()
        regression["target"] = np.linspace(0, 1, len(regression))
        regression_report = diagnose(regression, target_column="target", run_label_noise_check=False)
        self.assert_not_has(regression_report, "label_noise")

    def test_experiments_preserve_input(self):
        frame = base_frame()
        before = frame.copy(deep=True)
        run_imbalance_experiment(frame, "target")
        pd.testing.assert_frame_equal(frame, before)
        run_synthetic_data_experiment(frame, "target")
        pd.testing.assert_frame_equal(frame, before)

    def test_regression_baseline_reports_expected_metrics(self):
        frame = base_frame()
        frame["target"] = 2 * frame["feature_a"] - frame["feature_b"]
        before = frame.copy(deep=True)
        result = run_regression_baseline_experiment(frame, "target")
        self.assertTrue(result.ran_successfully)
        self.assertEqual({metric["metric"] for metric in result.metrics}, {"mae", "rmse", "r2"})
        self.assertIn("directional baseline", result.caveat)
        pd.testing.assert_frame_equal(frame, before)

    def test_repeated_findings_do_not_multiply_check_weight(self):
        bundle = load_breast_cancer()
        frame = pd.DataFrame(bundle.data, columns=bundle.feature_names)
        frame["target"] = bundle.target
        report = assess_readiness(frame, task="classification", target_column="target")
        self.assertGreater(report.score, 0)
        self.assertLess(report.score, 100)

    def test_low_cardinality_numeric_flags_are_not_treated_as_continuous_outliers(self):
        frame = base_frame()
        frame["binary_flag"] = np.tile([0, 1], len(frame) // 2)
        frame["ordinal_code"] = np.tile([0, 1, 2, 3], len(frame) // 4)
        report = diagnose(frame, run_label_noise_check=False)
        distribution_messages = [i.message for i in report.issues if i.check == "distribution"]
        self.assertFalse(any("binary_flag" in message or "ordinal_code" in message for message in distribution_messages))


if __name__ == "__main__":
    unittest.main()
