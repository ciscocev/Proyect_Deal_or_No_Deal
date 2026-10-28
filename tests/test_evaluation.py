import unittest

import pandas as pd

from src.model.evaluation import (
    classification_metrics,
    evaluate_naive_bayes,
    majority_class,
    split_by_predictor_groups,
    split_train_test,
)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.dataframe = pd.DataFrame(
            {
                "feature": ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"],
                "number": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                "target": ["yes", "yes", "no", "no", "yes", "yes", "no", "no", "yes", "yes"],
            },
            index=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        )

    def test_split_is_reproducible_and_tracks_source_positions(self):
        first = split_train_test(self.dataframe, "target", test_size=0.2, seed=17)
        second = split_train_test(self.dataframe, "target", test_size=0.2, seed=17)

        self.assertEqual(first.train_positions, second.train_positions)
        self.assertEqual(first.test_positions, second.test_positions)
        self.assertFalse(set(first.train_positions) & set(first.test_positions))
        self.assertEqual(len(first.test), 2)
        self.assertEqual(set(first.train_positions) | set(first.test_positions), set(range(10)))

    def test_stratified_split_retains_both_classes(self):
        split = split_train_test(self.dataframe, "target", test_size=0.2, seed=42)
        self.assertEqual(set(split.train["target"]), {"yes", "no"})
        self.assertEqual(set(split.test["target"]), {"yes", "no"})

    def test_metrics_match_manually_computed_confusion_matrix(self):
        metrics = classification_metrics(
            ["Alto", "Alto", "Bajo", "Bajo"],
            ["Alto", "Bajo", "Bajo", "Alto"],
            labels=["Alto", "Bajo"],
        )

        self.assertEqual(
            metrics["confusion_matrix"],
            {"Alto": {"Alto": 1, "Bajo": 1}, "Bajo": {"Alto": 1, "Bajo": 1}},
        )
        self.assertEqual(metrics["correct"], 2)
        self.assertEqual(metrics["incorrect"], 2)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["per_class"]["Alto"]["precision"], 0.5)
        self.assertEqual(metrics["per_class"]["Alto"]["recall"], 0.5)
        self.assertEqual(metrics["per_class"]["Alto"]["f1"], 0.5)

    def test_metrics_handle_class_without_predictions_and_empty_input(self):
        metrics = classification_metrics(["A", "A"], ["B", "B"], labels=["A", "B"])
        self.assertEqual(metrics["per_class"]["A"]["precision"], 0.0)
        self.assertEqual(metrics["per_class"]["A"]["recall"], 0.0)
        self.assertTrue(metrics["undefined"]["A"]["precision"])
        self.assertTrue(metrics["undefined"]["B"]["recall"])
        with self.assertRaisesRegex(ValueError, "sin observaciones"):
            classification_metrics([], [])

    def test_rejects_unequal_metric_inputs_and_invalid_split(self):
        with self.assertRaisesRegex(ValueError, "misma longitud"):
            classification_metrics(["a"], [])
        with self.assertRaises(ValueError):
            split_train_test(self.dataframe, "target", test_size=1.0)

    def test_training_only_learns_training_vocabulary(self):
        data = pd.DataFrame(
            {"feature": ["known", "known", "only-test-a", "only-test-b"],
             "target": ["A", "A", "B", "B"]}
        )
        result = evaluate_naive_bayes(
            data, "target", ["feature"], test_size=0.5, seed=0
        )
        train_positions = result["split"].train_positions
        test_positions = result["split"].test_positions

        self.assertFalse(set(train_positions) & set(test_positions))
        training_categories = set(result["split"].train["feature"])
        self.assertEqual(set(result["model"].categories["feature"]), training_categories)
        self.assertTrue(
            set(result["split"].test["feature"]) - training_categories
        )
        self.assertTrue(
            any("Categorías desconocidas" in warning for warning in result["prediction_warnings"])
        )

    def test_majority_baseline_uses_training_labels_and_same_test_set(self):
        result = evaluate_naive_bayes(
            self.dataframe, "target", ["feature", "number"], seed=21
        )
        expected_majority = majority_class(result["split"].train["target"].tolist())
        self.assertEqual(result["baseline_class"], expected_majority)
        self.assertEqual(result["baseline_metrics"]["total"], result["metrics"]["total"])
        self.assertEqual(result["test_rows"], len(result["actual"]))
        self.assertEqual(
            result["training_rows"], sum(result["model"].class_counts.values())
        )

    def test_grouped_split_has_no_shared_predictor_combinations(self):
        split = split_by_predictor_groups(
            self.dataframe, "target", ["feature"], test_size=0.2, seed=13
        )
        train_features = set(split.train["feature"])
        test_features = set(split.test["feature"])
        self.assertFalse(train_features & test_features)
        self.assertFalse(set(split.train_positions) & set(split.test_positions))
        self.assertEqual(split.strategy, "predictor_groups")


if __name__ == "__main__":
    unittest.main()