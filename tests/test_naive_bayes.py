import unittest

import pandas as pd

from src.model.naive_bayes import fit_naive_bayes
from src.model.prediction import predict_observation


class NaiveBayesTests(unittest.TestCase):
    def setUp(self):
        self.dataframe = pd.DataFrame(
            {
                "color": ["rojo", "rojo", "azul", "azul"],
                "weather": ["sol", "lluvia", "sol", "lluvia"],
                "class": ["si", "si", "no", "no"],
            }
        )

    def test_priors_and_laplace_conditionals(self):
        model = fit_naive_bayes(self.dataframe, "class", ["color"])

        self.assertEqual(model.prior_probability("si"), 0.5)
        self.assertEqual(model.prior_probability("no"), 0.5)
        self.assertAlmostEqual(model.conditional_probability("color", "rojo", "si"), 0.6)
        self.assertAlmostEqual(model.conditional_probability("color", "rojo", "no"), 0.2)
        self.assertAlmostEqual(model.conditional_probability("color", "verde", "si"), 0.2)

    def test_prediction_is_manually_verifiable_and_normalized(self):
        model = fit_naive_bayes(self.dataframe, "class", ["color"])
        result = predict_observation(model, {"color": "rojo"})

        self.assertEqual(result["predicted_class"], "si")
        self.assertAlmostEqual(result["class_probabilities"]["si"], 0.75)
        self.assertAlmostEqual(result["class_probabilities"]["no"], 0.25)
        self.assertAlmostEqual(sum(result["class_probabilities"].values()), 1.0)
        self.assertEqual(result["prior_probabilities"], {"si": 0.5, "no": 0.5})
        self.assertAlmostEqual(
            result["class_details"]["si"]["predictors"]["color"][
                "conditional_probability"
            ],
            0.6,
        )
        self.assertIn("log_score", result["class_details"]["si"])

    def test_unknown_category_uses_unknown_bucket_and_warning(self):
        model = fit_naive_bayes(self.dataframe, "class", ["color"])
        result = predict_observation(model, {"color": "verde"})

        self.assertAlmostEqual(result["class_probabilities"]["si"], 0.5)
        self.assertTrue(any("desconocidas" in warning for warning in result["warnings"]))
        self.assertTrue(
            result["class_details"]["si"]["predictors"]["color"]["unknown_category"]
        )

    def test_zero_count_conditional_warns_about_low_evidence(self):
        model = fit_naive_bayes(self.dataframe, "class", ["color"])
        result = predict_observation(model, {"color": "rojo"})

        self.assertTrue(
            any("Poca evidencia empírica" in warning for warning in result["warnings"])
        )
        self.assertIn(
            "log_contribution",
            result["class_details"]["no"]["predictors"]["color"],
        )

    def test_multiple_predictors_and_other_dataset(self):
        model = fit_naive_bayes(
            self.dataframe, target="class", predictors=["color", "weather"]
        )
        result = predict_observation(model, {"color": "rojo", "weather": "sol"})
        self.assertEqual(result["predicted_class"], "si")
        self.assertIn("weather", result["class_details"]["si"]["predictors"])

        alternate = pd.DataFrame(
            {"shape": [1, 1, 2, 2], "label": ["A", "A", "B", "B"]}
        )
        alternate_model = fit_naive_bayes(alternate, "label", ["shape"])
        alternate_result = predict_observation(alternate_model, {"shape": 2})
        self.assertEqual(alternate_result["predicted_class"], "B")

    def test_reproducible_fit_and_prediction(self):
        first = fit_naive_bayes(self.dataframe, "class", ["color"])
        second = fit_naive_bayes(self.dataframe, "class", ["color"])
        self.assertEqual(
            predict_observation(first, {"color": "rojo"}),
            predict_observation(second, {"color": "rojo"}),
        )

    def test_missing_values_rejected_by_default_and_can_be_category(self):
        dataframe = self.dataframe.copy()
        dataframe.loc[0, "color"] = None
        with self.assertRaisesRegex(ValueError, "missing_strategy"):
            fit_naive_bayes(dataframe, "class", ["color"])

        model = fit_naive_bayes(
            dataframe, "class", ["color"], missing_strategy="category"
        )
        result = predict_observation(model, {"color": None})
        self.assertIn("predicted_class", result)
        self.assertGreater(
            result["class_details"]["si"]["predictors"]["color"][
                "conditional_probability"
            ],
            0,
        )

    def test_fit_does_not_modify_original_dataframe(self):
        original = self.dataframe.copy(deep=True)
        fit_naive_bayes(self.dataframe, "class", ["color"])
        pd.testing.assert_frame_equal(self.dataframe, original)

    def test_invalid_target_predictors_and_observation_are_rejected(self):
        with self.assertRaises(ValueError):
            fit_naive_bayes(self.dataframe, "missing", ["color"])
        with self.assertRaisesRegex(ValueError, "no deben repetirse"):
            fit_naive_bayes(self.dataframe, "class", ["color", "color"])
        with self.assertRaisesRegex(ValueError, "no puede usarse"):
            fit_naive_bayes(self.dataframe, "class", ["class"])
        with self.assertRaisesRegex(ValueError, "al menos un predictor"):
            fit_naive_bayes(self.dataframe, "class", [])
        model = fit_naive_bayes(self.dataframe, "class", ["color"])
        with self.assertRaisesRegex(ValueError, "observación"):
            predict_observation(model, {"weather": "sol"})

    def test_small_and_single_class_training_adds_warnings(self):
        dataframe = pd.DataFrame({"feature": ["a"], "label": ["only"]})
        model = fit_naive_bayes(dataframe, "label", ["feature"])
        result = predict_observation(model, {"feature": "new"})
        self.assertGreaterEqual(len(result["warnings"]), 2)
        self.assertEqual(result["predicted_class"], "only")


if __name__ == "__main__":
    unittest.main()