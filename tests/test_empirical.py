import unittest

import pandas as pd

from src.probability.empirical import calculate_event_probabilities


class EmpiricalProbabilityTests(unittest.TestCase):
    def test_calculates_simple_joint_conditional_and_bayes_probabilities(self):
        dataframe = pd.DataFrame(
            {
                "weather": ["sunny", "sunny", "rainy", "rainy"],
                "play": ["yes", "no", "yes", "no"],
            }
        )

        result = calculate_event_probabilities(
            dataframe, "weather", ("value", "sunny"), "play", ("value", "yes")
        )

        self.assertEqual(result["conteo_A"], 2)
        self.assertEqual(result["conteo_B"], 2)
        self.assertEqual(result["conteo_conjunto"], 1)
        self.assertEqual(result["P(A)"], 0.5)
        self.assertEqual(result["P(B)"], 0.5)
        self.assertEqual(result["P(A intersección B)"], 0.25)
        self.assertEqual(result["P(A dado B)"], 0.5)
        self.assertEqual(result["P(B dado A)"], 0.5)
        self.assertEqual(result["Bayes P(B dado A)"], 0.5)

    def test_counts_missing_values_and_leaves_input_unchanged(self):
        dataframe = pd.DataFrame(
            {"first": ["x", None, "x"], "second": [None, "y", "y"]}
        )
        original = dataframe.copy(deep=True)

        result = calculate_event_probabilities(
            dataframe, "first", ("missing", None), "second", ("value", "y")
        )

        self.assertEqual(result["conteo_A"], 1)
        self.assertEqual(result["conteo_B"], 2)
        self.assertEqual(result["conteo_conjunto"], 1)
        pd.testing.assert_frame_equal(dataframe, original)

    def test_rejects_undefined_conditioning_events(self):
        dataframe = pd.DataFrame({"first": ["x", "y"], "second": ["a", "b"]})

        with self.assertRaisesRegex(ValueError, r"P\(B\) = 0"):
            calculate_event_probabilities(
                dataframe, "first", ("value", "x"), "second", ("value", "missing")
            )

    def test_rejects_invalid_columns_and_event_types(self):
        dataframe = pd.DataFrame({"first": ["x"]})

        with self.assertRaisesRegex(ValueError, "No existe la columna"):
            calculate_event_probabilities(
                dataframe, "missing", ("value", "x"), "first", ("value", "x")
            )
        with self.assertRaisesRegex(ValueError, "tipo de evento"):
            calculate_event_probabilities(
                dataframe, "first", ("other", "x"), "first", ("value", "x")
            )


if __name__ == "__main__":
    unittest.main()
