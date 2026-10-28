import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.frequencies import class_distribution, frequency_table
from src.analysis.profiling import (
    describe_column,
    is_continuous_numeric,
    profile_dataset,
)
from src.analysis.relationships import (
    conditional_proportions,
    joint_frequencies,
    probability_comparison,
    theoretical_high_probability,
)
from src.data.loader import load_csv, validate_game_dataset
from src.visualization.plots import (
    distribution_plot,
    game_eda_figures,
    probability_comparison_plot,
    target_relationship_plot,
)


class EdaTests(unittest.TestCase):
    def setUp(self):
        self.dataframe = pd.DataFrame(
            {
                "maletines_restantes": [5, 4, 3, 2],
                "altos_revelados": [1, 2, 3, 4],
                "ultimo_premio": ["Bajo", "Alto", "Bajo", "Alto"],
                "premio_final": ["Alto", "Bajo", "Alto", "Bajo"],
            }
        )

    def test_load_csv_and_expected_schema(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample.csv"
            self.dataframe.to_csv(path, index=False)
            loaded = load_csv(path, expected_columns=self.dataframe.columns)
            self.assertEqual(loaded.shape, (4, 4))
            with self.assertRaisesRegex(ValueError, "columnas"):
                load_csv(path, expected_columns=("a", "b"))

    def test_validate_game_dataset_accepts_valid_rows_and_rejects_extra_column(self):
        validate_game_dataset(self.dataframe)
        with self.assertRaisesRegex(ValueError, "exactamente"):
            validate_game_dataset(self.dataframe.assign(extra=1))

    def test_profile_detects_missing_duplicates_constants_and_low_information(self):
        dataframe = self.dataframe.copy()
        dataframe["constant"] = ["x", "x", "x", "x"]
        dataframe["mostly_same"] = ["a", "a", "a", "b"]
        dataframe.loc[0, "mostly_same"] = None
        dataframe = pd.concat([dataframe, dataframe.iloc[[1]]], ignore_index=True)

        profile = profile_dataset(dataframe, low_information_threshold=0.75)

        self.assertEqual(profile["rows"], 5)
        self.assertEqual(profile["columns"], 6)
        self.assertEqual(profile["duplicate_rows"], 1)
        self.assertEqual(profile["missing_total"], 1)
        self.assertIn("constant", profile["constant_columns"])
        self.assertIn("mostly_same", profile["low_information_columns"])

    def test_profile_reports_missing_proportions_and_insufficient_columns(self):
        dataframe = pd.DataFrame(
            {
                "enough": [1, 2, 3, 4],
                "sparse": [1, None, None, None],
                "empty": [None, None, None, None],
            }
        )

        profile = profile_dataset(dataframe, minimum_observations=2)

        self.assertEqual(profile["missing"]["sparse"], 3)
        self.assertEqual(profile["missing_proportion"]["sparse"], 0.75)
        self.assertEqual(profile["observed_counts"]["sparse"], 1)
        self.assertEqual(profile["missing_proportion"]["empty"], 1.0)
        self.assertEqual(profile["insufficient_data_columns"], ["sparse", "empty"])

    def test_profile_rejects_invalid_dataframe_and_thresholds(self):
        for value in (0, 1.1, float("nan"), True, "0.5"):
            with self.subTest(threshold=value), self.assertRaises(ValueError):
                profile_dataset(self.dataframe, low_information_threshold=value)
        with self.assertRaisesRegex(ValueError, "DataFrame"):
            profile_dataset([{"x": 1}])
        with self.assertRaisesRegex(ValueError, "Serie"):
            is_continuous_numeric([1, 2, 3])

    def test_describe_column_and_frequency_table(self):
        result = describe_column(self.dataframe, "maletines_restantes")
        self.assertEqual(result["distinct_values"], 4)
        self.assertIn("statistics", result)
        self.assertEqual(result["missing_proportion"], 0.0)
        self.assertEqual(result["distribution_type"], "bar")
        frequencies = frequency_table(self.dataframe["premio_final"])
        self.assertEqual(int(frequencies["frecuencia"].sum()), 4)
        self.assertAlmostEqual(float(frequencies["porcentaje"].sum()), 100)

    def test_frequency_table_normalizes_over_non_missing_values_when_excluding_missing(self):
        series = pd.Series(["a", "a", "b", None])

        frequencies = frequency_table(series, include_missing=False)

        self.assertEqual(int(frequencies["frecuencia"].sum()), 3)
        self.assertAlmostEqual(float(frequencies["frecuencia_relativa"].sum()), 1.0)
        self.assertAlmostEqual(float(frequencies["porcentaje"].sum()), 100.0)
        self.assertEqual(len(frequency_table(pd.Series(dtype="object"))), 0)

    def test_describe_column_reports_missing_proportion_and_numeric_statistics(self):
        dataframe = pd.DataFrame({"measurement": [1.0, 3.0, None, 5.0]})

        description = describe_column(dataframe, "measurement")

        self.assertEqual(description["missing"], 1)
        self.assertEqual(description["missing_proportion"], 0.25)
        self.assertEqual(description["observed"], 3)
        self.assertEqual(description["distribution_type"], "bar")
        self.assertEqual(description["statistics"]["mean"], 3.0)

    def test_class_distribution_reports_majority_baseline(self):
        distribution = class_distribution(
            pd.Series(["Alto", "Bajo", "Bajo", "Bajo"])
        )
        self.assertEqual(distribution["majority_count"], 3)
        self.assertEqual(distribution["majority_baseline"], 0.75)

    def test_joint_and_conditional_frequencies(self):
        joint = joint_frequencies(self.dataframe, "ultimo_premio", "premio_final")
        conditional = conditional_proportions(
            self.dataframe, "ultimo_premio", "premio_final"
        )
        self.assertEqual(int(joint.to_numpy().sum()), 4)
        self.assertTrue(all(abs(value - 1) < 1e-12 for value in conditional.sum(axis=1)))

    def test_theoretical_probability_uses_latest_reveal(self):
        high_latest = self.dataframe.iloc[1]
        low_latest = self.dataframe.iloc[0]
        self.assertEqual(theoretical_high_probability(high_latest), 3 / 4)
        self.assertEqual(theoretical_high_probability(low_latest), 5 / 5)

    def test_theoretical_probability_rejects_impossible_or_malformed_counts(self):
        invalid_rows = (
            {"maletines_restantes": 1, "altos_revelados": 0, "ultimo_premio": "Bajo"},
            {"maletines_restantes": 5, "altos_revelados": 7, "ultimo_premio": "Bajo"},
            {"maletines_restantes": 5, "altos_revelados": 1.5, "ultimo_premio": "Bajo"},
        )
        for row in invalid_rows:
            with self.subTest(row=row), self.assertRaises(ValueError):
                theoretical_high_probability(row)
        with self.assertRaisesRegex(ValueError, "variable de agrupación"):
            probability_comparison(self.dataframe, "ultimo_premio")
        with self.assertRaisesRegex(ValueError, "no deben repetirse"):
            probability_comparison(
                self.dataframe, ["maletines_restantes", "maletines_restantes"]
            )

    def test_probability_comparison_reports_empirical_and_theoretical_values(self):
        result = probability_comparison(self.dataframe, ["maletines_restantes"])
        self.assertIn("probabilidad_empirica_alto", result.columns)
        self.assertIn("probabilidad_teorica_media", result.columns)
        self.assertEqual(len(result), 4)

    def test_probability_comparison_matches_manually_calculated_small_dataset(self):
        dataframe = pd.DataFrame(
            {
                "maletines_restantes": [5, 5, 5, 5],
                "altos_revelados": [2, 2, 2, 2],
                "ultimo_premio": ["Bajo"] * 4,
                "premio_final": ["Alto", "Alto", "Bajo", "Bajo"],
            }
        )

        result = probability_comparison(dataframe, ["maletines_restantes"])

        self.assertEqual(result.loc[0, "partidas"], 4)
        self.assertEqual(result.loc[0, "altos_observados"], 2)
        self.assertAlmostEqual(result.loc[0, "probabilidad_teorica_media"], 4 / 5)
        self.assertAlmostEqual(result.loc[0, "probabilidad_empirica_alto"], 2 / 4)
        self.assertAlmostEqual(
            result.loc[0, "diferencia_empirica_menos_teorica"], -0.3
        )

    def test_plotly_figures_are_generated(self):
        self.assertGreater(len(distribution_plot(self.dataframe, "premio_final").data), 0)
        self.assertGreater(
            len(target_relationship_plot(self.dataframe, "ultimo_premio").data), 0
        )
        self.assertGreater(len(probability_comparison_plot(self.dataframe).data), 0)
        self.assertEqual(len(game_eda_figures(self.dataframe)), 8)

    def test_distribution_plot_uses_histogram_for_continuous_numeric_data(self):
        dataframe = pd.DataFrame({"measurement": [float(value) for value in range(25)]})

        figure = distribution_plot(dataframe, "measurement")

        self.assertEqual(figure.data[0].type, "histogram")

    def test_distribution_plot_uses_bars_for_categorical_and_discrete_numeric_data(self):
        categorical = pd.DataFrame({"category": ["a", "b", "a"]})
        discrete_numeric = pd.DataFrame({"rating": [1, 2, 1, 3]})

        self.assertEqual(distribution_plot(categorical, "category").data[0].type, "bar")
        self.assertEqual(
            distribution_plot(discrete_numeric, "rating").data[0].type,
            "bar",
        )

    def test_relationship_plot_adapts_to_continuous_numeric_columns(self):
        dataframe = pd.DataFrame(
            {
                "measurement": [float(value) for value in range(25)],
                "category": ["a", "b"] * 12 + ["a"],
                "outcome": [float(value * 2) for value in range(25)],
            }
        )

        self.assertEqual(
            target_relationship_plot(dataframe, "measurement", "category").data[0].type,
            "box",
        )
        self.assertEqual(
            target_relationship_plot(dataframe, "measurement", "outcome").data[0].type,
            "scatter",
        )


if __name__ == "__main__":
    unittest.main()
