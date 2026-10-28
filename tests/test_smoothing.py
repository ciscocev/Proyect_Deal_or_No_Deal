import unittest

from src.probability.smoothing import laplace_smoothing


class LaplaceSmoothingTests(unittest.TestCase):
    def test_smooths_zero_count_to_positive_probability(self):
        self.assertAlmostEqual(laplace_smoothing(0, 10, 2), 1 / 12)

    def test_smoothed_categories_sum_to_one(self):
        probabilities = [
            laplace_smoothing(4, 10, 2),
            laplace_smoothing(6, 10, 2),
        ]
        self.assertAlmostEqual(sum(probabilities), 1.0)

    def test_empty_observations_produce_uniform_smoothed_probability(self):
        self.assertEqual(laplace_smoothing(0, 0, 4), 0.25)

    def test_rejects_invalid_counts_categories_and_alpha(self):
        invalid_inputs = (
            (-1, 10, 2, 1),
            (11, 10, 2, 1),
            (0, 10, 0, 1),
            (0, 10, 2, 0),
        )
        for arguments in invalid_inputs:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                laplace_smoothing(*arguments)


if __name__ == "__main__":
    unittest.main()
