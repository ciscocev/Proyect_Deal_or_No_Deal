import unittest

from src.probability.simple import simple_probability


class SimpleProbabilityTests(unittest.TestCase):
    def test_probability_from_observed_counts(self):
        self.assertEqual(simple_probability(3, 10), 0.3)

    def test_zero_favorable_observations(self):
        self.assertEqual(simple_probability(0, 10), 0.0)

    def test_rejects_empty_or_invalid_counts(self):
        for counts in ((0, 0), (-1, 10), (11, 10), (1.5, 10), (True, 10)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                simple_probability(*counts)


if __name__ == "__main__":
    unittest.main()