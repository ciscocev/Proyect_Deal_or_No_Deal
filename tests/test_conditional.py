import unittest

from src.probability.conditional import conditional_probability


class ConditionalProbabilityTests(unittest.TestCase):
    def test_conditional_probability(self):
        self.assertEqual(conditional_probability(2, 4), 0.5)

    def test_event_impossible_under_condition(self):
        self.assertEqual(conditional_probability(0, 4), 0.0)

    def test_zero_condition_probability_is_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "P\\(B\\) = 0"):
            conditional_probability(0, 0)

    def test_rejects_intersection_larger_than_condition(self):
        with self.assertRaises(ValueError):
            conditional_probability(5, 4)


if __name__ == "__main__":
    unittest.main()