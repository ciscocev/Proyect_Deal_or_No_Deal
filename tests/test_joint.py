import unittest

from src.probability.joint import joint_probability


class JointProbabilityTests(unittest.TestCase):
    def test_joint_probability_from_intersection_count(self):
        self.assertEqual(joint_probability(2, 8), 0.25)

    def test_no_joint_occurrences(self):
        self.assertEqual(joint_probability(0, 8), 0.0)

    def test_rejects_intersection_larger_than_sample(self):
        with self.assertRaises(ValueError):
            joint_probability(5, 4)


if __name__ == "__main__":
    unittest.main()