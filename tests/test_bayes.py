import unittest

from src.probability.bayes import bayes_components, bayes_theorem


class BayesTests(unittest.TestCase):
    def test_bayes_theorem_with_manually_checkable_example(self):
        # P(positivo|enfermedad)=0.9, P(enfermedad)=0.01, P(positivo)=0.108.
        posterior = bayes_theorem(0.9, 0.01, 0.108)
        self.assertAlmostEqual(posterior, 1 / 12)

    def test_components_explain_calculation(self):
        result = bayes_components(0.9, 0.01, 0.108)
        self.assertAlmostEqual(result["numerator"], 0.009)
        self.assertAlmostEqual(result["posterior"], 0.009 / 0.108)
        self.assertEqual(result["likelihood"], 0.9)
        self.assertEqual(result["prior"], 0.01)
        self.assertEqual(result["evidence_probability"], 0.108)

    def test_zero_evidence_is_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "P\\(X\\) = 0"):
            bayes_theorem(0.5, 0.5, 0)

    def test_rejects_incoherent_components_and_invalid_probabilities(self):
        with self.assertRaisesRegex(ValueError, "incoherentes"):
            bayes_theorem(0.9, 0.8, 0.5)
        with self.assertRaises(ValueError):
            bayes_theorem(1.1, 0.5, 0.8)


if __name__ == "__main__":
    unittest.main()