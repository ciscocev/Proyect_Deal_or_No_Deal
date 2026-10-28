import unittest

from src.data.validation import calculate_remaining_highs, validate_dataset


class DataValidationTests(unittest.TestCase):
    def test_remaining_highs_from_small_manually_checkable_counts(self):
        self.assertEqual(calculate_remaining_highs(2, "Alto"), 3)
        self.assertEqual(calculate_remaining_highs(2, "Bajo"), 4)

    def test_remaining_highs_rejects_invalid_count_and_category(self):
        for revealed, latest in (
            (-1, "Bajo"),
            (7, "Bajo"),
            (True, "Bajo"),
            (0, "other"),
            (0, None),
        ):
            with self.subTest(revealed=revealed, latest=latest), self.assertRaises(ValueError):
                calculate_remaining_highs(revealed, latest)

    def test_validate_dataset_rejects_empty_data(self):
        with self.assertRaisesRegex(ValueError, "no contiene registros"):
            validate_dataset([])


if __name__ == "__main__":
    unittest.main()
