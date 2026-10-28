import unittest

import pandas as pd

from src.utils.state import (
    EVALUATION_KEY,
    MODEL_KEY,
    PREDICTION_KEY,
    initialize_state,
    set_dataset,
    set_variable_selection,
)


class SessionStateTests(unittest.TestCase):
    def setUp(self):
        self.state = {}
        initialize_state(self.state)

    def test_dataset_load_invalidates_model_evaluation_and_selection(self):
        frame = pd.DataFrame({"x": ["a", "b"], "y": ["yes", "no"]})
        set_dataset(self.state, frame, "first.csv")
        set_variable_selection(self.state, "y", ["x"])
        self.state[MODEL_KEY] = object()
        self.state[PREDICTION_KEY] = {"class": "yes"}
        self.state[EVALUATION_KEY] = {"accuracy": 1.0}

        changed = set_dataset(
            self.state,
            pd.DataFrame({"x": ["a", "c"], "y": ["yes", "no"]}),
            "second.csv",
        )

        self.assertTrue(changed)
        self.assertIsNone(self.state["selected_target"])
        self.assertEqual(self.state["selected_predictors"], [])
        self.assertNotIn(MODEL_KEY, self.state)
        self.assertNotIn(PREDICTION_KEY, self.state)
        self.assertNotIn(EVALUATION_KEY, self.state)

    def test_variable_selection_invalidates_stale_results(self):
        set_dataset(
            self.state,
            pd.DataFrame({"x": ["a"], "y": ["yes"], "z": [1]}),
            "data.csv",
        )
        set_variable_selection(self.state, "y", ["x"])
        self.state[MODEL_KEY] = object()
        self.state[EVALUATION_KEY] = {"accuracy": 0.8}

        changed = set_variable_selection(self.state, "y", ["z"])

        self.assertTrue(changed)
        self.assertNotIn(MODEL_KEY, self.state)
        self.assertNotIn(EVALUATION_KEY, self.state)

    def test_identical_dataset_does_not_clear_current_selection(self):
        frame = pd.DataFrame({"x": ["a", "b"], "y": ["yes", "no"]})
        set_dataset(self.state, frame, "one.csv")
        set_variable_selection(self.state, "y", ["x"])
        self.state[MODEL_KEY] = "model"

        changed = set_dataset(self.state, frame.copy(), "same-content.csv")

        self.assertFalse(changed)
        self.assertEqual(self.state["selected_target"], "y")
        self.assertEqual(self.state[MODEL_KEY], "model")


if __name__ == "__main__":
    unittest.main()
