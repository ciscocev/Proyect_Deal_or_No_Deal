"""Estado de sesión de la aplicación, sin dependencia directa de Streamlit."""

from collections.abc import MutableMapping, Sequence
from hashlib import sha256

import pandas as pd
from pandas.util import hash_pandas_object

DATASET_KEY = "dataset"
DATASET_NAME_KEY = "dataset_name"
DATASET_FINGERPRINT_KEY = "dataset_fingerprint"
TARGET_KEY = "selected_target"
PREDICTORS_KEY = "selected_predictors"
MODEL_KEY = "trained_model"
MODEL_SIGNATURE_KEY = "model_signature"
MODEL_TRAIN_ROWS_KEY = "model_train_rows"
MODEL_TEST_ROWS_KEY = "model_test_rows"
PREDICTION_KEY = "prediction_result"
EVALUATION_KEY = "evaluation_result"
PROBABILITY_RESULT_KEY = "probability_result"

_MODEL_STATE_KEYS = (
    MODEL_KEY,
    MODEL_SIGNATURE_KEY,
    MODEL_TRAIN_ROWS_KEY,
    MODEL_TEST_ROWS_KEY,
    PREDICTION_KEY,
    "model_split_info",
)
_EVALUATION_STATE_KEYS = (EVALUATION_KEY,)


def dataframe_fingerprint(dataframe: pd.DataFrame) -> str:
    """Calcula una huella estable para invalidar resultados obsoletos."""
    digest = sha256()
    digest.update(repr(tuple(dataframe.columns)).encode("utf-8"))
    digest.update(repr(tuple(map(str, dataframe.dtypes))).encode("utf-8"))
    digest.update(hash_pandas_object(dataframe, index=True, categorize=True).values.tobytes())
    return digest.hexdigest()


def initialize_state(state: MutableMapping[str, object]) -> None:
    """Inicializa las claves persistentes de la sesión sin sobrescribir valores."""
    state.setdefault(DATASET_KEY, None)
    state.setdefault(DATASET_NAME_KEY, None)
    state.setdefault(DATASET_FINGERPRINT_KEY, None)
    state.setdefault(TARGET_KEY, None)
    state.setdefault(PREDICTORS_KEY, [])
    state.setdefault(PROBABILITY_RESULT_KEY, None)


def _clear_keys(state: MutableMapping[str, object], keys: Sequence[str]) -> None:
    """Elimina las claves de sesión indicadas para invalidar resultados obsoletos."""
    for key in keys:
        state.pop(key, None)


def set_dataset(
    state: MutableMapping[str, object], dataframe: pd.DataFrame, name: str
) -> bool:
    """Guarda un dataset y limpia dependencias solo si cambió su contenido."""
    fingerprint = dataframe_fingerprint(dataframe)
    changed = fingerprint != state.get(DATASET_FINGERPRINT_KEY)
    state[DATASET_KEY] = dataframe.copy(deep=True)
    state[DATASET_NAME_KEY] = name
    if changed:
        state[DATASET_FINGERPRINT_KEY] = fingerprint
        state[TARGET_KEY] = None
        state[PREDICTORS_KEY] = []
        _clear_keys(
            state,
            (*_MODEL_STATE_KEYS, *_EVALUATION_STATE_KEYS, PROBABILITY_RESULT_KEY),
        )
    return changed


def set_variable_selection(
    state: MutableMapping[str, object], target: str, predictors: Sequence[str]
) -> bool:
    """Guarda la selección explícita e invalida el modelo/evaluación obsoletos."""
    normalized_predictors = list(predictors)
    signature = (
        state.get(DATASET_FINGERPRINT_KEY),
        target,
        tuple(normalized_predictors),
    )
    changed = (
        state.get(TARGET_KEY) != target
        or state.get(PREDICTORS_KEY) != normalized_predictors
    )
    state[TARGET_KEY] = target
    state[PREDICTORS_KEY] = normalized_predictors
    if changed:
        _clear_keys(
            state,
            (*_MODEL_STATE_KEYS, *_EVALUATION_STATE_KEYS, PROBABILITY_RESULT_KEY),
        )
    state["current_selection_signature"] = signature
    return changed


def clear_model_results(state: MutableMapping[str, object]) -> None:
    """Limpia modelo, predicción y evaluación conservando dataset y selección."""
    _clear_keys(state, (*_MODEL_STATE_KEYS, *_EVALUATION_STATE_KEYS))