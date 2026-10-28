"""Cálculo reutilizable de probabilidades empíricas para eventos en DataFrames."""

import pandas as pd

from src.probability.bayes import bayes_components
from src.probability.conditional import conditional_probability
from src.probability.joint import joint_probability
from src.probability.simple import simple_probability


def _event_mask(
    dataframe: pd.DataFrame, column: str, event: tuple[str, object]
) -> pd.Series:
    """Devuelve la máscara booleana de un evento de valor o de dato faltante."""
    if column not in dataframe.columns:
        raise ValueError(f"No existe la columna del evento: {column}")
    if not isinstance(event, tuple) or len(event) != 2:
        raise ValueError("Cada evento debe ser una tupla (tipo, valor).")

    kind, value = event
    if kind == "missing":
        return dataframe[column].isna()
    if kind != "value":
        raise ValueError("El tipo de evento debe ser 'value' o 'missing'.")
    return dataframe[column].eq(value).fillna(False)


def calculate_event_probabilities(
    dataframe: pd.DataFrame,
    column_a: str,
    event_a: tuple[str, object],
    column_b: str,
    event_b: tuple[str, object],
) -> dict[str, float | int]:
    """Calcula conteos y probabilidades de dos eventos sin modificar los datos."""
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("dataframe debe ser un DataFrame.")

    mask_a = _event_mask(dataframe, column_a, event_a)
    mask_b = _event_mask(dataframe, column_b, event_b)
    count_a = int(mask_a.sum())
    count_b = int(mask_b.sum())
    count_joint = int((mask_a & mask_b).sum())
    total = len(dataframe)

    p_a = simple_probability(count_a, total)
    p_b = simple_probability(count_b, total)
    p_joint = joint_probability(count_joint, total)
    p_a_given_b = conditional_probability(count_joint, count_b)
    p_b_given_a = conditional_probability(count_joint, count_a)
    bayes = bayes_components(p_a_given_b, p_b, p_a)

    return {
        "P(A)": p_a,
        "P(B)": p_b,
        "P(A intersección B)": p_joint,
        "P(A dado B)": p_a_given_b,
        "P(B dado A)": p_b_given_a,
        "Bayes P(B dado A)": bayes["posterior"],
        "conteo_A": count_a,
        "conteo_B": count_b,
        "conteo_conjunto": count_joint,
        "total": total,
    }
