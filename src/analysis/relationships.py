"""Tablas de relación y comparación probabilística para variables discretas."""

from collections.abc import Mapping, Sequence
from numbers import Integral

import pandas as pd

from src.data.validation import calculate_remaining_highs
from src.probability.conditional import conditional_probability
from src.probability.simple import simple_probability


def joint_frequencies(
    dataframe: pd.DataFrame, row_variable: str, column_variable: str
) -> pd.DataFrame:
    """Devuelve frecuencias conjuntas sin eliminar ni alterar observaciones."""
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("dataframe debe ser un DataFrame de pandas.")
    if (
        not isinstance(row_variable, str)
        or not row_variable
        or not isinstance(column_variable, str)
        or not column_variable
    ):
        raise ValueError("row_variable y column_variable deben ser nombres de columnas.")
    missing_columns = {row_variable, column_variable} - set(dataframe.columns)
    if missing_columns:
        raise KeyError(f"No existen las columnas: {sorted(missing_columns)}")
    return pd.crosstab(
        dataframe[row_variable],
        dataframe[column_variable],
        dropna=False,
    )


def conditional_proportions(
    dataframe: pd.DataFrame, condition: str, outcome: str
) -> pd.DataFrame:
    """Devuelve P(outcome | condition) estimada por frecuencias observadas."""
    if condition == outcome:
        raise ValueError("condition y outcome deben ser columnas distintas.")
    counts = joint_frequencies(dataframe, condition, outcome)
    proportions = []
    for _, row in counts.iterrows():
        condition_count = int(row.sum())
        proportions.append(
            [
                conditional_probability(int(count), condition_count)
                if condition_count
                else 0.0
                for count in row
            ]
        )
    return pd.DataFrame(proportions, index=counts.index, columns=counts.columns)


def theoretical_high_probability(row: pd.Series | dict[str, object]) -> float:
    """Calcula la probabilidad exacta de premio Alto para una fila del juego."""
    if not isinstance(row, (pd.Series, Mapping)):
        raise ValueError("row debe ser una Serie o un mapeo.")
    required = ("maletines_restantes", "altos_revelados", "ultimo_premio")
    if any(column not in row for column in required):
        raise ValueError("row debe incluir las tres variables predictoras del juego.")
    raw_remaining_cases = row["maletines_restantes"]
    if isinstance(raw_remaining_cases, bool) or not isinstance(
        raw_remaining_cases, Integral
    ):
        if not (
            isinstance(raw_remaining_cases, str)
            and raw_remaining_cases.isdecimal()
        ):
            raise ValueError("maletines_restantes debe ser un número entero.")
    remaining_cases = int(raw_remaining_cases)
    if not 2 <= remaining_cases <= 5:
        raise ValueError("maletines_restantes debe estar entre 2 y 5.")
    raw_revealed_highs = row["altos_revelados"]
    if isinstance(raw_revealed_highs, bool) or not isinstance(
        raw_revealed_highs, Integral
    ):
        if not (
            isinstance(raw_revealed_highs, str) and raw_revealed_highs.isdecimal()
        ):
            raise ValueError("altos_revelados debe ser un número entero.")
    remaining_highs = calculate_remaining_highs(
        int(raw_revealed_highs), row["ultimo_premio"]
    )
    if not 0 <= remaining_highs <= remaining_cases:
        raise ValueError("La fila contiene un estado imposible para la probabilidad teórica.")
    return simple_probability(remaining_highs, remaining_cases)


def probability_comparison(
    dataframe: pd.DataFrame, conditions: Sequence[str]
) -> pd.DataFrame:
    """Compara P(Alto) observada con la media de probabilidad exacta por estrato."""
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("dataframe debe ser un DataFrame de pandas.")
    if dataframe.empty:
        raise ValueError("No se puede comparar probabilidades sin observaciones.")
    if isinstance(conditions, str) or not isinstance(conditions, Sequence):
        raise ValueError("Se requiere al menos una variable de agrupación.")
    if not conditions:
        raise ValueError("Se requiere al menos una variable de agrupación.")
    if any(
        not isinstance(column, str) or not column for column in conditions
    ):
        raise ValueError("conditions debe ser una secuencia de nombres de columnas.")
    if len(set(conditions)) != len(conditions):
        raise ValueError("Las variables de agrupación no deben repetirse.")
    missing_columns = set(conditions) - set(dataframe.columns)
    required = {
        "maletines_restantes",
        "altos_revelados",
        "ultimo_premio",
        "premio_final",
    }
    if missing_columns or not required.issubset(dataframe.columns):
        raise ValueError("Faltan columnas necesarias para la comparación probabilística.")

    working = dataframe.copy()
    working["_probabilidad_teorica_alto"] = working.apply(
        theoretical_high_probability, axis=1
    )
    grouped = working.groupby(list(conditions), dropna=False, observed=True)
    result = grouped.agg(
        partidas=("premio_final", "size"),
        altos_observados=("premio_final", lambda values: (values == "Alto").sum()),
        probabilidad_teorica_media=("_probabilidad_teorica_alto", "mean"),
    )
    result["probabilidad_empirica_alto"] = [
        simple_probability(int(high_count), int(game_count))
        for high_count, game_count in zip(
            result["altos_observados"], result["partidas"]
        )
    ]
    result["diferencia_empirica_menos_teorica"] = (
        result["probabilidad_empirica_alto"] - result["probabilidad_teorica_media"]
    )
    return result.reset_index()


def predictor_relationships(
    dataframe: pd.DataFrame, target: str = "premio_final"
) -> dict[str, dict[str, pd.DataFrame]]:
    """Calcula tablas conjuntas y condicionales para los tres predictores del juego."""
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("dataframe debe ser un DataFrame de pandas.")
    predictors = ("maletines_restantes", "altos_revelados", "ultimo_premio")
    return {
        predictor: {
            "joint": joint_frequencies(dataframe, predictor, target),
            "conditional": conditional_proportions(dataframe, predictor, target),
            "probability_comparison": probability_comparison(dataframe, [predictor]),
        }
        for predictor in predictors
    }