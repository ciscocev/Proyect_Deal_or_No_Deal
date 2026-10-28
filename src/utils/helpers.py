"""Utilidades pequeñas compartidas por las páginas de Streamlit."""

from collections.abc import Sequence
from typing import Any

import pandas as pd


def display_value(value: Any) -> str:
    """Convierte valores de categorías en etiquetas legibles para controles."""
    missing = pd.isna(value)
    if not hasattr(missing, "__len__") and bool(missing):
        return "(Faltante)"
    return str(value)


def dataframe_is_probably_identifier(series: pd.Series) -> bool:
    """Heurística visual no vinculante para advertir sobre columnas tipo ID."""
    name = str(series.name).strip().lower()
    if name in {"id", "uuid", "index", "indice", "identificador"} or name.endswith("_id"):
        return True
    return (
        len(series) > 0
        and series.nunique(dropna=True) == len(series)
        and ("id" in name or "uuid" in name)
    )


def validate_predictor_selection(
    columns: Sequence[str], target: str, predictors: Sequence[str]
) -> None:
    """Valida las variables indicadas explícitamente por el usuario."""
    if target not in columns:
        raise ValueError("La variable objetivo no existe en el dataset.")
    if not predictors:
        raise ValueError("Selecciona al menos un predictor.")
    if target in predictors:
        raise ValueError("La variable objetivo no puede seleccionarse como predictor.")
    if len(set(predictors)) != len(predictors):
        raise ValueError("No repitas variables predictoras.")
    if set(predictors) - set(columns):
        raise ValueError("Hay predictores que no existen en el dataset.")