"""Resumen descriptivo de DataFrames para análisis exploratorio."""

from collections.abc import Mapping
from math import isfinite
from numbers import Real
from typing import Any

import pandas as pd


def is_continuous_numeric(series: pd.Series, distinct_threshold: int = 20) -> bool:
    """Classifica una Serie numérica como continua según su cardinalidad."""
    if not isinstance(series, pd.Series):
        raise ValueError("series debe ser una Serie de pandas.")
    if (
        isinstance(distinct_threshold, bool)
        or not isinstance(distinct_threshold, int)
        or distinct_threshold < 1
    ):
        raise ValueError("distinct_threshold debe ser un entero positivo.")
    return (
        pd.api.types.is_numeric_dtype(series)
        and int(series.nunique(dropna=True)) > distinct_threshold
    )


def profile_dataset(
    dataframe: pd.DataFrame,
    low_information_threshold: float = 0.95,
    minimum_observations: int = 10,
) -> dict[str, Any]:
    """Resume tamaño, tipos, faltantes, duplicados y variables poco informativas."""
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("dataframe debe ser un DataFrame de pandas.")
    if (
        isinstance(low_information_threshold, bool)
        or not isinstance(low_information_threshold, Real)
        or not isfinite(float(low_information_threshold))
        or not 0 < low_information_threshold <= 1
    ):
        raise ValueError("low_information_threshold debe estar entre 0 y 1.")
    if (
        isinstance(minimum_observations, bool)
        or not isinstance(minimum_observations, int)
        or minimum_observations < 1
    ):
        raise ValueError("minimum_observations debe ser un entero positivo.")

    rows, columns = dataframe.shape
    distinct_values = dataframe.nunique(dropna=True)
    missing = dataframe.isna().sum()
    observed_counts = rows - missing
    constant_columns = [
        column for column in dataframe.columns if distinct_values[column] <= 1
    ]
    low_information_columns: list[str] = []
    for column in dataframe.columns:
        non_missing = dataframe[column].dropna()
        if non_missing.empty:
            continue
        dominant_share = non_missing.value_counts(normalize=True).iloc[0]
        if dominant_share >= low_information_threshold:
            low_information_columns.append(column)

    return {
        "rows": rows,
        "columns": columns,
        "column_names": list(dataframe.columns),
        "head": dataframe.head(),
        "dtypes": dataframe.dtypes.astype(str).to_dict(),
        "missing": missing.to_dict(),
        "missing_proportion": (
            (missing / rows).to_dict() if rows else {column: 0.0 for column in dataframe}
        ),
        "observed_counts": observed_counts.to_dict(),
        "insufficient_data_threshold": minimum_observations,
        "insufficient_data_columns": [
            column
            for column in dataframe.columns
            if observed_counts[column] < minimum_observations
        ],
        "missing_total": int(missing.sum()),
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "distinct_values": distinct_values.to_dict(),
        "constant_columns": constant_columns,
        "low_information_columns": low_information_columns,
        "describe": dataframe.describe(include="all").transpose(),
    }


def describe_column(dataframe: pd.DataFrame, column: str) -> Mapping[str, Any]:
    """Devuelve frecuencias y estadísticas descriptivas de una variable."""
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("dataframe debe ser un DataFrame de pandas.")
    if not isinstance(column, str) or not column:
        raise ValueError("column debe ser el nombre de una columna.")
    if column not in dataframe.columns:
        raise KeyError(f"No existe la columna: {column}")

    values = dataframe[column]
    counts = values.value_counts(dropna=False, sort=False)
    frequencies = counts / len(dataframe) if len(dataframe) else counts.astype(float)
    missing = int(values.isna().sum())
    distinct_values = int(values.nunique(dropna=True))
    is_numeric = pd.api.types.is_numeric_dtype(values)
    distribution_type = "histogram" if is_continuous_numeric(values) else "bar"
    result: dict[str, Any] = {
        "column": column,
        "dtype": str(values.dtype),
        "missing": missing,
        "missing_proportion": missing / len(values) if len(values) else 0.0,
        "observed": len(values) - missing,
        "distinct_values": distinct_values,
        "is_numeric": is_numeric,
        "distribution_type": distribution_type,
        "absolute_frequency": counts,
        "relative_frequency": frequencies,
        "percentage": frequencies * 100,
    }
    if pd.api.types.is_numeric_dtype(values):
        result["statistics"] = values.describe()
    return result