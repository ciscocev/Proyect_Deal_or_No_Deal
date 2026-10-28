"""Funciones reutilizables para frecuencias y distribución de clases."""

import pandas as pd

from src.probability.simple import simple_probability


def frequency_table(series: pd.Series, include_missing: bool = True) -> pd.DataFrame:
    """Construye frecuencias absolutas, relativas y porcentajes de una variable."""
    if not isinstance(series, pd.Series):
        raise ValueError("series debe ser una Serie de pandas.")
    if not isinstance(include_missing, bool):
        raise ValueError("include_missing debe ser booleano.")
    counts = series.value_counts(dropna=not include_missing, sort=False)
    total = len(series) if include_missing else int(counts.sum())
    proportions = (
        counts.map(lambda count: simple_probability(int(count), total))
        if total
        else counts.astype(float)
    )
    return pd.DataFrame(
        {
            "frecuencia": counts,
            "frecuencia_relativa": proportions,
            "porcentaje": proportions * 100,
        }
    )


def class_distribution(series: pd.Series) -> dict[str, float | int]:
    """Calcula la frecuencia por clase y la referencia de clase mayoritaria."""
    table = frequency_table(series)
    total = len(series)
    majority_count = int(table["frecuencia"].max()) if not table.empty else 0
    return {
        "counts": table["frecuencia"].to_dict(),
        "proportions": table["frecuencia_relativa"].to_dict(),
        "total": total,
        "majority_count": majority_count,
        "majority_baseline": (
            simple_probability(majority_count, total) if total else 0.0
        ),
    }