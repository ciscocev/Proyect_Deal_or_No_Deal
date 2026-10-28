"""Clasificador Naive Bayes categórico implementado con conteos propios."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any

import pandas as pd

from src.probability.simple import simple_probability
from src.probability.smoothing import laplace_smoothing

MISSING_CATEGORY = "<MISSING>"
UNKNOWN_CATEGORY = "<UNKNOWN>"


def _is_missing(value: object) -> bool:
    """Detecta un escalar faltante y rechaza celdas que contengan arreglos."""
    missing = pd.isna(value)
    if hasattr(missing, "__len__"):
        raise ValueError("Cada celda debe contener un único valor categórico.")
    return bool(missing)


def _category_key(value: object, missing_strategy: str) -> Any:
    """Normaliza categorías y reserva etiquetas internas para faltantes/desconocidos."""
    if _is_missing(value):
        if missing_strategy == "category":
            return MISSING_CATEGORY
        raise ValueError("Se encontraron valores faltantes; elija missing_strategy='category'.")
    try:
        hash(value)
    except TypeError as error:
        raise ValueError("Las variables predictoras y objetivo deben ser categóricas y hashables.") from error
    if value == MISSING_CATEGORY or value == UNKNOWN_CATEGORY:
        raise ValueError(f"El valor {value!r} está reservado para el modelo.")
    if hasattr(value, "item"):
        return value.item()
    return value


@dataclass(frozen=True)
class CategoricalNaiveBayes:
    """Conteos y vocabularios aprendidos, sin conservar ni modificar el DataFrame."""

    target: str
    predictors: tuple[str, ...]
    classes: tuple[Any, ...]
    categories: dict[str, tuple[Any, ...]]
    class_counts: dict[Any, int]
    conditional_counts: dict[Any, dict[str, dict[Any, int]]]
    training_rows: int
    alpha: float
    missing_strategy: str
    warnings: tuple[str, ...]

    def prior_probability(self, class_value: Any) -> float:
        """Devuelve la probabilidad previa observada de una clase."""
        if class_value not in self.class_counts:
            raise KeyError(f"Clase desconocida: {class_value!r}")
        return simple_probability(self.class_counts[class_value], self.training_rows)

    def conditional_probability(
        self, predictor: str, value: object, class_value: Any
    ) -> float:
        """Devuelve P(predictor=value | clase) con suavizado de Laplace."""
        if predictor not in self.categories:
            raise KeyError(f"Predictor desconocido: {predictor}")
        if class_value not in self.class_counts:
            raise KeyError(f"Clase desconocida: {class_value!r}")
        if _is_missing(value):
            if self.missing_strategy == "category":
                category = MISSING_CATEGORY
            else:
                category = _category_key(value, self.missing_strategy)
        elif value == MISSING_CATEGORY and self.missing_strategy == "category":
            category = MISSING_CATEGORY
        else:
            category = _category_key(value, self.missing_strategy)
        known_categories = self.categories[predictor]
        if category not in known_categories:
            category = UNKNOWN_CATEGORY
        count = self.conditional_counts[class_value][predictor].get(category, 0)
        category_count = len(known_categories) + 1
        return laplace_smoothing(
            count,
            self.class_counts[class_value],
            category_count,
            self.alpha,
        )


def fit_naive_bayes(
    dataframe: pd.DataFrame,
    target: str,
    predictors: tuple[str, ...] | list[str],
    alpha: float = 1.0,
    missing_strategy: str = "error",
) -> CategoricalNaiveBayes:
    """Entrena Naive Bayes categórico con predictores seleccionados explícitamente.

    `missing_strategy="error"` rechaza faltantes. Con `"category"`, los faltantes
    predictivos se aprenden como una categoría propia; filas sin clase objetivo
    siempre se rechazan porque no aportan una etiqueta supervisada.
    """
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        raise ValueError("dataframe debe ser un DataFrame no vacío.")
    if not isinstance(target, str) or not target or target not in dataframe.columns:
        raise ValueError("target debe nombrar una columna existente.")
    if isinstance(predictors, str):
        raise ValueError("predictors debe ser una secuencia de nombres, no un string.")
    predictor_tuple = tuple(predictors)
    if not predictor_tuple:
        raise ValueError("Debe seleccionarse al menos un predictor.")
    if any(not isinstance(column, str) or not column for column in predictor_tuple):
        raise ValueError("Cada predictor debe ser el nombre de una columna.")
    if len(set(predictor_tuple)) != len(predictor_tuple):
        raise ValueError("Los predictores no deben repetirse.")
    if target in predictor_tuple:
        raise ValueError("La variable objetivo no puede usarse como predictor.")
    missing_columns = set(predictor_tuple) - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Predictores inexistentes: {sorted(missing_columns)}")
    if missing_strategy not in ("error", "category"):
        raise ValueError("missing_strategy debe ser 'error' o 'category'.")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, Real)
        or not isfinite(float(alpha))
        or alpha <= 0
    ):
        raise ValueError("alpha debe ser un número positivo.")

    selected = dataframe.loc[:, [*predictor_tuple, target]].copy(deep=True)
    if selected[target].isna().any():
        raise ValueError("El objetivo contiene faltantes; no pueden entrenarse sin etiqueta.")
    if missing_strategy == "error" and selected.loc[:, predictor_tuple].isna().any().any():
        raise ValueError("Hay faltantes en los predictores; use missing_strategy='category'.")

    normalized_target = [
        _category_key(value, "error") for value in selected[target].tolist()
    ]
    class_values = tuple(dict.fromkeys(normalized_target))
    class_counts = dict(Counter(normalized_target))
    row_categories: dict[str, list[Any]] = {}
    categories: dict[str, tuple[Any, ...]] = {}
    for predictor in predictor_tuple:
        values = [
            _category_key(value, missing_strategy)
            for value in selected[predictor].tolist()
        ]
        row_categories[predictor] = values
        categories[predictor] = tuple(dict.fromkeys(values))

    conditional_counts: dict[Any, dict[str, dict[Any, int]]] = {}
    for class_value in class_values:
        class_indices = [
            index for index, value in enumerate(normalized_target) if value == class_value
        ]
        conditional_counts[class_value] = {}
        for predictor in predictor_tuple:
            values = [row_categories[predictor][index] for index in class_indices]
            conditional_counts[class_value][predictor] = dict(Counter(values))

    warnings: list[str] = []
    if len(normalized_target) < 2:
        warnings.append("El entrenamiento tiene menos de dos observaciones.")
    rare_classes = [value for value, count in class_counts.items() if count < 2]
    if rare_classes:
        warnings.append(
            "Hay clases con menos de dos observaciones: "
            + ", ".join(repr(value) for value in rare_classes)
        )
    if len(class_values) < 2:
        warnings.append("El entrenamiento contiene una sola clase.")

    return CategoricalNaiveBayes(
        target=target,
        predictors=predictor_tuple,
        classes=class_values,
        categories=categories,
        class_counts=class_counts,
        conditional_counts=conditional_counts,
        training_rows=len(normalized_target),
        alpha=float(alpha),
        missing_strategy=missing_strategy,
        warnings=tuple(warnings),
    )