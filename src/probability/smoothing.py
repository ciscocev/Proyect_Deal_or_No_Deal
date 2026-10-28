"""Suavizado de Laplace para frecuencias de variables categóricas."""

from math import isfinite
from numbers import Integral, Real


def laplace_smoothing(
    category_count: int,
    total_count: int,
    number_of_categories: int,
    alpha: float = 1.0,
) -> float:
    """Estima la probabilidad de una categoría con suavizado aditivo.

    Calcula (category_count + alpha) / (total_count + alpha * K), donde K
    es el número de categorías posibles. Con alpha=1, una categoría no
    observada recibe probabilidad positiva en lugar de cero.
    """
    for value, name in (
        (category_count, "category_count"),
        (total_count, "total_count"),
        (number_of_categories, "number_of_categories"),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} debe ser un entero.")
    if total_count < 0:
        raise ValueError("total_count no puede ser negativo.")
    if category_count < 0 or category_count > total_count:
        raise ValueError("category_count debe estar entre cero y total_count.")
    if number_of_categories < 1:
        raise ValueError("number_of_categories debe ser al menos uno.")
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise ValueError("alpha debe ser un número positivo y finito.")
    alpha_value = float(alpha)
    if not isfinite(alpha_value) or alpha_value <= 0:
        raise ValueError("alpha debe ser un número positivo y finito.")

    return (int(category_count) + alpha_value) / (
        int(total_count) + alpha_value * int(number_of_categories)
    )