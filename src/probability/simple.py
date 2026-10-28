"""Probabilidades simples calculadas a partir de frecuencias observadas."""

from numbers import Integral


def simple_probability(event_count: int, total_count: int) -> float:
    """Calcula P(A) = N(A) / N.

    El total debe ser positivo y la frecuencia favorable debe estar entre cero
    y el total. Un conjunto de observaciones vacío no define una probabilidad
    empírica.
    """
    if isinstance(event_count, bool) or not isinstance(event_count, Integral):
        raise ValueError("event_count debe ser un entero no negativo.")
    if isinstance(total_count, bool) or not isinstance(total_count, Integral):
        raise ValueError("total_count debe ser un entero positivo.")
    if total_count <= 0:
        raise ValueError("total_count debe ser mayor que cero.")
    if event_count < 0 or event_count > total_count:
        raise ValueError("event_count debe estar entre cero y total_count.")
    return int(event_count) / int(total_count)