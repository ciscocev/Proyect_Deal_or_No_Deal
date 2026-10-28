"""Probabilidad de que dos eventos ocurran simultáneamente."""

from src.probability.simple import simple_probability


def joint_probability(joint_count: int, total_count: int) -> float:
    """Calcula P(A ∩ B) = N(A ∩ B) / N usando las frecuencias observadas."""
    return simple_probability(joint_count, total_count)