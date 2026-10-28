"""Probabilidad condicional basada en conteos observados."""

from numbers import Integral


def conditional_probability(
    joint_count: int, condition_count: int
) -> float:
    """Calcula P(A | B) = N(A ∩ B) / N(B).

    Si N(B) = 0, P(A | B) no está definida y se informa con ValueError.
    """
    if isinstance(joint_count, bool) or not isinstance(joint_count, Integral):
        raise ValueError("joint_count debe ser un entero no negativo.")
    if isinstance(condition_count, bool) or not isinstance(condition_count, Integral):
        raise ValueError("condition_count debe ser un entero no negativo.")
    if condition_count <= 0:
        raise ValueError("P(B) = 0: la probabilidad condicional no está definida.")
    if joint_count < 0 or joint_count > condition_count:
        raise ValueError("joint_count debe estar entre cero y condition_count.")
    return int(joint_count) / int(condition_count)