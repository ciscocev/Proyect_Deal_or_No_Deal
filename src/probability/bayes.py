"""Teorema de Bayes con desglose explícito de sus componentes."""

from math import isfinite
from numbers import Real


def _validate_probability(value: float, name: str) -> float:
    """Valida una probabilidad escalar y la devuelve como float."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} debe ser una probabilidad numérica entre cero y uno.")
    probability = float(value)
    if not isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError(f"{name} debe estar entre cero y uno.")
    return probability


def bayes_components(
    likelihood: float, prior: float, evidence_probability: float
) -> dict[str, float]:
    """Devuelve los componentes de P(C|X) = P(X|C)P(C) / P(X).

    `likelihood` es P(X|C), `prior` es P(C) y `evidence_probability` es P(X).
    La evidencia debe ser positiva y coherente con un posterior no mayor que uno.
    """
    validated_likelihood = _validate_probability(likelihood, "likelihood")
    validated_prior = _validate_probability(prior, "prior")
    validated_evidence = _validate_probability(
        evidence_probability, "evidence_probability"
    )
    if validated_evidence == 0:
        raise ValueError("P(X) = 0: el posterior de Bayes no está definido.")

    numerator = validated_likelihood * validated_prior
    posterior = numerator / validated_evidence
    if posterior > 1:
        raise ValueError(
            "Los componentes son incoherentes: P(X|C)P(C) no puede superar P(X)."
        )
    return {
        "likelihood": validated_likelihood,
        "prior": validated_prior,
        "evidence_probability": validated_evidence,
        "numerator": numerator,
        "posterior": posterior,
    }


def bayes_theorem(
    likelihood: float, prior: float, evidence_probability: float
) -> float:
    """Calcula P(C | X) a partir de P(X | C), P(C) y P(X)."""
    return bayes_components(likelihood, prior, evidence_probability)["posterior"]