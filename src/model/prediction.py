"""Predicciones explicables de Naive Bayes categórico."""

from __future__ import annotations

from collections.abc import Mapping
from math import exp, log
from typing import Any

from src.model.naive_bayes import (
    CategoricalNaiveBayes,
    MISSING_CATEGORY,
    UNKNOWN_CATEGORY,
    _is_missing,
)


def predict_observation(
    model: CategoricalNaiveBayes,     observation: Mapping[str, object]
) -> dict[str, Any]:
    """Predice una fila y devuelve probabilidades, términos y advertencias.

    El posterior se normaliza a partir de log(P(clase)) + sum(log(P(valor|clase))),
    evitando el producto directo de muchas probabilidades pequeñas.
    """
    if not isinstance(observation, Mapping):
        raise ValueError("observation debe ser un mapeo de predictor a valor.")
    if set(observation) != set(model.predictors):
        missing = set(model.predictors) - set(observation)
        extra = set(observation) - set(model.predictors)
        raise ValueError(
            f"La observación debe contener solo los predictores entrenados; "
            f"faltantes={sorted(missing)}, adicionales={sorted(extra)}."
        )

    warnings = list(model.warnings)
    normalized_values: dict[str, object] = {}
    unknown_predictors: list[str] = []
    for predictor in model.predictors:
        value = observation[predictor]
        if _is_missing(value):
            if model.missing_strategy == "category":
                value = MISSING_CATEGORY
            else:
                raise ValueError(
                    f"El predictor {predictor!r} está faltante; "
                    "entrene con missing_strategy='category' para admitirlo."
                )
        if value in (MISSING_CATEGORY, UNKNOWN_CATEGORY):
            if value == UNKNOWN_CATEGORY:
                raise ValueError(f"El valor {UNKNOWN_CATEGORY!r} está reservado.")
            if model.missing_strategy != "category":
                raise ValueError("El valor reservado para faltantes no está habilitado.")
        try:
            hash(value)
        except TypeError as error:
            raise ValueError(f"El valor de {predictor!r} debe ser categórico y hashable.") from error
        if hasattr(value, "item"):
            value = value.item()
        normalized_values[predictor] = value
        if value not in model.categories[predictor] and value != MISSING_CATEGORY:
            unknown_predictors.append(predictor)

    if unknown_predictors:
        warnings.append(
            "Categorías desconocidas tratadas con el bucket reservado de Laplace en: "
            + ", ".join(unknown_predictors)
        )

    log_scores: dict[Any, float] = {}
    explanations: dict[Any, dict[str, Any]] = {}
    low_evidence: list[str] = []
    for class_value in model.classes:
        prior = model.prior_probability(class_value)
        log_prior = log(prior)
        total_log_score = log_prior
        feature_details: dict[str, dict[str, Any]] = {}

        for predictor in model.predictors:
            raw_value = normalized_values[predictor]
            unknown = (
                raw_value not in model.categories[predictor]
                and raw_value != MISSING_CATEGORY
            )
            value_for_count = UNKNOWN_CATEGORY if unknown else raw_value
            count = model.conditional_counts[class_value][predictor].get(
                value_for_count, 0
            )
            if count == 0:
                low_evidence.append(
                    f"{predictor}={raw_value!r} para clase {class_value!r}"
                )
            category_count = len(model.categories[predictor]) + 1
            denominator = model.class_counts[class_value] + (
                model.alpha * category_count
            )
            conditional = model.conditional_probability(
                predictor, raw_value, class_value
            )
            log_conditional = log(conditional)
            total_log_score += log_conditional
            feature_details[predictor] = {
                "value": raw_value,
                "count": count,
                "class_count": model.class_counts[class_value],
                "category_count_including_unknown": category_count,
                "denominator": denominator,
                "conditional_probability": conditional,
                "log_contribution": log_conditional,
                "unknown_category": unknown,
            }

        log_scores[class_value] = total_log_score
        explanations[class_value] = {
            "prior_probability": prior,
            "log_prior": log_prior,
            "predictors": feature_details,
            "log_score": total_log_score,
        }

    if low_evidence:
        examples = ", ".join(low_evidence[:5])
        remaining = len(low_evidence) - 5
        if remaining:
            examples += f" y {remaining} combinación(es) más"
        warnings.append(
            "Poca evidencia empírica: estas combinaciones tienen frecuencia cero "
            "y su probabilidad condicional depende del suavizado de Laplace: "
            + examples
        )

    max_log_score = max(log_scores.values())
    weights = {
        class_value: exp(score - max_log_score)
        for class_value, score in log_scores.items()
    }
    weight_sum = sum(weights.values())
    posterior = {
        class_value: weight / weight_sum for class_value, weight in weights.items()
    }
    predicted_class = max(model.classes, key=posterior.__getitem__)

    for class_value in model.classes:
        explanations[class_value]["unnormalized_weight"] = weights[class_value]
        explanations[class_value]["posterior_probability"] = posterior[class_value]

    return {
        "predicted_class": predicted_class,
        "class_probabilities": posterior,
        "prior_probabilities": {
            class_value: model.prior_probability(class_value)
            for class_value in model.classes
        },
        "class_details": explanations,
        "warnings": warnings,
    }