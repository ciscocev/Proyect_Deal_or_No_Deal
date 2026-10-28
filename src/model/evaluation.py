"""Separación reproducible, métricas y evaluación fuera de muestra."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Hashable, Sequence

import pandas as pd

from src.model.naive_bayes import CategoricalNaiveBayes, fit_naive_bayes
from src.model.prediction import predict_observation


@dataclass(frozen=True)
class DataSplit:
    """Partición de filas, acompañada de sus posiciones en el DataFrame original."""

    train: pd.DataFrame
    test: pd.DataFrame
    train_positions: tuple[int, ...]
    test_positions: tuple[int, ...]
    strategy: str
    warnings: tuple[str, ...] = ()


def _validate_split_inputs(
    dataframe: pd.DataFrame, target: str, test_size: float
) -> None:
    """Valida las entradas compartidas por las estrategias de partición."""
    if not isinstance(dataframe, pd.DataFrame) or len(dataframe) < 2:
        raise ValueError("Se necesitan al menos dos filas en un DataFrame.")
    if target not in dataframe.columns:
        raise ValueError(f"No existe la variable objetivo: {target}")
    if dataframe[target].isna().any():
        raise ValueError("La variable objetivo contiene faltantes.")
    if isinstance(test_size, bool) or not isinstance(test_size, (int, float)):
        raise ValueError("test_size debe ser una proporción entre cero y uno.")
    if not 0 < test_size < 1:
        raise ValueError("test_size debe ser mayor que cero y menor que uno.")


def _make_split(
    dataframe: pd.DataFrame,
    train_positions: Sequence[int],
    test_positions: Sequence[int],
    strategy: str,
    warnings: Sequence[str] = (),
) -> DataSplit:
    """Construye particiones independientes y verifica que no compartan filas."""
    train_ids = tuple(sorted(train_positions))
    test_ids = tuple(sorted(test_positions))
    if not train_ids or not test_ids:
        raise ValueError("La división debe dejar filas tanto en entrenamiento como en prueba.")
    if set(train_ids).intersection(test_ids):
        raise RuntimeError("La división contiene posiciones compartidas entre particiones.")
    return DataSplit(
        train=dataframe.iloc[list(train_ids)].copy(deep=True),
        test=dataframe.iloc[list(test_ids)].copy(deep=True),
        train_positions=train_ids,
        test_positions=test_ids,
        strategy=strategy,
        warnings=tuple(warnings),
    )


def split_train_test(
    dataframe: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    seed: int = 42,
    stratified: bool = True,
) -> DataSplit:
    """Divide filas reproduciblemente; estratifica por clase cuando es posible.

    Los identificadores devueltos son posiciones, no etiquetas del índice,
    por lo que también son inequívocos si el DataFrame tiene índices repetidos.
    Las observaciones duplicadas no se eliminan.
    """
    _validate_split_inputs(dataframe, target, test_size)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed debe ser un entero.")
    if not isinstance(stratified, bool):
        raise ValueError("stratified debe ser booleano.")

    rng = random.Random(seed)
    warnings: list[str] = []
    train_positions: list[int] = []
    test_positions: list[int] = []

    if stratified:
        labels = dataframe[target].tolist()
        class_positions: dict[Hashable, list[int]] = {}
        for position, label in enumerate(labels):
            try:
                class_positions.setdefault(label, []).append(position)
            except TypeError as error:
                raise ValueError("Las etiquetas objetivo deben ser categóricas y hashables.") from error

        for label, positions in class_positions.items():
            rng.shuffle(positions)
            if len(positions) == 1:
                train_positions.extend(positions)
                warnings.append(
                    f"La clase {label!r} tiene una sola fila y quedó solo en entrenamiento."
                )
                continue
            number_for_test = round(len(positions) * test_size)
            number_for_test = min(max(number_for_test, 1), len(positions) - 1)
            test_positions.extend(positions[:number_for_test])
            train_positions.extend(positions[number_for_test:])
    else:
        positions = list(range(len(dataframe)))
        rng.shuffle(positions)
        number_for_test = min(
            max(round(len(dataframe) * test_size), 1), len(dataframe) - 1
        )
        test_positions = positions[:number_for_test]
        train_positions = positions[number_for_test:]

    return _make_split(
        dataframe,
        train_positions,
        test_positions,
        "stratified_random" if stratified else "random",
        warnings,
    )


def split_by_predictor_groups(
    dataframe: pd.DataFrame,
    target: str,
    predictors: Sequence[str],
    test_size: float = 0.2,
    seed: int = 42,
) -> DataSplit:
    """Separa grupos completos de combinaciones predictoras para prueba.

    Responde a la pregunta de generalización a combinaciones no vistas; la
    distribución de clases y el tamaño exacto de prueba pueden diferir.
    """
    _validate_split_inputs(dataframe, target, test_size)
    if not predictors or target in predictors:
        raise ValueError("Indique predictores válidos, distintos del objetivo.")
    if len(set(predictors)) != len(predictors):
        raise ValueError("Los predictores de agrupación no deben repetirse.")
    missing = set(predictors) - set(dataframe.columns)
    if missing:
        raise ValueError(f"Predictores inexistentes: {sorted(missing)}")
    if dataframe.loc[:, predictors].isna().any().any():
        raise ValueError("La separación agrupada requiere predictores sin faltantes.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed debe ser un entero.")

    groups: dict[tuple[Hashable, ...], list[int]] = {}
    for position, values in enumerate(dataframe.loc[:, predictors].itertuples(index=False, name=None)):
        key = tuple(value.item() if hasattr(value, "item") else value for value in values)
        try:
            groups.setdefault(key, []).append(position)
        except TypeError as error:
            raise ValueError("Los valores de agrupación deben ser hashables.") from error
    if len(groups) < 2:
        raise ValueError("Se requieren al menos dos combinaciones predictoras distintas.")

    group_keys = list(groups)
    random.Random(seed).shuffle(group_keys)
    target_rows = max(round(len(dataframe) * test_size), 1)
    test_keys: list[tuple[Hashable, ...]] = []
    test_rows = 0
    for key in group_keys:
        if test_rows >= target_rows or len(test_keys) == len(group_keys) - 1:
            break
        test_keys.append(key)
        test_rows += len(groups[key])
    test_key_set = set(test_keys)
    test_positions = [
        position
        for key in test_key_set
        for position in groups[key]
    ]
    train_positions = [
        position
        for key, positions in groups.items()
        if key not in test_key_set
        for position in positions
    ]
    warning = (
        f"Separación agrupada: {len(test_keys)} combinaciones predictoras "
        f"retenidas; prueba de {len(test_positions)} filas."
    )
    return _make_split(
        dataframe, train_positions, test_positions, "predictor_groups", (warning,)
    )


def classification_metrics(
    actual: Sequence[Hashable],
    predicted: Sequence[Hashable],
    labels: Sequence[Hashable] | None = None,
) -> dict[str, Any]:
    """Calcula matriz de confusión y métricas one-vs-rest sin librerías ML.

    La matriz usa filas como clases reales y columnas como predichas. Cuando
    precisión o sensibilidad no estén definidas por denominador cero, reporta
    0.0 y expone el indicador correspondiente en `undefined`.
    """
    actual_values = list(actual)
    predicted_values = list(predicted)
    if len(actual_values) != len(predicted_values):
        raise ValueError("actual y predicted deben tener la misma longitud.")
    if not actual_values:
        raise ValueError("No se pueden calcular métricas sin observaciones.")
    for value in actual_values + predicted_values:
        try:
            hash(value)
        except TypeError as error:
            raise ValueError("Las etiquetas deben ser hashables.") from error

    label_values = list(dict.fromkeys(() if labels is None else labels))
    for value in actual_values + predicted_values:
        if value not in label_values:
            label_values.append(value)
    confusion = {
        actual_label: {
            predicted_label: sum(
                truth == actual_label and guess == predicted_label
                for truth, guess in zip(actual_values, predicted_values)
            )
            for predicted_label in label_values
        }
        for actual_label in label_values
    }
    total = len(actual_values)
    correct = sum(truth == guess for truth, guess in zip(actual_values, predicted_values))
    per_class: dict[Hashable, dict[str, float | int]] = {}
    undefined: dict[Hashable, dict[str, bool]] = {}
    for label in label_values:
        true_positive = confusion[label][label]
        false_positive = sum(
            confusion[other][label] for other in label_values if other != label
        )
        false_negative = sum(
            confusion[label][other] for other in label_values if other != label
        )
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = (
            true_positive / precision_denominator if precision_denominator else 0.0
        )
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[label] = {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        undefined[label] = {
            "precision": precision_denominator == 0,
            "recall": recall_denominator == 0,
        }

    return {
        "labels": label_values,
        "confusion_matrix": confusion,
        "accuracy": correct / total,
        "correct": correct,
        "incorrect": total - correct,
        "total": total,
        "per_class": per_class,
        "undefined": undefined,
    }


def majority_class(training_target: Sequence[Hashable]) -> Hashable:
    """Selecciona la clase mayoritaria exclusivamente de etiquetas de entrenamiento."""
    if len(training_target) == 0:
        raise ValueError("El entrenamiento no contiene etiquetas.")
    counts = Counter(training_target)
    return max(counts, key=counts.get)


def _feature_overlap(
    train: pd.DataFrame, test: pd.DataFrame, predictors: Sequence[str]
) -> int:
    """Cuenta combinaciones predictoras que aparecen en ambas particiones."""
    train_combinations = set(
        train.loc[:, predictors].itertuples(index=False, name=None)
    )
    test_combinations = set(
        test.loc[:, predictors].itertuples(index=False, name=None)
    )
    return len(train_combinations.intersection(test_combinations))


def analyze_errors(
    test: pd.DataFrame,
    predictions: Sequence[Hashable],
    predictors: Sequence[str],
    target: str,
    positive_class: Hashable,
) -> dict[str, Any]:
    """Identifica aciertos y errores tipo FP/FN respecto a una clase positiva."""
    if len(test) != len(predictions):
        raise ValueError("Debe existir una predicción por fila de prueba.")
    details = test.loc[:, [*predictors, target]].copy()
    details["prediccion"] = list(predictions)
    details["correcto"] = details[target] == details["prediccion"]
    false_positives = details[
        (details[target] != positive_class)
        & (details["prediccion"] == positive_class)
    ]
    false_negatives = details[
        (details[target] == positive_class)
        & (details["prediccion"] != positive_class)
    ]
    return {
        "correct": details[details["correcto"]].copy(),
        "incorrect": details[~details["correcto"]].copy(),
        "false_positives": false_positives.copy(),
        "false_negatives": false_negatives.copy(),
        "error_combinations": (
            details.loc[~details["correcto"], [*predictors, target, "prediccion"]]
            .value_counts(dropna=False)
            .rename("frecuencia")
            .reset_index()
        ),
    }


def evaluate_naive_bayes(
    dataframe: pd.DataFrame,
    target: str,
    predictors: Sequence[str],
    test_size: float = 0.2,
    seed: int = 42,
    alpha: float = 1.0,
    stratified: bool = True,
    positive_class: Hashable | None = None,
    grouped: bool = False,
    missing_strategy: str = "error",
) -> dict[str, Any]:
    """Ajusta solo en entrenamiento y compara Naive Bayes con mayoría."""
    if not predictors:
        raise ValueError("Debe indicarse al menos un predictor.")
    split = (
        split_by_predictor_groups(dataframe, target, predictors, test_size, seed)
        if grouped
        else split_train_test(dataframe, target, test_size, seed, stratified)
    )
    model: CategoricalNaiveBayes = fit_naive_bayes(
        split.train,
        target,
        list(predictors),
        alpha=alpha,
        missing_strategy=missing_strategy,
    )
    prediction_details = [
        predict_observation(
            model,
            {predictor: row[predictor] for predictor in predictors},
        )
        for _, row in split.test.iterrows()
    ]
    predictions = [result["predicted_class"] for result in prediction_details]
    actual = split.test[target].tolist()
    baseline = majority_class(split.train[target].tolist())
    baseline_predictions = [baseline] * len(actual)
    labels = list(model.classes)
    positive = positive_class
    if positive is None and "Alto" in labels:
        positive = "Alto"
    metrics = classification_metrics(actual, predictions, labels=labels)
    baseline_metrics = classification_metrics(
        actual, baseline_predictions, labels=labels
    )
    error_details = analyze_errors(
        split.test, predictions, predictors, target, positive
    ) if positive in labels else analyze_errors(
        split.test, predictions, predictors, target, labels[0]
    )
    return {
        "split": split,
        "model": model,
        "predictions": predictions,
        "prediction_details": prediction_details,
        "actual": actual,
        "metrics": metrics,
        "baseline_class": baseline,
        "baseline_metrics": baseline_metrics,
        "positive_class": positive,
        "errors": error_details,
        "overlapping_predictor_combinations": _feature_overlap(
            split.train, split.test, predictors
        ),
        "training_rows": len(split.train),
        "test_rows": len(split.test),
        "split_warnings": split.warnings,
        "prediction_warnings": [
            warning
            for result in prediction_details
            for warning in result["warnings"]
        ],
    }