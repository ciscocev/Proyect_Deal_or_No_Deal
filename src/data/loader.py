"""Carga de CSV y validación de la estructura de partidas."""

from pathlib import Path
from collections.abc import Sequence

import pandas as pd

from src.data.validation import CSV_COLUMNS, validate_observation


def load_csv(
    path: str | Path, expected_columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """Carga un CSV sin modificarlo y, opcionalmente, exige un esquema concreto."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo CSV: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    if expected_columns is not None and tuple(dataframe.columns) != tuple(expected_columns):
        raise ValueError(
            "Las columnas del CSV no coinciden con las esperadas: "
            f"{tuple(dataframe.columns)}"
        )
    return dataframe


def validate_game_dataset(dataframe: pd.DataFrame) -> None:
    """Valida que un DataFrame tenga exactamente el esquema y las filas del juego."""
    if tuple(dataframe.columns) != CSV_COLUMNS:
        raise ValueError(
            "El dataset de partidas debe tener exactamente estas columnas y orden: "
            + ", ".join(CSV_COLUMNS)
        )
    if dataframe.empty:
        raise ValueError("El dataset de partidas no contiene registros.")
    if dataframe.isna().any().any():
        raise ValueError("El dataset de partidas contiene valores faltantes.")

    for row_number, row in enumerate(dataframe.to_dict(orient="records"), start=1):
        try:
            validate_observation(row)
        except ValueError as error:
            raise ValueError(f"Fila {row_number}: {error}") from error