from collections.abc import Mapping, Sequence
from numbers import Integral


CSV_COLUMNS = (
    "maletines_restantes",
    "altos_revelados",
    "ultimo_premio",
    "premio_final",
)


def calculate_remaining_highs(revealed_highs: int, latest_prize: str) -> int:
    """Calcula premios altos restantes y rechaza conteos o categorías imposibles."""
    if isinstance(revealed_highs, bool) or not isinstance(revealed_highs, Integral):
        raise ValueError("altos_revelados debe ser un número entero.")
    revealed_highs = int(revealed_highs)
    if not 0 <= revealed_highs <= 6:
        raise ValueError("altos_revelados debe estar entre 0 y 6.")
    if not isinstance(latest_prize, str) or latest_prize not in ("Alto", "Bajo"):
        raise ValueError("ultimo_premio debe ser 'Alto' o 'Bajo'.")
    remaining = 6 - revealed_highs - int(latest_prize == "Alto")
    if not 0 <= remaining <= 5:
        raise ValueError("La cantidad de premios altos restantes es imposible.")
    return remaining


def _read_integer(value: object, field: str) -> int:
    """Convierte una celda entera CSV y rechaza booleanos y formatos ambiguos."""
    if isinstance(value, bool):
        raise ValueError(f"{field} debe ser un número entero.")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    raise ValueError(f"{field} debe ser un número entero.")


def validate_observation(observation: Mapping[str, object]) -> None:
    """Valida que una fila represente un estado posible de una partida."""
    if not isinstance(observation, Mapping):
        raise ValueError("observation debe ser un mapeo de columnas a valores.")
    if tuple(observation.keys()) != CSV_COLUMNS:
        raise ValueError(
            "Las columnas deben ser exactamente, y en este orden: "
            + ", ".join(CSV_COLUMNS)
        )

    remaining_cases = _read_integer(
        observation["maletines_restantes"], "maletines_restantes"
    )
    revealed_highs = _read_integer(observation["altos_revelados"], "altos_revelados")
    latest_prize = observation["ultimo_premio"]
    initial_prize = observation["premio_final"]

    if remaining_cases not in (2, 3, 4, 5):
        raise ValueError("maletines_restantes debe estar entre 2 y 5.")
    if not isinstance(latest_prize, str) or latest_prize not in ("Alto", "Bajo"):
        raise ValueError("ultimo_premio debe ser 'Alto' o 'Bajo'.")
    if not isinstance(initial_prize, str) or initial_prize not in ("Alto", "Bajo"):
        raise ValueError("premio_final debe ser 'Alto' o 'Bajo'.")

    previously_opened = 20 - remaining_cases - 1
    previously_revealed_lows = previously_opened - revealed_highs
    latest_is_low = int(latest_prize == "Bajo")
    remaining_highs = calculate_remaining_highs(revealed_highs, str(latest_prize))
    remaining_lows = 14 - previously_revealed_lows - latest_is_low

    if not 0 <= revealed_highs <= 6:
        raise ValueError("altos_revelados debe estar entre 0 y 6.")
    if not 0 <= previously_revealed_lows <= 14:
        raise ValueError("La cantidad de premios bajos revelados no es posible.")
    if not 0 <= remaining_highs <= remaining_cases:
        raise ValueError(
            "Los premios altos restantes deben estar entre 0 y los maletines cerrados."
        )
    if not 0 <= remaining_lows <= remaining_cases:
        raise ValueError(
            "Los premios bajos restantes deben estar entre 0 y los maletines cerrados."
        )
    if initial_prize == "Alto" and remaining_highs == 0:
        raise ValueError("El maletín inicial no puede ser Alto: no quedan premios altos.")
    if initial_prize == "Bajo" and remaining_lows == 0:
        raise ValueError("El maletín inicial no puede ser Bajo: no quedan premios bajos.")


def validate_dataset(rows: Sequence[Mapping[str, object]]) -> None:
    """Valida cada fila de un dataset no vacío y señala la posición del error."""
    if not rows:
        raise ValueError("El dataset no contiene registros.")
    for row_number, observation in enumerate(rows, start=1):
        try:
            validate_observation(observation)
        except ValueError as error:
            raise ValueError(f"Fila {row_number}: {error}") from error