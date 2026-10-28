"""Generación reproducible de partidas sintéticas de Vas o No Vas."""

from __future__ import annotations

import argparse
import csv
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final

from faker import Faker

from src.data.validation import (
    CSV_COLUMNS,
    calculate_remaining_highs,
    validate_dataset as validate_rows,
    validate_observation,
)

ALTO: Final = "Alto"
BAJO: Final = "Bajo"
TOTAL_MALETINES: Final = 20
ALTOS_TOTALES: Final = 6
BAJOS_TOTALES: Final = 14
MOMENTOS_VALIDOS: Final = tuple(range(2, 6))
COLUMNS: Final = CSV_COLUMNS
DEFAULT_OUTPUT: Final = Path(__file__).resolve().parents[2] / "data" / "raw" / "partidas.csv"


def crear_maletines(rng: random.Random) -> list[str]:
    """Crea los 20 premios requeridos y los distribuye con una mezcla aleatoria."""
    maletines = [ALTO] * ALTOS_TOTALES + [BAJO] * BAJOS_TOTALES
    rng.shuffle(maletines)
    return maletines


def validar_maletines(maletines: Iterable[str]) -> None:
    """Comprueba que una partida tenga exactamente sus 6 premios altos y 14 bajos."""
    premios = list(maletines)
    if len(premios) != TOTAL_MALETINES:
        raise ValueError("Cada partida debe tener exactamente 20 maletines.")
    if premios.count(ALTO) != ALTOS_TOTALES or premios.count(BAJO) != BAJOS_TOTALES:
        raise ValueError("Cada partida debe tener 6 premios altos y 14 bajos.")


def abrir_maletines(
    maletines: list[str], indice_inicial: int, rng: random.Random
) -> list[tuple[int, str, int]]:
    """Abre los maletines distintos del inicial y registra estados entre 2 y 5."""
    if len(maletines) != TOTAL_MALETINES or not 0 <= indice_inicial < TOTAL_MALETINES:
        raise ValueError("Se requieren 20 maletines y un índice inicial válido.")

    indices_disponibles = [index for index in range(TOTAL_MALETINES) if index != indice_inicial]
    rng.shuffle(indices_disponibles)

    estados: list[tuple[int, str, int]] = []
    altos_revelados = 0
    for abiertos, indice_abierto in enumerate(indices_disponibles, start=1):
        premio = maletines[indice_abierto]
        maletines_restantes = TOTAL_MALETINES - abiertos
        if maletines_restantes in MOMENTOS_VALIDOS:
            estados.append((maletines_restantes, premio, altos_revelados))
        if premio == ALTO:
            altos_revelados += 1
    return estados


def altos_restantes(
    maletines_restantes: int, altos_revelados: int, ultimo_premio: str
) -> int:
    """Calcula los premios altos aún cerrados usando las variables observadas."""
    return calculate_remaining_highs(altos_revelados, ultimo_premio)


def generar_partida(rng: random.Random, fake: Faker) -> dict[str, int | str]:
    """Simula una partida y usa Faker para elegir maletín y momento del registro."""
    maletines = crear_maletines(rng)
    validar_maletines(maletines)

    indice_inicial = fake.random_int(min=0, max=TOTAL_MALETINES - 1)
    premio_final = maletines[indice_inicial]
    estados = abrir_maletines(maletines, indice_inicial, rng)
    if len(estados) != len(MOMENTOS_VALIDOS):
        raise RuntimeError("La partida no produjo todos los momentos válidos.")

    maletines_restantes, ultimo_premio, altos_revelados = estados[
        fake.random_int(min=0, max=len(estados) - 1)
    ]
    return {
        "maletines_restantes": maletines_restantes,
        "altos_revelados": altos_revelados,
        "ultimo_premio": ultimo_premio,
        "premio_final": premio_final,
    }


def generar_dataset(
    numero_partidas: int = 1000, seed: int = 42
) -> list[dict[str, int | str]]:
    """Genera partidas reproducibles mediante Faker y random de la biblioteca estándar."""
    if isinstance(numero_partidas, bool) or not isinstance(numero_partidas, int) or numero_partidas < 1:
        raise ValueError("numero_partidas debe ser un entero mayor que cero.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed debe ser un entero.")

    rng = random.Random(seed)
    fake = Faker()
    fake.seed_instance(seed)
    registros = [generar_partida(rng, fake) for _ in range(numero_partidas)]
    validar_dataset(registros)
    return registros


def validar_registro(registro: Mapping[str, object]) -> None:
    """Valida categorías, estructura y coherencia matemática de una fila."""
    validate_observation(registro)


def validar_dataset(registros: Iterable[Mapping[str, object]]) -> None:
    """Valida todas las filas y rechaza datasets vacíos."""
    filas = list(registros)
    if not filas:
        raise ValueError("El dataset no contiene registros.")
    validate_rows(filas)


def exportar_csv(registros: Iterable[Mapping[str, object]], ruta: str | Path) -> Path:
    """Exporta las cuatro columnas a un archivo nuevo sin sobrescribir otro."""
    filas = list(registros)
    validar_dataset(filas)
    destino = Path(ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)

    with destino.open("x", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=COLUMNS)
        escritor.writeheader()
        escritor.writerows(filas)
    return destino


def comprobar_aleatoriedad(registros: Iterable[Mapping[str, object]]) -> float:
    """Devuelve la proporción de premios altos observada, sin alterar ni rechazarla."""
    filas = list(registros)
    if not filas:
        raise ValueError("No se puede calcular la proporción de un dataset vacío.")
    return sum(registro["premio_final"] == ALTO for registro in filas) / len(filas)


def main() -> None:
    """Ejecuta la interfaz de línea de comandos y exporta un dataset generado."""
    parser = argparse.ArgumentParser(description="Genera partidas sintéticas del juego.")
    parser.add_argument("--games", type=int, default=1000, help="Cantidad de partidas.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla reproducible.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Ruta del CSV de salida (no se sobrescriben archivos).",
    )
    args = parser.parse_args()
    registros = generar_dataset(args.games, seed=args.seed)
    destino = exportar_csv(registros, args.output)
    print(f"CSV generado: {destino} ({len(registros)} partidas, semilla {args.seed})")


if __name__ == "__main__":
    main()
