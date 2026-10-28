import random
import csv
import tempfile
import unittest
from pathlib import Path

from src.data.generator import (
    ALTO,
    BAJO,
    COLUMNS,
    crear_maletines,
    exportar_csv,
    abrir_maletines,
    altos_restantes,
    comprobar_aleatoriedad,
    generar_dataset,
    validar_dataset,
    validar_maletines,
    validar_registro,
)


class GeneratorTests(unittest.TestCase):
    def test_maletines_tienen_seis_altos_y_catorce_bajos(self):
        maletines = crear_maletines(random.Random(123))
        validar_maletines(maletines)
        self.assertEqual(len(maletines), 20)
        self.assertEqual(maletines.count(ALTO), 6)
        self.assertEqual(maletines.count(BAJO), 14)

    def test_registros_matematicos_son_coherentes(self):
        registros = generar_dataset(1000, seed=42)
        validar_dataset(registros)
        for registro in registros:
            validar_registro(registro)
            restantes = int(registro["maletines_restantes"])
            revelados = int(registro["altos_revelados"])
            altos = altos_restantes(restantes, revelados, str(registro["ultimo_premio"]))
            self.assertGreaterEqual(altos, 0)
            self.assertLessEqual(altos, restantes)

    def test_dataset_es_reproducible(self):
        self.assertEqual(generar_dataset(100, seed=7), generar_dataset(100, seed=7))

    def test_proporcion_observada_se_calcula_sin_alterar_datos(self):
        registros = generar_dataset(1000, seed=42)
        original = list(registros)
        proporcion = comprobar_aleatoriedad(registros)
        self.assertGreaterEqual(proporcion, 0)
        self.assertLessEqual(proporcion, 1)
        self.assertEqual(registros, original)

    def test_momento_de_registro_cubre_los_cuatro_valores(self):
        registros = generar_dataset(1000, seed=42)
        self.assertEqual(
            {registro["maletines_restantes"] for registro in registros},
            {2, 3, 4, 5},
        )

    def test_csv_tiene_exclusivamente_las_columnas_requeridas(self):
        registros = generar_dataset(10, seed=12)
        with tempfile.TemporaryDirectory() as temporary_directory:
            ruta = Path(temporary_directory) / "partidas.csv"
            exportar_csv(registros, ruta)
            with ruta.open(newline="", encoding="utf-8") as archivo:
                lector = csv.DictReader(archivo)
                self.assertEqual(tuple(lector.fieldnames or ()), COLUMNS)
                self.assertEqual(len(list(lector)), 10)

    def test_exportar_no_sobrescribe_archivo_existente(self):
        registros = generar_dataset(5, seed=3)
        with tempfile.TemporaryDirectory() as temporary_directory:
            ruta = Path(temporary_directory) / "partidas.csv"
            ruta.write_text("contenido original", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                exportar_csv(registros, ruta)
            self.assertEqual(ruta.read_text(encoding="utf-8"), "contenido original")

    def test_rechaza_cantidad_de_partidas_invalida(self):
        with self.assertRaisesRegex(ValueError, "numero_partidas"):
            generar_dataset(0)

    def test_rechaza_premio_inicial_imposible(self):
        registro = {
            "maletines_restantes": 5,
            "altos_revelados": 0,
            "ultimo_premio": "Alto",
            "premio_final": "Bajo",
        }
        with self.assertRaisesRegex(ValueError, "no puede ser Bajo"):
            validar_registro(registro)
