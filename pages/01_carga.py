from pathlib import Path

import pandas as pd
import streamlit as st

from src.data.loader import load_csv, validate_game_dataset
from src.utils.state import dataframe_fingerprint, initialize_state, set_dataset
from src.utils.ui import apply_app_styles, show_step_navigation

apply_app_styles()
initialize_state(st.session_state)
show_step_navigation(1)
st.title("1 · Carga y validación de datos")
st.caption(
    "Carga un archivo CSV o prueba con el dataset de ejemplo. Tus datos se "
    "mantienen en esta sesión; el archivo original no se modifica."
)

demo_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "partidas.csv"
upload = st.file_uploader(
    "Selecciona un archivo CSV",
    type=["csv"],
    help="Se admiten archivos .csv. El dataset actual se conserva si el nuevo archivo no es válido.",
)
if upload is not None:
    try:
        external = pd.read_csv(upload)
        if external.empty:
            raise ValueError("El archivo CSV no contiene registros.")
        if external.columns.has_duplicates:
            raise ValueError("Hay nombres de columnas repetidos; cámbialos y vuelve a cargar.")
        if any(not str(column).strip() for column in external.columns):
            raise ValueError("Todas las columnas deben tener un nombre.")
        source_name = f"{upload.name} · externo"
        if st.session_state.get("dataset_fingerprint") != dataframe_fingerprint(external):
            set_dataset(st.session_state, external, source_name)
        elif st.session_state.get("dataset_name") != source_name:
            st.session_state["dataset_name"] = source_name
        st.success(
            f"Archivo listo: {len(external):,} filas y "
            f"{len(external.columns)} columnas. Puedes continuar a Exploración."
        )
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"No se pudo cargar el archivo: {error}")
        st.caption("Revisa que sea un CSV legible, con encabezados y al menos una fila.")

st.divider()
left, right = st.columns([1, 2], vertical_alignment="center")
with left:
    if st.button("Probar con datos de ejemplo", type="primary", width="stretch"):
        try:
            demo = load_csv(demo_path)
            expected_columns = (
                "maletines_restantes",
                "altos_revelados",
                "ultimo_premio",
                "premio_final",
            )
            if tuple(demo.columns) == expected_columns:
                validate_game_dataset(demo)
            set_dataset(st.session_state, demo, "partidas.csv · demostración")
            st.success("Dataset de ejemplo cargado.")
        except (OSError, ValueError, pd.errors.ParserError) as error:
            st.error(f"No se pudo cargar el dataset de ejemplo: {error}")
with right:
    st.info(
        "¿Es tu primera vez? Usa los datos de ejemplo para seguir el recorrido "
        "sin preparar un archivo propio."
    )

dataframe = st.session_state.get("dataset")
if dataframe is not None:
    st.subheader("Dataset activo")
    st.write(f"**{st.session_state.get('dataset_name', 'Dataset en sesión')}**")
    metrics = st.columns(3)
    metrics[0].metric("Filas", f"{len(dataframe):,}")
    metrics[1].metric("Columnas", len(dataframe.columns))
    metrics[2].metric("Valores faltantes", f"{int(dataframe.isna().sum().sum()):,}")
    st.dataframe(dataframe.head(8), width="stretch", hide_index=True)
    st.success("Carga completada. Continúa a «2 · Exploración del dataset».")
else:
    st.warning("Aún no hay un dataset. Carga un CSV o selecciona los datos de ejemplo.")
