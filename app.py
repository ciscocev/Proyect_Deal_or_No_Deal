from pathlib import Path

import pandas as pd
import streamlit as st

from src.data.loader import load_csv, validate_game_dataset
from src.utils.state import initialize_state, set_dataset
from src.utils.ui import WORKFLOW_STAGES, apply_app_styles, stage_complete

st.set_page_config(
    page_title="Deal or No Deal | Analizador bayesiano",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="auto",
)
apply_app_styles()
initialize_state(st.session_state)
dataset_path = Path(__file__).resolve().parent / "data" / "raw" / "partidas.csv"
if st.session_state.get("dataset") is None and dataset_path.is_file():
    try:
        demo = load_csv(dataset_path)
        expected_game_columns = (
            "maletines_restantes",
            "altos_revelados",
            "ultimo_premio",
            "premio_final",
        )
        if tuple(demo.columns) == expected_game_columns:
            validate_game_dataset(demo)
        set_dataset(st.session_state, demo, "partidas.csv · demostración")
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"No se pudo iniciar el dataset de demostración: {error}")

with st.sidebar:
    st.markdown("# 🎲 DEAL OR NO DEAL")
    st.caption("Analizador de datos y clasificador bayesiano")
    st.markdown("---")
    if st.session_state.get("dataset") is not None:
        st.caption(f"Dataset activo: {st.session_state.get('dataset_name')}")
        st.caption(
            f"{len(st.session_state['dataset']):,} filas · "
            f"{len(st.session_state['dataset'].columns)} columnas"
        )

st.title("🎲 DEAL OR NO DEAL")
st.subheader("Convierte tus datos en evidencia")
st.markdown(
    "Un recorrido guiado para explorar tus datos, entrenar Naive Bayes y "
    "entender la evidencia detrás de cada predicción."
)
st.info(
    "No necesitas experiencia previa: avanza paso a paso, consulta las ayudas y "
    "vuelve a etapas anteriores cuando quieras. Tus datos y selecciones se conservan."
)

if dataset_path.is_file():
    st.success(
        f"Dataset activo: **{st.session_state.get('dataset_name', 'partidas.csv · demostración')}**"
    )
else:
    st.warning("No se encontró el dataset de demostración. Puedes cargar un CSV en la etapa 1.")

st.markdown("### Tu recorrido · 7 etapas")
stage_descriptions = (
    "Carga un CSV y valida que se pueda leer.",
    "Revisa las columnas, faltantes y distribuciones.",
    "Decide qué clase estimar y qué predictores usar.",
    "Consulta probabilidades de eventos en los datos.",
    "Entrena Bayes y reserva datos para evaluación.",
    "Introduce una observación, revisa evidencia y predicción.",
    "Mide el rendimiento con datos que no entrenaron el modelo.",
)
for start in range(0, len(WORKFLOW_STAGES), 3):
    row = list(enumerate(WORKFLOW_STAGES[start : start + 3], start=start + 1))
    for column, (number, (title, page)) in zip(st.columns(3), row):
        with column:
            with st.container(border=True):
                st.markdown(f"#### {number} · {title}")
                st.write(stage_descriptions[number - 1])
                complete = stage_complete(number)
                st.caption("✓ Completado" if complete else "Pendiente")
                st.page_link(page, label="Abrir sección", width="stretch")

if st.session_state.get("selected_predictors"):
    st.markdown("### Tu selección actual")
    st.write(
        f"Objetivo: **{st.session_state.get('selected_target')}** · "
        f"Predictores: **{', '.join(st.session_state['selected_predictors'])}**"
    )
st.markdown(
    "**Al finalizar podrás responder:** qué clase estima el modelo, qué evidencia "
    "y variables influyeron, cómo rindió en datos de prueba y bajo qué limitaciones "
    "conviene interpretar los resultados."
)