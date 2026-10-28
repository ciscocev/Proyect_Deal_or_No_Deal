"""Shared Streamlit presentation styles."""

import streamlit as st

WORKFLOW_STAGES = (
    ("Carga de datos", "pages/01_carga.py"),
    ("Exploración", "pages/02_exploracion.py"),
    ("Variables", "pages/03_variables.py"),
    ("Probabilidades", "pages/04_probabilidades.py"),
    ("Entrenamiento", "pages/05_entrenamiento.py"),
    ("Predicción", "pages/06_prediccion.py"),
    ("Evaluación", "pages/07_evaluacion.py"),
)


def apply_app_styles() -> None:
    """Apply the shared black-and-yellow visual theme to the current page."""
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
            --deal-night: #090909;
            --deal-sidebar: #111111;
            --deal-panel: #1B1B1B;
            --deal-white: #FFFFFF;
            --deal-muted: #D1D1D1;
            --deal-gold: #FFD400;
            --deal-border: #383838;
        }
        .stApp,
        [data-testid="stAppViewContainer"] {
            background: var(--deal-night);
            color: var(--deal-white);
        }
        [data-testid="stSidebar"] {
            background: var(--deal-sidebar);
            border-right: 1px solid var(--deal-border);
        }
        .stApp h1 {
            color: var(--deal-gold);
        }
        .stApp h2, .stApp h3, .stApp h4 {
            color: var(--deal-white);
        }
        .stApp p, .stApp li, .stApp label,
        [data-testid="stCaptionContainer"],
        [data-testid="stWidgetLabel"] {
            color: var(--deal-muted);
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] li,
        [data-testid="stSidebar"] label {
            color: var(--deal-muted);
        }
        [data-testid="stSidebar"] a {
            color: var(--deal-gold);
        }
        .stApp a {
            color: var(--deal-gold);
        }
        [data-testid="stMetric"] {
            background: var(--deal-panel);
            border: 1px solid var(--deal-border);
            border-radius: 12px;
            padding: 0.85rem 1rem;
        }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"] {
            color: var(--deal-white);
        }
        .stButton > button {
            border-radius: 9px;
            font-weight: 600;
            transition: border-color 120ms ease, background-color 120ms ease;
        }
        .stButton > button[kind="primary"] {
            background: var(--deal-gold);
            border-color: var(--deal-gold);
            color: #090909;
        }
        .stButton > button[kind="primary"] * {
            color: #090909;
        }
        .stButton > button[kind="primary"]:hover {
            background: #FFE45C;
            border-color: #FFE45C;
            color: #090909;
        }
        .stButton > button[kind="secondary"] {
            background: var(--deal-panel);
            border-color: #52637A;
            color: var(--deal-white);
        }
        .stButton > button[kind="secondary"] * {
            color: var(--deal-white);
        }
        input, textarea {
            color: var(--deal-white) !important;
        }
        [data-testid="stFileUploader"] section {
            background: var(--deal-panel);
            border-color: #565656;
        }
        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            border: 1px solid var(--deal-border);
            border-radius: 8px;
            overflow: hidden;
        }
        .deal-card {
            box-sizing: border-box;
            height: 100%;
            min-height: 142px;
            padding: 1.1rem 1.15rem;
            background: var(--deal-panel);
            border: 1px solid var(--deal-border);
            border-top: 3px solid var(--deal-gold);
            border-radius: 12px;
        }
        .deal-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(min(100%, 205px), 1fr));
            gap: 0.85rem;
        }
        .deal-card h3 {
            margin: 0 0 0.6rem;
            color: var(--deal-white);
            font-size: 1.08rem;
        }
        .deal-card p {
            margin: 0;
            color: var(--deal-muted);
            line-height: 1.5;
        }
        @media (max-width: 760px) {
            .deal-card {
                min-height: 120px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def stage_complete(stage_number: int) -> bool:
    """Indica si el estado actual contiene el resultado clave de una etapa."""
    state = st.session_state
    if stage_number in (1, 2):
        return state.get("dataset") is not None
    if stage_number == 3:
        return bool(state.get("selected_target") and state.get("selected_predictors"))
    if stage_number == 4:
        return state.get("probability_result") is not None
    if stage_number == 5:
        return state.get("trained_model") is not None
    if stage_number == 6:
        return state.get("prediction_result") is not None
    if stage_number == 7:
        return state.get("evaluation_result") is not None
    return False


def show_step_navigation(current_stage: int) -> None:
    """Show progress, direct navigation, and back/next links without resetting state."""
    if not 1 <= current_stage <= len(WORKFLOW_STAGES):
        raise ValueError("current_stage debe estar entre 1 y 7.")

    title, _ = WORKFLOW_STAGES[current_stage - 1]
    st.progress(
        current_stage / len(WORKFLOW_STAGES),
        text=f"Etapa {current_stage} de {len(WORKFLOW_STAGES)} · {title}",
    )
    with st.container(border=True):
        columns = st.columns([1, 3, 1])
        if current_stage > 1:
            previous_title, previous_page = WORKFLOW_STAGES[current_stage - 2]
            columns[0].page_link(
                previous_page,
                label=f"← Anterior · {previous_title}",
                width="stretch",
            )
        columns[1].caption(
            "Puedes moverte entre etapas en cualquier momento. Tus datos y "
            "selecciones permanecen guardados durante esta sesión."
        )
        if current_stage < len(WORKFLOW_STAGES):
            next_title, next_page = WORKFLOW_STAGES[current_stage]
            columns[2].page_link(
                next_page,
                label=f"Siguiente · {next_title} →",
                width="stretch",
            )
        with st.expander("Ver recorrido y estado de cada etapa"):
            for number, (stage_title, page) in enumerate(WORKFLOW_STAGES, start=1):
                status = "✓ Listo" if stage_complete(number) else "Pendiente"
                if number == current_stage:
                    status = "● En curso"
                st.page_link(
                    page,
                    label=f"{number}. {stage_title} · {status}",
                    width="stretch",
                )
