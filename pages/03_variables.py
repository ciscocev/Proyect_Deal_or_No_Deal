import streamlit as st

from src.analysis.frequencies import frequency_table
from src.analysis.profiling import is_continuous_numeric
from src.analysis.relationships import conditional_proportions, joint_frequencies
from src.utils.helpers import validate_predictor_selection
from src.utils.state import (
    initialize_state,
    set_variable_selection,
)
from src.utils.ui import apply_app_styles, show_step_navigation
from src.visualization.plots import distribution_plot, target_relationship_plot

apply_app_styles()
initialize_state(st.session_state)
show_step_navigation(3)
st.title("3 · Selección de variables")
st.caption(
    "Define qué quieres estimar y qué evidencia estará disponible para Bayes. "
    "Tú eliges las variables; los datos te ayudan a justificar la decisión."
)

dataframe = st.session_state.get("dataset")
if dataframe is None:
    st.warning("Primero carga un dataset en «1 · Carga de datos».")
    st.page_link("pages/01_carga.py", label="Ir a cargar datos")
    st.stop()
columns = list(dataframe.columns)
current_target = st.session_state.get("selected_target")
target_index = columns.index(current_target) if current_target in columns else None
target = st.selectbox(
    "Variable objetivo (Y)",
    columns,
    index=target_index,
    key="target_selector",
    placeholder="Selecciona qué resultado quieres estimar",
)
if target is None:
    st.info("Primero elige una variable objetivo para ver las opciones predictoras.")
    st.stop()

predictor_options = [column for column in columns if column != target]
current_predictors = [
    column
    for column in st.session_state.get("selected_predictors", [])
    if column in predictor_options
]
predictors = st.multiselect(
    "Variables predictoras (X)",
    predictor_options,
    default=current_predictors,
    key="predictor_selector",
    help=(
        "Bayes usará los valores de estas columnas para estimar la clase. "
        "La variable objetivo no puede ser predictor."
    ),
)
selection_changed = (
    st.session_state.get("selected_target") != target
    or st.session_state.get("selected_predictors") != predictors
)
if selection_changed:
    set_variable_selection(st.session_state, target, predictors)
    st.info(
        "Selección actualizada y guardada en esta sesión. "
        "Los resultados del modelo anterior se invalidaron."
    )

st.info(
    "La variable objetivo es la clase que quieres estimar. Los predictores son "
    "la evidencia que Bayes tendrá disponible para hacerlo."
)
st.latex(r"P(\text{objetivo}\mid\text{predictores})")

st.markdown("#### Conoce las variables")
evidence_column = st.selectbox("Columna descriptiva", columns, key="selection_evidence_column")
st.dataframe(
    frequency_table(dataframe[evidence_column]),
    width="stretch",
)

if predictors:
    st.markdown(f"#### Explora los predictores frente a **{target}**")
    st.caption(
        "Para cada predictor marcado puedes revisar su propia distribución, "
        "comparar sus valores entre las clases y consultar las frecuencias "
        "conjuntas y condicionales. Son asociaciones observadas, no relaciones "
        "causales ni garantía de capacidad predictiva."
    )
    st.markdown(f"**Distribución del objetivo · {target}**")
    st.plotly_chart(distribution_plot(dataframe, target), width="stretch")
    for predictor in predictors:
        with st.expander(f"Evidencia: {predictor} y {target}", expanded=False):
            st.markdown(f"**Distribución de {predictor}**")
            st.plotly_chart(
                distribution_plot(dataframe, predictor),
                width="stretch",
            )
            st.markdown(f"**Comparación de {predictor} entre clases de {target}**")
            st.plotly_chart(
                target_relationship_plot(dataframe, predictor, target),
                width="stretch",
            )
            if is_continuous_numeric(dataframe[predictor]) or is_continuous_numeric(
                dataframe[target]
            ):
                st.caption(
                    "La tabla conserva los valores numéricos originales y puede "
                    "tener muchas filas; la gráfica resume visualmente su relación."
                )
            st.markdown("**Frecuencias conjuntas**")
            st.dataframe(
                joint_frequencies(dataframe, predictor, target),
                width="stretch",
            )
            st.markdown(f"**Distribución de {target} condicionada a {predictor}**")
            st.dataframe(
                conditional_proportions(dataframe, predictor, target).style.format(
                    "{:.1%}"
                ),
                width="stretch",
            )
else:
    st.info("Elige al menos un predictor para revisar su relación con el objetivo.")

with st.expander("Justifica tus decisiones", expanded=False):
    justifications: dict[str, str] = {}
    for column in columns:
        if column == target:
            continue
        if column in predictors:
            label = f"Justificación para incluir «{column}»"
            key = f"include_reason_{column}"
        else:
            label = f"Justificación para excluir «{column}»"
            key = f"exclude_reason_{column}"
        justifications[column] = st.text_input(
            label, key=key, placeholder="¿Qué información aporta o por qué no se usará?"
        )
    target_justification = st.text_input(
        f"Justificación para elegir «{target}» como objetivo",
        key="target_reason",
        placeholder="¿Qué resultado quieres estimar?",
    )

if st.button("Guardar justificaciones", type="primary"):
    try:
        validate_predictor_selection(columns, target, predictors)
        st.session_state["selection_justifications"] = {
            **justifications,
            "__target__": target_justification,
        }
        st.success("Justificaciones guardadas en esta sesión.")
    except ValueError as error:
        st.error(str(error))

if st.session_state.get("selected_target") in columns:
    st.success(
        f"Selección activa: objetivo **{st.session_state['selected_target']}**; "
        f"predictores **{', '.join(st.session_state.get('selected_predictors', []))}**."
    )
    if st.session_state.get("selected_predictors"):
        st.success(
            "La selección está disponible para entrenar el clasificador."
        )