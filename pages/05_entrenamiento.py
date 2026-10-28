import pandas as pd
import streamlit as st

from src.model.naive_bayes import fit_naive_bayes
from src.model.evaluation import split_train_test
from src.utils.state import (
    DATASET_FINGERPRINT_KEY,
    MODEL_KEY,
    MODEL_SIGNATURE_KEY,
    initialize_state,
)
from src.utils.ui import apply_app_styles, show_step_navigation

apply_app_styles()
initialize_state(st.session_state)
show_step_navigation(5)
st.title("5 · Entrenamiento del clasificador")
st.caption(
    "Separa filas de prueba y aprende Naive Bayes únicamente con las filas de "
    "entrenamiento. La división se puede reproducir mediante la semilla."
)

dataframe = st.session_state.get("dataset")
target = st.session_state.get("selected_target")
predictors = st.session_state.get("selected_predictors", [])
if dataframe is None or target not in getattr(dataframe, "columns", []):
    st.warning("Carga datos y selecciona el objetivo en las etapas anteriores.")
    st.page_link("pages/01_carga.py", label="Ir a cargar datos")
    st.stop()
if not predictors:
    st.warning("Regresa a «3 · Selección de variables» y selecciona al menos un predictor.")
    st.page_link("pages/03_variables.py", label="Elegir predictores")
    st.stop()

with st.expander("Ajustes del entrenamiento", expanded=False):
    config_columns = st.columns(3)
    alpha = config_columns[0].number_input(
        "Suavizado Laplace (α)", min_value=0.01, value=1.0, step=0.1,
        help="Evita que una categoría no observada produzca probabilidad cero.",
    )
    test_size_percent = config_columns[1].slider(
        "Datos reservados para comprobar el modelo (%)",
        min_value=10,
        max_value=40,
        value=20,
        help="Estas filas no se usan para aprender las probabilidades.",
    )
    seed = config_columns[2].number_input(
        "Semilla reproducible", min_value=0, value=42, step=1
    )
    missing_strategy = st.selectbox(
        "Cómo tratar los valores faltantes",
        ["error", "category"],
        format_func=lambda option: (
            "Avisar y detener" if option == "error" else "Tratarlos como categoría"
        ),
    )
test_size = test_size_percent / 100
signature = (
    st.session_state.get(DATASET_FINGERPRINT_KEY),
    target,
    tuple(predictors),
    float(alpha),
    test_size,
    int(seed),
    missing_strategy,
)
model = st.session_state.get(MODEL_KEY)
if model is not None and st.session_state.get(MODEL_SIGNATURE_KEY) != signature:
    for key in (
        "trained_model",
        "model_signature",
        "model_train_rows",
        "model_test_rows",
        "model_split_info",
        "prediction_result",
        "prediction_result_signature",
        "prediction_assignment_confirmed",
    ):
        st.session_state.pop(key, None)
    for predictor in predictors:
        st.session_state.pop(f"prediction_input_{predictor}", None)
    model = None
    st.info("Los datos o parámetros cambiaron; el modelo anterior se invalidó.")

st.info(
    f"El entrenamiento aprenderá de aproximadamente {100 - test_size_percent}% "
    f"de las filas; el {test_size_percent}% restante se reserva para comprobación."
)
if st.button("Entrenar clasificador bayesiano", type="primary"):
    try:
        split = split_train_test(
            dataframe,
            target,
            test_size=test_size,
            seed=int(seed),
            stratified=True,
        )
        fitted_model = fit_naive_bayes(
            split.train,
            target,
            predictors,
            alpha=float(alpha),
            missing_strategy=missing_strategy,
        )
        st.session_state[MODEL_KEY] = fitted_model
        st.session_state[MODEL_SIGNATURE_KEY] = signature
        st.session_state["model_train_rows"] = split.train
        st.session_state["model_test_rows"] = split.test
        st.session_state["model_split_info"] = split
        st.session_state.pop("prediction_result", None)
        model = fitted_model
        st.success(
            f"Modelo entrenado con {len(split.train)} filas. "
            f"{len(split.test)} filas quedaron reservadas y no se usaron para aprender."
        )
        if split.warnings:
            for warning in split.warnings:
                st.warning(warning)
    except (ValueError, KeyError) as error:
        st.error(f"No se pudo entrenar: {error}")

if model is None:
    st.info("Aún no hay un modelo vigente. Entrena para habilitar las predicciones.")
    st.stop()

st.markdown("#### Modelo entrenado")
model_metrics = st.columns(3)
model_metrics[0].metric("Filas de entrenamiento", f"{model.training_rows:,}")
model_metrics[1].metric("Filas reservadas", len(st.session_state.get("model_test_rows", [])))
model_metrics[2].metric("Clases", len(model.classes))
st.markdown("**Probabilidad inicial por clase**")
prior_rows = [
    {
        "clase": str(class_value),
        "P(clase)": model.prior_probability(class_value),
    }
    for class_value in model.classes
]
st.dataframe(
    pd.DataFrame(prior_rows).style.format({"P(clase)": "{:.1%}"}),
    width="stretch",
    hide_index=True,
)
if model.warnings:
    for warning in model.warnings:
        st.warning(warning)
st.info(
    "Naive Bayes supone independencia condicional entre predictores dada la clase. "
    "Este supuesto puede afectar las estimaciones si las variables están relacionadas."
)
st.success("Siguiente: usa «6 · Predicción» para introducir una observación.")