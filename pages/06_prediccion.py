import pandas as pd
import streamlit as st

from src.model.prediction import predict_observation
from src.utils.state import MODEL_KEY, initialize_state
from src.utils.ui import apply_app_styles, show_step_navigation

apply_app_styles()
initialize_state(st.session_state)
show_step_navigation(6)
st.title("6 · Predicción de una nueva observación")
st.caption(
    "Introduce los valores conocidos. Primero revisa las probabilidades y la "
    "evidencia; después puedes asignar la clase con mayor probabilidad."
)

model = st.session_state.get(MODEL_KEY)
if model is None:
    st.warning("Aún no hay un modelo entrenado con la selección actual.")
    st.page_link(
        "pages/05_entrenamiento.py",
        label="Entrenar el clasificador",
    )
    st.stop()

st.markdown("**Variables del modelo**")
st.write(
    f"Objetivo: **{model.target}** · Predictores: "
    + ", ".join(f"**{name}**" for name in model.predictors)
)
observation: dict[str, object] = {}
observation_complete = True
for predictor in model.predictors:
    category_values = list(model.categories[predictor])
    choice_data: list[tuple[str, object]] = [
        ("value", value) for value in category_values
    ]
    choice_data.append(("unknown", None))
    if model.missing_strategy == "category":
        choice_data.append(("missing", None))

    def format_choice(
        index: int, values: list[tuple[str, object]] = choice_data
    ) -> str:
        """Presenta categorías conocidas, faltantes y desconocidas como etiquetas."""
        kind, value = values[index]
        if kind == "unknown":
            return "Otra categoría (desconocida)"
        if kind == "missing":
            return "(Faltante)"
        return str(value)

    selected_index = st.selectbox(
        f"Valor para {predictor}",
        list(range(len(choice_data))),
        index=None,
        placeholder="Selecciona el valor observado",
        format_func=format_choice,
        key=f"prediction_input_{predictor}",
    )
    if selected_index is None:
        observation_complete = False
        continue
    kind, value = choice_data[selected_index]
    if kind == "unknown":
        unknown_value = "__NEW_CATEGORY__"
        while unknown_value in category_values:
            unknown_value += "_"
        observation[predictor] = unknown_value
    else:
        observation[predictor] = None if kind == "missing" else value

observation_signature = tuple(
    (predictor, type(value).__name__, repr(value))
    for predictor, value in observation.items()
)
if st.button(
    "Calcular probabilidades",
    type="primary",
    disabled=not observation_complete,
):
    try:
        result = predict_observation(model, observation)
        st.session_state["prediction_result"] = result
        st.session_state["prediction_result_signature"] = observation_signature
        st.session_state["prediction_assignment_confirmed"] = False
    except ValueError as error:
        st.error(f"No se pudo calcular la predicción: {error}")
if not observation_complete:
    st.caption("Selecciona un valor para cada predictor para continuar.")

result = st.session_state.get("prediction_result")
if (
    result is not None
    and st.session_state.get("prediction_result_signature")
    != observation_signature
):
    st.session_state.pop("prediction_result", None)
    st.session_state.pop("prediction_result_signature", None)
    st.session_state.pop("prediction_assignment_confirmed", None)
    result = None

if result:
    st.subheader("Probabilidades calculadas")
    probability_columns = st.columns(len(result["class_probabilities"]))
    for column, (class_name, probability) in zip(
        probability_columns, result["class_probabilities"].items()
    ):
        column.metric(f"P({class_name} | X)", f"{probability:.1%}")
    st.caption(
        "Son estimaciones basadas en los datos, no certezas ni probabilidades "
        "necesariamente calibradas."
    )

    st.markdown("#### ¿Qué evidencia se utilizó?")
    st.latex(
        r"P(C\mid X_1,\ldots,X_n)"
        r"=\frac{P(C)\prod_i P(X_i\mid C)}"
        r"{\sum_{C'}P(C')\prod_i P(X_i\mid C')}"
    )
    st.caption(
        "Las previas se calculan con las filas de entrenamiento. Las condicionales "
        "usan suavizado de Laplace; Naive Bayes supone independencia condicional "
        "entre predictores dada la clase."
    )
    comparison_rows = [
        {
            "clase": str(class_name),
            "probabilidad previa": details["prior_probability"],
            "peso relativo antes de normalizar": details["unnormalized_weight"],
            "probabilidad posterior": details["posterior_probability"],
        }
        for class_name, details in result["class_details"].items()
    ]
    st.dataframe(
        pd.DataFrame(comparison_rows).style.format(
            {
                "probabilidad previa": "{:.2%}",
                "peso relativo antes de normalizar": "{:.4f}",
                "probabilidad posterior": "{:.2%}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    for class_name, details in result["class_details"].items():
        with st.expander(f"Variables que apoyan la clase «{class_name}»"):
            evidence_rows = [
                {
                    "variable": predictor,
                    "valor observado": str(terms["value"]),
                    "frecuencia en clase": terms["count"],
                    "P(valor | clase)": terms["conditional_probability"],
                    "contribución logarítmica": terms["log_contribution"],
                    "categoría desconocida": terms["unknown_category"],
                }
                for predictor, terms in details["predictors"].items()
            ]
            st.dataframe(
                pd.DataFrame(evidence_rows).style.format(
                    {
                        "P(valor | clase)": "{:.2%}",
                        "contribución logarítmica": "{:.4f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
    for warning in result["warnings"]:
        st.warning(warning)

    if st.session_state.get("prediction_assignment_confirmed"):
        st.success(f"Clase asignada por mayor probabilidad: **{result['predicted_class']}**")
    elif st.button("Asignar clase con mayor probabilidad", type="primary"):
        st.session_state["prediction_assignment_confirmed"] = True
        st.rerun()

st.info(
    "La evidencia no demuestra causalidad. Si tus predictores están muy relacionados "
    "o el caso es distinto a los datos de entrenamiento, interpreta la estimación "
    "con cautela y consulta la evaluación fuera de muestra."
)
