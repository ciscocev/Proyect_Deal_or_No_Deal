import pandas as pd
import streamlit as st

from src.model.evaluation import evaluate_naive_bayes
from src.utils.state import (
    DATASET_FINGERPRINT_KEY,
    EVALUATION_KEY,
    initialize_state,
)
from src.utils.ui import apply_app_styles, show_step_navigation

apply_app_styles()
initialize_state(st.session_state)
show_step_navigation(7)
st.title("7 · Evaluación e interpretación")
st.caption("El rendimiento se calcula con filas no usadas al ajustar los parámetros del modelo.")

dataframe = st.session_state.get("dataset")
target = st.session_state.get("selected_target")
predictors = st.session_state.get("selected_predictors", [])
if dataframe is None or target not in getattr(dataframe, "columns", []):
    st.warning("Carga un dataset y selecciona las variables en las etapas anteriores.")
    st.page_link("pages/01_carga.py", label="Ir a cargar datos")
    st.stop()
if not predictors:
    st.warning("Regresa a «3 · Selección de variables» y elige predictores.")
    st.page_link("pages/03_variables.py", label="Elegir predictores")
    st.stop()

classes = dataframe[target].dropna().unique().tolist()
if not classes:
    st.error("La variable objetivo no contiene clases.")
    st.stop()
default_positive_index = classes.index("Alto") if "Alto" in classes else 0
control_columns = st.columns(6)
test_size_percent = control_columns[0].slider(
    "Conjunto de prueba (%)", 10, 40, 20, key="evaluation_test_percent"
)
seed = control_columns[1].number_input(
    "Semilla de división", min_value=0, value=42, step=1, key="evaluation_seed"
)
alpha = control_columns[2].number_input(
    "Suavizado Laplace (α)", min_value=0.01, value=1.0, step=0.1,
    key="evaluation_alpha",
)
split_mode = control_columns[3].selectbox(
    "Estrategia",
    ["Aleatoria estratificada", "Agrupada por predictores"],
    key="evaluation_split_mode",
)
missing_strategy = control_columns[4].selectbox(
    "Valores faltantes",
    ["error", "category"],
    format_func=lambda value: "Rechazar" if value == "error" else "Tratar como categoría",
    key="evaluation_missing_strategy",
)
positive_class = control_columns[5].selectbox(
    "Clase positiva para FP/FN",
    classes,
    index=default_positive_index,
    key="evaluation_positive_class",
)
grouped = split_mode == "Agrupada por predictores"
evaluation_signature = (
    st.session_state.get(DATASET_FINGERPRINT_KEY),
    target,
    tuple(predictors),
    test_size_percent,
    int(seed),
    float(alpha),
    grouped,
    missing_strategy,
    positive_class,
)
if st.session_state.get("evaluation_signature") != evaluation_signature:
    st.session_state.pop(EVALUATION_KEY, None)

st.info(
    "La clase mayoritaria de referencia se determina solo con entrenamiento. "
    "Naive Bayes y la referencia se comparan en las mismas filas de prueba."
)
if grouped:
    st.warning(
        "La evaluación agrupada retiene combinaciones completas de predictores. "
        "El tamaño y balance de prueba pueden diferir; mide generalización a combinaciones no vistas."
    )

if st.button("Ejecutar evaluación", type="primary"):
    try:
        result = evaluate_naive_bayes(
            dataframe,
            target=target,
            predictors=predictors,
            test_size=test_size_percent / 100,
            seed=int(seed),
            alpha=float(alpha),
            stratified=not grouped,
            grouped=grouped,
            positive_class=positive_class,
            missing_strategy=missing_strategy,
        )
        st.session_state[EVALUATION_KEY] = result
        st.session_state["evaluation_signature"] = evaluation_signature
    except (ValueError, KeyError, RuntimeError) as error:
        st.session_state.pop(EVALUATION_KEY, None)
        st.error(f"No se pudo evaluar el modelo: {error}")

result = st.session_state.get(EVALUATION_KEY)
if result is None:
    st.info("Configura los parámetros y ejecuta la evaluación.")
    st.stop()

for warning in result["split_warnings"]:
    st.warning(warning)
if result["prediction_warnings"]:
    st.warning(
        f"Se generaron {len(result['prediction_warnings'])} advertencias de predicción, "
        "por ejemplo por categorías desconocidas."
    )

summary_top = st.columns(2)
summary_top[0].metric("Entrenamiento", f"{result['training_rows']:,}")
summary_top[1].metric("Prueba", f"{result['test_rows']:,}")
summary_bottom = st.columns(2)
summary_bottom[0].metric(
    "Combinaciones repetidas entre conjuntos",
    result["overlapping_predictor_combinations"],
)
summary_bottom[1].metric(
    "Clase mayoritaria de entrenamiento",
    str(result["baseline_class"]),
)

st.subheader("Matriz de confusión · Naive Bayes")
metrics = result["metrics"]
labels = metrics["labels"]
matrix = pd.DataFrame(
    metrics["confusion_matrix"],
    index=labels,
    columns=labels,
)
matrix.index.name = "Real ↓"
matrix.columns.name = "Predicha →"
st.dataframe(matrix, width="stretch")

metric_rows = []
for model_name, model_metrics in (
    ("Naive Bayes", result["metrics"]),
    ("Referencia mayoritaria", result["baseline_metrics"]),
):
    metric_rows.append(
        {
            "Modelo": model_name,
            "Exactitud": model_metrics["accuracy"],
            "Aciertos": model_metrics["correct"],
            "Errores": model_metrics["incorrect"],
        }
    )
st.subheader("Comparación general")
st.dataframe(pd.DataFrame(metric_rows), width="stretch", hide_index=True)
st.subheader("Métricas por clase")
class_metric_rows = []
for model_name, model_metrics in (
    ("Naive Bayes", result["metrics"]),
    ("Referencia mayoritaria", result["baseline_metrics"]),
):
    for class_name, values in model_metrics["per_class"].items():
        class_metric_rows.append(
            {
                "Modelo": model_name,
                "Clase": str(class_name),
                "Precisión": values["precision"],
                "Sensibilidad (recall)": values["recall"],
                "F1": values["f1"],
                "TP": values["true_positive"],
                "FP": values["false_positive"],
                "FN": values["false_negative"],
            }
        )
class_metrics = pd.DataFrame(class_metric_rows)
st.dataframe(
    class_metrics.style.format(
        {
            "Precisión": "{:.1%}",
            "Sensibilidad (recall)": "{:.1%}",
            "F1": "{:.1%}",
        }
    ),
    width="stretch",
    hide_index=True,
)
if result["positive_class"] is not None:
    positive_name = str(result["positive_class"])
    st.markdown(f"**Enfoque en {positive_name}:**")
    positive_metrics = result["metrics"]["per_class"].get(result["positive_class"])
    if positive_metrics:
        st.write(
            f"Precisión {positive_metrics['precision']:.1%} · "
            f"sensibilidad {positive_metrics['recall']:.1%} · "
            f"F1 {positive_metrics['f1']:.1%}."
        )

errors = result["errors"]
st.subheader("Análisis de errores")
error_columns = st.columns(3)
error_columns[0].metric("Clasificaciones correctas", len(errors["correct"]))
error_columns[1].metric(
    f"Falsos positivos · {result['positive_class']}",
    len(errors["false_positives"]),
)
error_columns[2].metric(
    f"Falsos negativos · {result['positive_class']}",
    len(errors["false_negatives"]),
)
with st.expander("Ver falsos positivos y falsos negativos"):
    st.markdown("**Falsos positivos:**")
    st.dataframe(errors["false_positives"], width="stretch", hide_index=True)
    st.markdown("**Falsos negativos:**")
    st.dataframe(errors["false_negatives"], width="stretch", hide_index=True)
with st.expander("Combinaciones presentes en los errores"):
    st.dataframe(errors["error_combinations"], width="stretch", hide_index=True)

if result["overlapping_predictor_combinations"] > 0 and not grouped:
    st.warning(
        "Hay combinaciones de predictores repetidas entre entrenamiento y prueba. "
        "No se comparten índices ni etiquetas, pero la evaluación aleatoria mide "
        "desempeño también sobre combinaciones ya vistas."
    )
st.caption(
    "Una buena exactitud no garantiza que se identifique la clase minoritaria. "
    "Las asociaciones no son causales y Naive Bayes supone independencia condicional "
    "entre predictores dada la clase."
)

st.divider()
st.subheader("Conclusión para tomar decisiones")
st.markdown(
    f"- **Qué predice:** para una nueva observación, el modelo compara "
    f"probabilidades posteriores por clase y asigna la más alta; la predicción "
    f"individual se consulta en «6 · Predicción».\n"
    f"- **Qué tan bien funciona:** en estas {result['test_rows']:,} filas reservadas "
    f"obtuvo {metrics['accuracy']:.1%} de exactitud "
    f"({metrics['correct']} aciertos y {metrics['incorrect']} errores). "
    "Revisa también la precisión y sensibilidad de cada clase, en particular "
    "si hay clases poco frecuentes."
)
with st.expander("¿Cuándo sería razonable usarlo y cuándo no?"):
    st.markdown(
        "- **Considéralo como apoyo exploratorio** si el conjunto de prueba "
        "representa los casos donde se aplicará, hay ejemplos suficientes por "
        "clase y las métricas por clase son adecuadas para el costo de equivocarse.\n"
        "- **No lo uses como única base de una decisión importante** si hay "
        "clases ausentes o muy escasas, advertencias de datos desconocidos, "
        "rendimiento desigual entre clases, cambios en la población o predictores "
        "fuertemente relacionados.\n"
        "- Naive Bayes supone independencia condicional y sus probabilidades no "
        "garantizan certeza ni calibración. Los umbrales de aceptación dependen "
        "del contexto y del impacto de falsos positivos y falsos negativos."
    )
st.page_link("pages/06_prediccion.py", label="Volver a revisar una predicción")