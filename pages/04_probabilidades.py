import streamlit as st

from src.probability.empirical import calculate_event_probabilities
from src.utils.state import initialize_state
from src.utils.ui import apply_app_styles, show_step_navigation

apply_app_styles()
initialize_state(st.session_state)
show_step_navigation(4)
st.title("4 · Análisis de probabilidades")
st.caption("Define dos eventos categóricos y revisa conteos, fórmulas y resultados empíricos.")

dataframe = st.session_state.get("dataset")
if dataframe is None:
    st.warning("Primero carga un dataset en «1 · Carga de datos».")
    st.page_link("pages/01_carga.py", label="Ir a cargar datos")
    st.stop()


def event_options(column: str) -> list[tuple[str, object]]:
    """Construye opciones de valores únicos y de faltantes para una columna."""
    values = [("value", value) for value in dataframe[column].drop_duplicates().tolist()]
    if dataframe[column].isna().any():
        values.append(("missing", None))
    return values


columns = list(dataframe.columns)
left, right = st.columns(2)
with left:
    st.subheader("Evento A")
    column_a = st.selectbox("Variable de A", columns, key="prob_a_column")
    options_a = event_options(column_a)
    index_a = st.selectbox(
        "Valor de A",
        list(range(len(options_a))),
        key=f"prob_a_value_{column_a}",
        format_func=lambda index: (
            "Faltante" if options_a[index][0] == "missing" else str(options_a[index][1])
        ),
    )
    value_a = options_a[index_a]
with right:
    st.subheader("Evento B")
    column_b = st.selectbox("Variable de B", columns, key="prob_b_column")
    options_b = event_options(column_b)
    index_b = st.selectbox(
        "Valor de B",
        list(range(len(options_b))),
        key=f"prob_b_value_{column_b}",
        format_func=lambda index: (
            "Faltante" if options_b[index][0] == "missing" else str(options_b[index][1])
        ),
    )
    value_b = options_b[index_b]

st.markdown(
    "Se estiman **P(A)**, **P(B)**, **P(A ∩ B)**, **P(A | B)** y, mediante "
    "Bayes, **P(B | A)**. Si un evento condicionante tiene cero casos, la "
    "condicional correspondiente se informa como no definida."
)
if st.button("Calcular probabilidades", type="primary"):
    try:
        st.session_state["probability_result"] = calculate_event_probabilities(
            dataframe, column_a, value_a, column_b, value_b
        )
    except ValueError as error:
        st.session_state["probability_result"] = None
        st.error(f"No se pudo calcular: {error}")

result = st.session_state.get("probability_result")
if result is not None:
    metric_items = (
        ("P(A)", "P(A)"),
        ("P(B)", "P(B)"),
        ("P(A ∩ B)", "P(A intersección B)"),
        ("P(A | B)", "P(A dado B)"),
        ("P(B | A)", "P(B dado A)"),
    )
    for start in (0, 3):
        row_items = metric_items[start : start + (3 if start == 0 else 2)]
        for column, (metric_name, key) in zip(st.columns(len(row_items)), row_items):
            column.metric(metric_name, f"{result[key]:.4f}")

    st.markdown("#### Sustitución numérica")
    st.latex(
        rf"P(A) = \frac{{{result['conteo_A']}}}{{{result['total']}}} "
        rf"= {result['P(A)']:.4f}"
    )
    st.latex(
        rf"P(A \cap B) = \frac{{{result['conteo_conjunto']}}}{{{result['total']}}} "
        rf"= {result['P(A intersección B)']:.4f}"
    )
    st.latex(
        rf"P(A \mid B) = \frac{{{result['conteo_conjunto']}}}{{{result['conteo_B']}}} "
        rf"= {result['P(A dado B)']:.4f}"
    )
    st.latex(
        rf"P(B \mid A) = \frac{{P(A \mid B)P(B)}}{{P(A)}} "
        rf"= \frac{{{result['P(A dado B)']:.4f}\times{result['P(B)']:.4f}}}"
        rf"{{{result['P(A)']:.4f}}} = {result['Bayes P(B dado A)']:.4f}"
    )