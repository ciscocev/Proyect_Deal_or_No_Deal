import pandas as pd
import streamlit as st

from src.analysis.frequencies import frequency_table
from src.analysis.profiling import (
    describe_column,
    is_continuous_numeric,
    profile_dataset,
)
from src.analysis.relationships import conditional_proportions, joint_frequencies
from src.utils.helpers import dataframe_is_probably_identifier
from src.utils.state import dataframe_fingerprint, initialize_state
from src.utils.ui import apply_app_styles, show_step_navigation
from src.visualization.plots import distribution_plot, target_relationship_plot

apply_app_styles()
initialize_state(st.session_state)
show_step_navigation(2)
st.title("2 · Exploración del dataset")
st.caption("Entiende la estructura, calidad y distribución de tus datos antes de analizarlos.")

dataframe = st.session_state.get("dataset")
if dataframe is None:
    st.warning("Primero carga o selecciona un dataset en «1 · Carga de datos».")
    st.page_link("pages/01_carga.py", label="Ir a cargar datos")
    st.stop()

st.subheader(f"Dataset activo · {st.session_state.get('dataset_name')}")
profile = profile_dataset(dataframe)
metrics = st.columns(4)
metrics[0].metric("Registros", f"{profile['rows']:,}")
metrics[1].metric("Columnas", profile["columns"])
metrics[2].metric("Valores faltantes", profile["missing_total"])
metrics[3].metric("Filas duplicadas", profile["duplicate_rows"])

columns = list(dataframe.columns)
column_selection_key = (
    f"exploration_columns_{dataframe_fingerprint(dataframe)[:12]}"
)
selected_columns = st.multiselect(
    "Columnas que se utilizarán en la exploración",
    columns,
    default=columns,
    key=column_selection_key,
    help=(
        "Esta selección limita los gráficos, frecuencias y relaciones de la "
        "pestaña Exploración. No elimina ni cambia columnas del dataset."
    ),
)
missing_columns = [
    column for column in dataframe.columns if profile["missing"][column] > 0
]
if missing_columns:
    missing_summary = ", ".join(
        f"{column}: {profile['missing'][column]} "
        f"({profile['missing_proportion'][column]:.1%})"
        for column in missing_columns
    )
    st.warning(
        "Se detectaron valores faltantes: "
        f"{missing_summary}. No se eliminaron ni imputaron; algunas "
        "estadísticas y gráficas numéricas excluyen esos valores."
    )

tab_preview, tab_profile, tab_explore = st.tabs(
    ["Vista previa", "Perfil", "Explorar variables"]
)
with tab_preview:
    st.dataframe(dataframe.head(100), width="stretch", hide_index=True)
    if len(dataframe) > 100:
        st.caption("Se muestran las primeras 100 filas; el dataset completo permanece cargado.")
with tab_profile:
    profile_table = pd.DataFrame(
        [
            {
                "columna": column,
                "tipo": profile["dtypes"][column],
                "observaciones válidas": profile["observed_counts"][column],
                "faltantes": profile["missing"][column],
                "faltantes (%)": profile["missing_proportion"][column] * 100,
                "valores distintos": profile["distinct_values"][column],
                "constante": column in profile["constant_columns"],
                "baja información": column in profile["low_information_columns"],
                "datos insuficientes": column in profile["insufficient_data_columns"],
            }
            for column in dataframe.columns
        ]
    ).set_index("columna")
    st.dataframe(
        profile_table.style.format({"faltantes (%)": "{:.1f}%"}),
        width="stretch",
    )
    st.markdown("**Estadísticas descriptivas**")
    st.dataframe(profile["describe"], width="stretch")
    if profile["constant_columns"]:
        st.warning("Columnas constantes: " + ", ".join(profile["constant_columns"]))
    if profile["low_information_columns"]:
        st.warning(
            "Columnas con al menos 95% de un valor: "
            + ", ".join(profile["low_information_columns"])
        )
    if profile["insufficient_data_columns"]:
        insufficient_summary = ", ".join(
            f"{column} ({profile['observed_counts'][column]} válidos)"
            for column in profile["insufficient_data_columns"]
        )
        st.warning(
            "Columnas con menos de "
            f"{profile['insufficient_data_threshold']} observaciones válidas: "
            f"{insufficient_summary}."
        )
    identifiers = [
        column
        for column in dataframe.columns
        if dataframe_is_probably_identifier(dataframe[column])
    ]
    if identifiers:
        st.warning(
            "Posibles identificadores; revísalos antes de usarlos como predictores: "
            + ", ".join(identifiers)
        )
    if len(dataframe) < 10:
        st.warning("Hay pocos registros; los resúmenes y futuros modelos pueden ser inestables.")
    if profile["duplicate_rows"]:
        st.info(
            "Las filas idénticas se cuentan, pero no se eliminan automáticamente; "
            "pueden representar observaciones independientes."
        )
with tab_explore:
    if not selected_columns:
        st.info("Selecciona al menos una columna para explorar.")
    else:
        selected_column = st.selectbox(
            "Variable para distribución y frecuencias",
            selected_columns,
        )
        description = describe_column(dataframe, selected_column)
        st.caption(
            f"Tipo: {description['dtype']} · "
            f"Valores distintos: {description['distinct_values']:,} · "
            f"Faltantes: {description['missing']:,} "
            f"({description['missing_proportion']:.1%}) · "
            f"Observaciones válidas: {description['observed']:,}"
        )
        if description["distribution_type"] == "histogram":
            st.caption(
                "Al ser una variable numérica con muchos valores distintos, "
                "se muestra un histograma. Los faltantes se informan arriba "
                "y no se incluyen en sus intervalos."
            )
        else:
            st.caption(
                "La distribución se presenta por valor; los valores faltantes "
                "se incluyen en la tabla de frecuencias."
            )
        st.plotly_chart(
            distribution_plot(dataframe, selected_column),
            width="stretch",
        )
        if description["distribution_type"] == "bar":
            st.dataframe(
                frequency_table(dataframe[selected_column]),
                width="stretch",
            )
        if "statistics" in description:
            st.markdown("**Estadísticos descriptivos numéricos**")
            st.dataframe(
                description["statistics"].rename("valor").to_frame(),
                width="stretch",
            )

        st.markdown("#### Relación entre dos variables")
        if len(selected_columns) < 2:
            st.info("Selecciona al menos dos columnas para explorar una relación.")
        else:
            relation_cols = st.columns(2)
            first = relation_cols[0].selectbox(
                "Primera variable", selected_columns, index=0
            )
            second = relation_cols[1].selectbox(
                "Segunda variable",
                selected_columns,
                index=min(1, len(selected_columns) - 1),
            )
            if first != second:
                has_continuous_variable = is_continuous_numeric(
                    dataframe[first]
                ) or is_continuous_numeric(dataframe[second])
                if has_continuous_variable:
                    st.caption(
                        "Como al menos una variable es numérica y continua, "
                        "se muestra una gráfica de dispersión o de caja en vez "
                        "de una tabla de frecuencias con demasiados valores."
                    )
                else:
                    st.dataframe(
                        joint_frequencies(dataframe, first, second),
                        width="stretch",
                    )
                    st.dataframe(
                        conditional_proportions(dataframe, first, second),
                        width="stretch",
                    )
                st.plotly_chart(
                    target_relationship_plot(dataframe, first, second),
                    width="stretch",
                )
            else:
                st.info("Elige dos columnas diferentes para calcular la relación.")