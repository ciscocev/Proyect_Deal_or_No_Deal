"""Gráficas interactivas de exploración con Plotly."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.analysis.profiling import is_continuous_numeric
from src.analysis.relationships import theoretical_high_probability


def _apply_dark_style(figure: go.Figure) -> go.Figure:
    """Aplica el tema compartido al gráfico y devuelve la misma figura."""
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="#090909",
        plot_bgcolor="#1B1B1B",
        font={"color": "#FFFFFF"},
        colorway=["#FFD400", "#79C7FF", "#7DD3A7", "#E58BB7", "#FF9A64"],
        legend={"bgcolor": "rgba(27, 27, 27, 0.9)"},
    )
    figure.update_xaxes(
        gridcolor="#383838",
        linecolor="#777777",
        zerolinecolor="#565656",
    )
    figure.update_yaxes(
        gridcolor="#383838",
        linecolor="#777777",
        zerolinecolor="#565656",
    )
    return figure


def distribution_plot(
    dataframe: pd.DataFrame, column: str, title: str | None = None
) -> go.Figure:
    """Grafica categorías como barras y variables numéricas continuas como histograma."""
    if column not in dataframe.columns:
        raise KeyError(f"No existe la columna: {column}")

    values = dataframe[column]
    missing_count = int(values.isna().sum())
    if is_continuous_numeric(values):
        figure = px.histogram(
            dataframe,
            x=column,
            nbins=30,
            title=title or f"Distribución de {column}",
            labels={column: column, "count": "Frecuencia"},
        )
        figure.update_layout(xaxis_title=column, yaxis_title="Frecuencia")
        if missing_count:
            figure.add_annotation(
                text=f"Valores faltantes no graficados: {missing_count}",
                xref="paper",
                yref="paper",
                x=1,
                y=1.12,
                showarrow=False,
                xanchor="right",
            )
        return _apply_dark_style(figure)

    counts = dataframe[column].value_counts(dropna=False, sort=False).rename_axis(column)
    figure = px.bar(
        counts.reset_index(name="frecuencia"),
        x=column,
        y="frecuencia",
        title=title or f"Distribución de {column}",
        labels={column: column, "frecuencia": "Frecuencia"},
    )
    figure.update_layout(xaxis_title=column, yaxis_title="Frecuencia")
    return _apply_dark_style(figure)


def target_relationship_plot(
    dataframe: pd.DataFrame,
    predictor: str,
    target: str = "premio_final",
) -> go.Figure:
    """Shows an appropriate interactive relationship for numeric or categorical columns."""
    if predictor not in dataframe.columns:
        raise KeyError(f"No existe la columna: {predictor}")
    if target not in dataframe.columns:
        raise KeyError(f"No existe la columna: {target}")

    predictor_continuous = is_continuous_numeric(dataframe[predictor])
    target_continuous = is_continuous_numeric(dataframe[target])
    if predictor_continuous and target_continuous:
        figure = px.scatter(
            dataframe,
            x=predictor,
            y=target,
            title=f"Relación entre {predictor} y {target}",
            labels={predictor: predictor, target: target},
            trendline=None,
        )
        return _apply_dark_style(figure)
    if predictor_continuous or target_continuous:
        category = target if predictor_continuous else predictor
        measurement = predictor if predictor_continuous else target
        figure = px.box(
            dataframe,
            x=category,
            y=measurement,
            points="outliers",
            title=f"Distribución de {measurement} según {category}",
            labels={category: category, measurement: measurement},
        )
        return _apply_dark_style(figure)

    counts = pd.crosstab(dataframe[predictor], dataframe[target], dropna=False)
    long_counts = counts.rename_axis(index=predictor, columns=target).reset_index()
    long_counts = long_counts.melt(
        id_vars=predictor, var_name=target, value_name="frecuencia"
    )
    figure = px.bar(
        long_counts,
        x=predictor,
        y="frecuencia",
        color=target,
        barmode="group",
        title=f"{target} según {predictor}",
        labels={"frecuencia": "Frecuencia"},
    )
    return _apply_dark_style(figure)


def probability_comparison_plot(dataframe: pd.DataFrame) -> go.Figure:
    """Compara probabilidad empírica y teórica por observación."""
    required = {
        "maletines_restantes",
        "altos_revelados",
        "ultimo_premio",
        "premio_final",
    }
    if not required.issubset(dataframe.columns):
        raise ValueError("El DataFrame debe contener las cuatro variables del juego.")
    view = dataframe.copy()
    view["probabilidad_teorica_alto"] = view.apply(
        theoretical_high_probability, axis=1
    )
    view["premio_alto_observado"] = (view["premio_final"] == "Alto").astype(int)
    grouped = (
        view.groupby("probabilidad_teorica_alto", as_index=False, observed=True)
        .agg(
            probabilidad_empirica=("premio_alto_observado", "mean"),
            partidas=("premio_alto_observado", "size"),
        )
        .sort_values("probabilidad_teorica_alto")
    )
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=grouped["probabilidad_teorica_alto"],
            y=grouped["probabilidad_empirica"],
            mode="markers+lines",
            marker={"size": 8},
            text=grouped["partidas"],
            name="Probabilidad empírica",
            hovertemplate=(
                "Teórica: %{x:.3f}<br>Empírica: %{y:.3f}"
                "<br>Partidas: %{text}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Referencia perfecta",
            line={"dash": "dash"},
        )
    )
    figure.update_layout(
        title="Probabilidad teórica y frecuencia observada de premio Alto",
        xaxis_title="Probabilidad teórica",
        yaxis_title="Proporción empírica de premios Altos",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1]},
    )
    return _apply_dark_style(figure)


def game_eda_figures(dataframe: pd.DataFrame) -> dict[str, go.Figure]:
    """Crea las distribuciones requeridas y relaciones de predictores con el objetivo."""
    predictors = ("maletines_restantes", "altos_revelados", "ultimo_premio")
    figures = {
        "premio_final": distribution_plot(dataframe, "premio_final"),
        "maletines_restantes": distribution_plot(dataframe, "maletines_restantes"),
        "altos_revelados": distribution_plot(dataframe, "altos_revelados"),
        "ultimo_premio": distribution_plot(dataframe, "ultimo_premio"),
    }
    figures.update(
        {
            f"{predictor}_vs_premio_final": target_relationship_plot(
                dataframe, predictor
            )
            for predictor in predictors
        }
    )
    figures["probabilidad_teorica_vs_empirica"] = probability_comparison_plot(dataframe)
    return figures