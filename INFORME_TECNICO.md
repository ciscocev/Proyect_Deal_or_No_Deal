# Informe técnico — Deal or No Deal

## Propósito y decisiones de diseño

La aplicación convierte un CSV en evidencia que puede inspeccionarse antes de
una predicción. `app.py` es el punto de entrada; las siete páginas de
`pages/` son la capa de interfaz; los cálculos se concentran en módulos
independientes de `src/`. Las responsabilidades se separan entre carga y
generación de datos, análisis descriptivo, probabilidades, modelo, gráficas y
estado de sesión.

El dataset original no se modifica durante el análisis o entrenamiento. El
estado conserva datos, selección de objetivo/predictores, modelo y resultados
entre etapas; un cambio de dataset o de variables invalida resultados que ya
no correspondan. Los controles permiten volver a las etapas anteriores sin
reiniciar el flujo.

Las dependencias declaradas son Streamlit, pandas, Plotly y Faker. Se usa
Faker solo en la generación sintética. No hay bibliotecas de machine learning
ni funciones externas de clasificación o evaluación.

## Funcionamiento matemático

Las funciones propias implementan:

- Probabilidad simple: `P(A) = N(A) / N`.
- Probabilidad conjunta: `P(A ∩ B) = N(A ∩ B) / N`.
- Probabilidad condicional: `P(A | B) = N(A ∩ B) / N(B)`; si `N(B)=0`, el
  resultado no está definido y se informa explícitamente.
- Teorema de Bayes: `P(C | X) = P(X | C) P(C) / P(X)`.
- Suavizado categórico de Laplace:
  `P(Xj=x | C) = (N(Xj=x,C)+α) / (N(C)+αK)`.

Para varios predictores, Naive Bayes usa explícitamente la hipótesis de
independencia condicional:

`P(C | X1,...,Xk) ∝ P(C) × ∏ P(Xj | C)`.

Los pesos se acumulan en escala logarítmica para evitar subdesbordamiento y se
normalizan entre las clases. La interfaz expone priors, condicionales y
contribuciones de cada predictor antes de permitir asignar la clase de mayor
posterior. Faltantes y valores desconocidos siguen una estrategia seleccionada
por el usuario y se acompañan de advertencias. Las probabilidades son
estimaciones; no son certeza ni garantía de calibración. La asociación
observada tampoco establece causalidad.

## Evaluación reproducible

Se evaluó `data/raw/partidas.csv` con objetivo `premio_final`, predictores
`maletines_restantes`, `altos_revelados`, `ultimo_premio`, tamaño de prueba
20%, semilla 42, α=1 y división estratificada por clase. El particionador
separó **800 filas para entrenamiento y 200 para prueba**. El vocabulario y
las probabilidades del modelo se aprendieron solo con entrenamiento; la
referencia mayoritaria también se calculó desde sus etiquetas. No hubo
posiciones de fila compartidas.

Matriz de confusión de Naive Bayes (filas reales, columnas predichas):

| Real \ Predicha | Alto | Bajo |
|---|---:|---:|
| Alto | 14 | 39 |
| Bajo | 9 | 138 |

| Métrica en prueba | Naive Bayes | Referencia mayoritaria |
|---|---:|---:|
| Exactitud | 76.0% | 73.5% |
| Alto — precisión | 60.9% | 0.0% |
| Alto — sensibilidad (recall) | 26.4% | 0.0% |
| Bajo — precisión | 78.0% | 73.5% |
| Bajo — sensibilidad (recall) | 93.9% | 100.0% |
| Aciertos / errores | 152 / 48 | 147 / 53 |

La exactitud mejora 2.5 puntos porcentuales respecto a la referencia, pero la
sensibilidad para la clase `Alto` es baja: el modelo detectó 14 de 53 casos
`Alto` en prueba. Por ello, no se debe concluir que el rendimiento es adecuado
solo con la exactitud. La partición agrupada por combinaciones predictoras es
una evaluación complementaria con una pregunta de generalización distinta.

## Verificación y demostración

Las pruebas se ejecutan sin iniciar Streamlit:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Incluyen pruebas matemáticas, datos pequeños verificables manualmente, límites,
suavizado, predicción, separación train/test, métricas y no modificación de
datos originales. Para una demostración interactiva:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

Luego sigue las siete etapas descritas en la sección **Demostración guiada**
del `README.md`. La evaluación de este informe se reprodujo ejecutando
`evaluate_naive_bayes` directamente sobre el CSV con los parámetros indicados.
