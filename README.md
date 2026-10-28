# Deal or No Deal — Analizador bayesiano

Aplicación local e interactiva para cargar y explorar datos, estudiar
probabilidades empíricas y entrenar un clasificador Naive Bayes explicable.
Los cálculos están en funciones propias de `src/`; la interfaz Streamlit los
consume desde las siete etapas del flujo.

## Instalación

Desde la raíz del proyecto, instala las dependencias declaradas:

```powershell
python -m pip install -r requirements.txt
```

Se conserva **Faker** para la generación autorizada de datos; pandas se utiliza
para cargar y analizar CSV, y Plotly para construir gráficas interactivas.
`requirements.txt` no declara bibliotecas de machine learning. NumPy no se
importa directamente; puede estar instalado como dependencia transitiva de
pandas, pero no se usa como clasificador o calculador probabilístico.

## Ejecución local

Desde la raíz del proyecto:

```powershell
streamlit run app.py
```

Abre la dirección local indicada por Streamlit (por defecto
`http://localhost:8501`). La aplicación carga el dataset demostrativo al
iniciarse si no hay uno activo; también permite cargar un CSV propio. Cambiar
de etapa no reinicia la sesión.

## Estructura del proyecto

```text
app.py                         Entrada y página de inicio
pages/                         Interfaz por etapa (1 a 7)
src/data/                      Carga, validación y generación sintética
src/analysis/                  Perfilado, frecuencias y relaciones
src/probability/               Probabilidades empíricas y suavizado propio
src/model/                     Naive Bayes, predicción y evaluación
src/visualization/             Gráficos interactivos Plotly
src/utils/                     Estado de sesión, validaciones y presentación
data/raw/partidas.csv           Dataset sintético demostrativo
tests/                          Pruebas independientes de Streamlit
INFORME_TECNICO.md              Diseño, método matemático y evaluación
```

Responsabilidades principales:

- `app.py`: configura Streamlit, carga la demostración inicial y enlaza las
  etapas.
- `pages/01_carga.py` a `pages/07_evaluacion.py`: capturan elecciones,
  muestran resultados y llaman a módulos de `src/`; no implementan el
  entrenamiento o las métricas.
- `src/data/`: lectura CSV, validación de filas y generación Faker reproducible.
- `src/analysis/`: descripción de columnas, frecuencias y relaciones observadas.
- `src/probability/`: probabilidades simple, conjunta, condicional, Bayes y
  suavizado de Laplace con validación explícita.
- `src/model/`: estimación categórica Naive Bayes, explicación de predicciones,
  división train/test reproducible y métricas calculadas con código propio.
- `src/visualization/`: creación de figuras Plotly a partir de datos y análisis.
- `src/utils/state.py`: conserva selección y resultados entre páginas e
  invalida resultados obsoletos si cambian los datos o variables.
- `tests/`: ejemplos pequeños calculables a mano y casos límite, ejecutables
  sin iniciar Streamlit.

## Generar partidas sintéticas

```powershell
python -m src.data.generator --games 1000 --seed 42
```

El CSV se guarda como `data/raw/partidas.csv` y contiene exactamente
`maletines_restantes`, `altos_revelados`, `ultimo_premio` y `premio_final`.
El momento de observación se selecciona uniformemente entre los estados con
2, 3, 4 o 5 maletines cerrados. No se sobrescribe un archivo existente; se
puede indicar otra ubicación con `--output`.

## Análisis exploratorio (Etapa 2)

Las funciones reutilizables están en:

- `src/data/loader.py`: carga genérica de CSV y validación del esquema del juego.
- `src/analysis/profiling.py`: tamaño, tipos, conteos y proporciones de
  faltantes, observaciones válidas, duplicados, cardinalidad, variables
  constantes, de baja información o con menos de diez observaciones válidas,
  y estadísticas descriptivas.
- `src/analysis/frequencies.py`: frecuencias, porcentajes y referencia de clase
  mayoritaria.
- `src/analysis/relationships.py`: tablas conjuntas, proporciones condicionales
  y comparación de frecuencias empíricas con probabilidades teóricas.
- `src/visualization/plots.py`: objetos de figura Plotly para las distribuciones,
  relaciones con la variable objetivo y comparación de probabilidades.

Ejemplo de uso desde la raíz:

```powershell
python -c "from src.data.loader import load_csv, validate_game_dataset; from src.analysis.profiling import profile_dataset; from src.analysis.relationships import predictor_relationships; from src.visualization.plots import game_eda_figures; df=load_csv('data/raw/partidas.csv'); validate_game_dataset(df); print(profile_dataset(df)); print(predictor_relationships(df)); figures=game_eda_figures(df); figures['premio_final'].show()"
```

`load_csv` sirve para otros CSV; se puede pasar `expected_columns` cuando se
requiera un esquema concreto. La validación estricta de cuatro columnas se
aplica solo al dataset de partidas. Los duplicados se cuentan, pero nunca se
eliminan automáticamente. Las figuras son objetos Plotly y pueden mostrarse
con `.show()` o guardarse desde Plotly.

## Resultados del EDA de `partidas.csv`

El CSV original contiene 1,000 filas y cuatro columnas, sin valores faltantes.
Los tipos son enteros para `maletines_restantes` y `altos_revelados`, y texto
para `ultimo_premio` y `premio_final`. Hay 951 filas duplicadas en términos de
las cuatro variables (49 combinaciones distintas); se conservaron porque
observaciones iguales pueden corresponder a partidas independientes.

| Variable | Frecuencias observadas |
|---|---|
| `maletines_restantes` | 2: 256 (25.6%); 3: 267 (26.7%); 4: 240 (24.0%); 5: 237 (23.7%) |
| `altos_revelados` | 1: 1 (0.1%); 2: 11 (1.1%); 3: 104 (10.4%); 4: 295 (29.5%); 5: 397 (39.7%); 6: 192 (19.2%) |
| `ultimo_premio` | Alto: 327 (32.7%); Bajo: 673 (67.3%) |
| `premio_final` | Alto: 266 (26.6%); Bajo: 734 (73.4%) |

No hay columnas constantes ni con al menos 95% de sus valores no faltantes
concentrados en una sola categoría. Como referencia inicial, predecir siempre
la clase mayoritaria (`Bajo`) daría 73.4% de exactitud, pero no identificaría
ningún premio Alto. Una exactitud así no demuestra por sí sola un clasificador
útil.

### Relaciones y probabilidades

La proporción observada de `premio_final = Alto` y la media de la probabilidad
teórica `altos_restantes / maletines_restantes` por valor del predictor fueron:

| Condición | Partidas | Alto empírico | Teórica media |
|---|---:|---:|---:|
| Maletines restantes = 2 | 256 | 28.1% | 29.9% |
| Maletines restantes = 3 | 267 | 30.0% | 33.3% |
| Maletines restantes = 4 | 240 | 23.3% | 28.2% |
| Maletines restantes = 5 | 237 | 24.5% | 27.9% |
| Altos revelados = 1 | 1 | 100.0% | 80.0% |
| Altos revelados = 2 | 11 | 81.8% | 77.3% |
| Altos revelados = 3 | 104 | 61.5% | 62.8% |
| Altos revelados = 4 | 295 | 42.4% | 45.3% |
| Altos revelados = 5 | 397 | 16.9% | 22.9% |
| Altos revelados = 6 | 192 | 0.0% | 0.0% |
| Último premio = Alto | 327 | 24.2% | 26.5% |
| Último premio = Bajo | 673 | 27.8% | 31.6% |

El promedio teórico fila a fila fue 29.9%, frente a 26.6% de premios Altos
observados. La diferencia es compatible con variación muestral y la semilla
específica: en una comprobación en memoria de 100 semillas, la media observada
fue 29.9% (rango 26.4%–33.4%); no se modificaron ni regeneraron los datos
guardados. En particular, el grupo con un solo registro no permite una
comparación estable.

Se aprecia una asociación marcada entre más premios altos revelados y una
menor frecuencia de premio Alto inicial. La categoría del último premio,
considerada aisladamente, separa poco las frecuencias en esta muestra.
`maletines_restantes` tampoco muestra aquí una tendencia monotónica clara. Son
asociaciones descriptivas, no relaciones causales. Las probabilidades teóricas
se calculan con `altos_revelados` antes de la apertura más reciente y descuentan
un premio adicional únicamente cuando `ultimo_premio` es Alto.

### Gráficas

`game_eda_figures(df)` produce ocho figuras Plotly: distribución de la clase
objetivo, distribución de cada predictor, tres gráficas de predictores frente a
`premio_final` y una comparación entre probabilidades empíricas y teóricas. Las
frecuencias por predictor permiten observar las asociaciones descritas; la
comparación de probabilidad ayuda a distinguir la expectativa matemática de la
frecuencia finita observada.

## Pruebas

Las pruebas usan `unittest`, son independientes de Streamlit y se ejecutan con:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Las pruebas cubren carga y validación, perfiles, frecuencias, relaciones,
probabilidades, construcción de figuras y el generador.
Incluyen ejemplos pequeños comprobables a mano (por ejemplo, el posterior
0.75/0.25 de una observación categórica) y límites como conteos imposibles,
eventos condicionantes con frecuencia cero, evidencia nula, datos faltantes,
clases ausentes y divisiones reproducibles. Ninguna prueba requiere iniciar el
servidor de la interfaz.

## Demostración guiada

Se puede demostrar el flujo completo usando `data/raw/partidas.csv` (1,000
partidas sintéticas):

1. En **1 · Carga y validación**, carga el archivo o conserva la demostración
   que la aplicación activa al iniciar.
2. En **2 · Exploración**, revisa filas, columnas, valores faltantes,
   frecuencias y distribuciones.
3. En **3 · Selección de variables**, usa `premio_final` como objetivo y
   `maletines_restantes`, `altos_revelados` y `ultimo_premio` como predictores.
4. En **4 · Análisis de probabilidades**, selecciona dos eventos y calcula sus
   probabilidades empíricas.
5. En **5 · Entrenamiento**, deja la partición de prueba en 20%, la semilla en
   42 y Laplace α en 1, y entrena.
6. En **6 · Predicción**, selecciona los valores de una observación, calcula
   probabilidades, revisa las previas y condicionales por clase, y luego asigna
   la clase de mayor probabilidad.
7. En **7 · Evaluación**, examina la matriz de confusión, precisión, recall,
   exactitud y errores en las filas reservadas.

La demostración puede repetirse localmente con `streamlit run app.py`; las
pruebas `AppTest` cubren entrenamiento, predicción y evaluación conservando el
estado de sesión, y las pruebas unitarias verifican resultados numéricos sin
la interfaz.

El informe técnico con las decisiones, fórmulas, resultados y limitaciones está
en [`INFORME_TECNICO.md`](./INFORME_TECNICO.md).

## Probabilidades básicas (Etapa 4)

Los cálculos matemáticos propios están en `src/probability/` y no dependen de
Streamlit ni de bibliotecas de modelos:

- `simple.py`: `simple_probability(event_count, total_count)` calcula
  `P(A) = N(A) / N`.
- `joint.py`: `joint_probability(joint_count, total_count)` calcula
  `P(A ∩ B) = N(A ∩ B) / N`, usando la probabilidad simple para no duplicar
  la división y sus validaciones.
- `conditional.py`: `conditional_probability(joint_count, condition_count)`
  calcula `P(A | B) = N(A ∩ B) / N(B)`.
- `bayes.py`: `bayes_theorem(likelihood, prior, evidence_probability)` calcula
  `P(C | X) = P(X | C) P(C) / P(X)`. `bayes_components(...)` expone los
  componentes, el numerador y el posterior para explicar el resultado.
- `smoothing.py`: `laplace_smoothing(category_count, total_count,
  number_of_categories, alpha=1)` calcula
  `(category_count + alpha) / (total_count + alpha * K)`.

Ejemplo verificable desde la raíz del proyecto:

```python
from src.probability.simple import simple_probability
from src.probability.joint import joint_probability
from src.probability.conditional import conditional_probability
from src.probability.bayes import bayes_components
from src.probability.smoothing import laplace_smoothing

simple_probability(3, 10)             # 0.3
joint_probability(2, 8)                # 0.25
conditional_probability(2, 4)          # 0.5
bayes_components(0.9, 0.01, 0.108)     # posterior ~= 0.08333
laplace_smoothing(0, 10, 2)            # 1/12 ~= 0.08333
```

En el ejemplo de Bayes, `P(X|C)=0.9`, `P(C)=0.01` y `P(X)=0.108`; el
numerador es `0.009` y el posterior es `0.009 / 0.108 = 1/12`. Las entradas
probabilísticas deben ser finitas y estar entre cero y uno, y sus componentes
deben producir un posterior no mayor que uno.

Los métodos basados en frecuencias rechazan conteos negativos, conteos
favorables mayores que el total y conjuntos de observaciones vacíos. Si
`N(B)=0`, la probabilidad condicional no está definida y se lanza `ValueError`
en vez de devolver un valor engañoso. Del mismo modo, Bayes rechaza `P(X)=0`.
Con suavizado de Laplace, una frecuencia cero obtiene masa positiva; `K` debe
representar todas las categorías posibles, incluidas las no observadas, y
`alpha` debe ser positivo.

Las pruebas matemáticas independientes de Streamlit se ejecutan junto con las
pruebas previas mediante:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

## Clasificador Naive Bayes categórico (Etapa 5)

`src/model/naive_bayes.py` implementa `fit_naive_bayes(dataframe, target,
predictors, alpha=1.0, missing_strategy="error")`. Los predictores se deben
seleccionar explícitamente; la función rechaza nombres repetidos, columnas
inexistentes y el uso del objetivo como predictor. No transforma ni modifica el
DataFrame de origen. Las variables discretas numéricas se tratan como
categorías, por lo que cada valor distinto es un nivel categórico.

Para cada clase `C`, el modelo estima la probabilidad previa observada
`P(C) = N(C) / N`. Para un predictor categórico `Xj`, aplica Laplace:

`P(Xj=x | C) = (N(Xj=x, C) + alpha) / (N(C) + alpha * K)`

El vocabulario de cada predictor se aprende exclusivamente del DataFrame que
se pasa a `fit_naive_bayes`. El número `K` incluye las categorías vistas más un
bucket reservado para valores desconocidos. La puntuación de cada clase es
`log(P(C)) + sum(log(P(Xj=xj | C)))`; se normaliza entre clases para producir
probabilidades posteriores que suman uno. La clase con posterior máximo es la
predicción.

`src/model/prediction.py` expone `predict_observation(model, observation)`.
Devuelve la clase elegida, posterior por clase, probabilidades previas y, por
cada clase y predictor, conteo, denominador, probabilidad condicional,
contribución logarítmica y puntuación. Categorías desconocidas reciben la
probabilidad del bucket reservado con suavizado y una advertencia. También se
advierten combinaciones sin evidencia observada, clases con menos de dos
observaciones o entrenamientos muy pequeños. En la interfaz, primero se
presentan las probabilidades, las previas y las contribuciones de cada predictor;
el usuario asigna la clase después de revisar esa evidencia.

Los valores faltantes se rechazan por defecto. Para tratarlos como una
categoría explícita, tanto el ajuste como la predicción deben usar
`missing_strategy="category"`; los objetivos faltantes siempre se rechazan,
porque no existe etiqueta con la que aprender. La variable objetivo no puede
ser predictor. Ejemplo:

```python
import pandas as pd
from src.model.naive_bayes import fit_naive_bayes
from src.model.prediction import predict_observation

games = pd.read_csv("data/raw/partidas.csv")
predictors = ["maletines_restantes", "altos_revelados", "ultimo_premio"]
model = fit_naive_bayes(games, "premio_final", predictors, alpha=1)
result = predict_observation(
    model,
    {
        "maletines_restantes": 3,
        "altos_revelados": 4,
        "ultimo_premio": "Bajo",
    },
)
print(result["predicted_class"])
print(result["class_probabilities"])
print(result["class_details"])
```

En la observación de ejemplo, al ajustar con las 1,000 filas existentes y
`alpha=1`, las probabilidades previas son `P(Alto)=0.266` y `P(Bajo)=0.734`.
El modelo predice `Bajo`: los posteriores normalizados son aproximadamente
`P(Alto|X)=0.4739` y `P(Bajo|X)=0.5261`. Para la clase Alto, las
condicionales de `maletines_restantes=3`, `altos_revelados=4` y
`ultimo_premio=Bajo` son aproximadamente `0.2989`, `0.4615` y `0.6989`;
para Bajo son `0.2544`, `0.2308` y `0.6608`. La explicación devuelta también
incluye los conteos y denominadores de cada cálculo. Este ejemplo ilustra una
predicción; no es una evaluación fuera de muestra.

Es un modelo probabilístico pedagógico, no una afirmación causal ni una
certeza. Naive Bayes supone independencia condicional de los predictores dada
la clase; en el juego, los tres predictores están relacionados, así que ese
supuesto puede fallar y las probabilidades pueden quedar mal calibradas. Las
probabilidades se estiman solo con los datos que recibe el ajuste. Para una evaluación no sesgada, primero se separa entrenamiento y prueba y el
modelo se ajusta únicamente con entrenamiento. La evaluación está implementada
en la Etapa 7.

## Evaluación fuera de muestra (Etapa 7)

`src/model/evaluation.py` implementa una división reproducible de filas,
métricas propias, análisis de errores, clasificador mayoritario y una función
que coordina la evaluación. Ejemplo:

```python
import pandas as pd
from src.model.evaluation import evaluate_naive_bayes

games = pd.read_csv("data/raw/partidas.csv")
result = evaluate_naive_bayes(
    games,
    target="premio_final",
    predictors=["maletines_restantes", "altos_revelados", "ultimo_premio"],
    test_size=0.2,
    seed=42,
    positive_class="Alto",
)
print(result["metrics"])
print(result["baseline_metrics"])
print(result["errors"]["false_negatives"])
```

La partición aleatoria predeterminada es estratificada por clase siempre que
sea posible, mantiene las posiciones originales para auditar el aislamiento,
y no elimina observaciones duplicadas. Los vocabularios y parámetros de Naive
Bayes se ajustan exclusivamente en las filas de entrenamiento; el mismo
conjunto de prueba se usa para el modelo y la referencia mayoritaria. La clase
mayoritaria se obtiene solo de las etiquetas de entrenamiento. Si una clase
tiene una sola observación, esta permanece en entrenamiento y se emite una
advertencia.

Métricas reportadas:

- Exactitud: `(TP + TN) / total` (generalizada como aciertos / total).
- Precisión por clase: `TP / (TP + FP)`.
- Sensibilidad/recall: `TP / (TP + FN)`.
- F1: `2 * precision * recall / (precision + recall)`.

La matriz de confusión usa filas para clase real y columnas para predicción.
Cuando una división de precisión o recall tiene denominador cero, se informa
`0.0` y se marca en `undefined`, evitando errores o valores indefinidos ocultos.
`analyze_errors` entrega aciertos, errores, falsos positivos y falsos negativos
para la clase positiva, además de combinaciones de variables presentes en los
errores.

`split_by_predictor_groups` ofrece una evaluación complementaria que retiene
combinaciones completas de predictores para la prueba. Esta evita que una
combinación vista se repita entre particiones, pero no garantiza el mismo
balance de clases ni un tamaño exacto de prueba; mide generalización a
combinaciones no vistas y responde una pregunta distinta a la separación
aleatoria por filas.

### Evaluación de las 1,000 partidas (semilla 42)

En la partición estratificada aleatoria quedaron 800 filas para entrenamiento
y 200 para prueba. No hubo índices de origen compartidos, aunque 30
combinaciones de predictores aparecieron en ambas particiones; las partidas
duplicadas no fueron eliminadas. La evaluación agrupada retuvo 6 combinaciones
completas, con 786 filas de entrenamiento y 214 de prueba, sin combinaciones
predictoras compartidas.

Matriz de confusión aleatoria (filas reales, columnas predichas):

| Real \ Predicha | Alto | Bajo |
|---|---:|---:|
| Alto | 14 | 39 |
| Bajo | 9 | 138 |

| Métrica | Naive Bayes | Mayoritaria (Bajo) |
|---|---:|---:|
| Exactitud | 76.0% | 73.5% |
| Alto — precisión | 60.9% | 0.0% |
| Alto — sensibilidad | 26.4% | 0.0% |
| Alto — F1 | 36.8% | 0.0% |
| Bajo — precisión | 78.0% | 73.5% |
| Bajo — sensibilidad | 93.9% | 100.0% |
| Bajo — F1 | 85.2% | 84.7% |

Naive Bayes acertó 152/200 y falló 48. Para Alto tuvo 9 falsos positivos y 39
falsos negativos: la exactitud supera modestamente la referencia, pero la
sensibilidad de Alto sigue baja. Varios falsos negativos ocurrieron en filas
con `ultimo_premio=Bajo`, por ejemplo combinaciones `(maletines_restantes=2,
altos_revelados=5)` y `(3, 4)`. El desbalance hace que la exactitud favorezca
la clase Bajo y oculte esa limitación.

En la prueba agrupada, Naive Bayes obtuvo 77.6% de exactitud frente a 79.0%
para la referencia mayoritaria. Su precisión/recall/F1 de Alto fueron
42.1%/17.8%/25.0%; la referencia, al predecir siempre Bajo, obtuvo cero en
esas métricas. Este resultado muestra que su desempeño depende del esquema de
partición y que la generalización a combinaciones no vistas requiere cautela.

Las 951 filas duplicadas (por el conjunto de cuatro variables) representan
partidas que pueden ser independientes. En una partición aleatoria pueden
quedar instancias con predictores idénticos a ambos lados; eso no comparte
índices ni etiqueta por construcción, pero evalúa casos de combinaciones
conocidas y puede dar una perspectiva más optimista para combinaciones raras.
La separación agrupada reduce ese solapamiento, aunque las combinaciones
retenidas pueden tener distribuciones distintas. Además, los predictores del
juego están relacionados, lo que limita el supuesto de independencia
condicional de Naive Bayes. Ni las probabilidades ni las métricas son certezas
ni evidencia causal.

La suite de pruebas independiente de la interfaz se ejecuta con:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

## Interfaz Streamlit y flujo de siete etapas

Inicia desde la raíz del proyecto:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

La página de inicio enlaza directamente con las siete etapas. Cada página indica
el progreso, ofrece accesos a la etapa anterior y siguiente y permite saltar a
cualquier otra sin reiniciar la sesión. Las páginas cubren:

1. **Carga y validación** carga el CSV demo o un archivo externo, revisa que
   tenga contenido, encabezados únicos y registros; conserva el dataset previo
   si el archivo nuevo no es válido.
2. **Exploración** muestra vista previa, dimensiones, tipos, cantidad y
   proporción de faltantes por columna, duplicados, cardinalidad, estadísticas
   y advertencias; permite elegir columnas y consultar distribuciones y
   relaciones con Plotly. Las categorías y valores numéricos discretos se
   muestran como frecuencias; los numéricos con más de 20 valores distintos
   usan histogramas. Las relaciones con variables continuas usan gráficas de
   caja o dispersión.
3. **Variables** permite elegir objetivo y predictores manualmente, muestra
   frecuencias y relaciones observadas con el objetivo para apoyar la selección,
   y permite guardar justificaciones de inclusión/exclusión.
4. **Probabilidades** define dos eventos y muestra conteos, probabilidades
   simples, conjuntas, condicionales y Bayes con sustitución numérica.
5. **Entrenamiento** separa entrenamiento y reserva de prueba, configura α y la
   política de faltantes, y entrena Naive Bayes.
6. **Predicción** permite introducir una observación; primero presenta las
   probabilidades posteriores, previas y evidencia condicional por predictor,
   luego permite asignar la clase con la probabilidad más alta.
7. **Evaluación** calcula métricas fuera de muestra, matriz de confusión,
   comparación mayoritaria, falsos positivos/negativos y, opcionalmente,
   evaluación agrupada por combinaciones predictoras; incluye orientación
   interpretativa sobre el uso responsable y las limitaciones.

El estado de sesión conserva dataset, selección, modelo, predicción y
evaluación. Cambiar el contenido del dataset o las variables seleccionadas
invalida el modelo y resultados dependientes. Los cambios de parámetros de
ajuste también obligan a volver a entrenar. Los archivos subidos se leen en
memoria; el CSV original no se modifica. La evaluación ajusta exclusivamente
con entrenamiento y compara con la referencia en el mismo conjunto de prueba.
La identidad visual usa CSS y tipografías del sistema; no requiere imágenes ni
recursos multimedia externos. El tema global está en
`.streamlit/config.toml`: fondo negro `#090909`, superficies `#1B1B1B`, texto
blanco y acento amarillo `#FFD400`. Los estilos
compartidos de `src/utils/ui.py` mantienen controles y tarjetas coherentes en
todas las páginas; los gráficos Plotly usan fondos oscuros compatibles y las
tarjetas de inicio se reorganizan según el ancho disponible.

`app.py` es el punto de entrada; las páginas se limitan a la interfaz y llaman
a funciones de `src/`. Los cálculos de eventos empíricos se centralizan en
`src/probability/empirical.py`, que reutiliza las operaciones de probabilidad
simple, conjunta, condicional y de Bayes; el análisis descriptivo y Naive Bayes
también reutilizan las funciones comunes de probabilidad y suavizado. Los
módulos de análisis, datos, modelo, probabilidad y visualización mantienen
responsabilidades separadas y no importan páginas ni Streamlit, por lo que las
pruebas pueden ejecutarse independientemente de la interfaz. El primer uso de
la aplicación carga automáticamente las partidas de demostración si no hay
otro dataset activo.
