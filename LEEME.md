# Clasificación de Sentimiento en Reseñas de Películas

> English version: [README.md](README.md)

Proyecto de Procesamiento de Lenguaje Natural (NLP) que clasifica **reseñas de películas de IMDB como positivas o negativas** usando solo su texto.

El proyecto compara tres estrategias de preprocesamiento del texto (limpieza básica, lematización con NLTK y lematización con spaCy), dos representaciones **TF-IDF** (unigramas y unigramas + bigramas) y cuatro modelos. El modelo elegido, **LinearSVC con TF-IDF de unigramas + bigramas**, obtuvo un **F1 de 0.9139 en validación** y **F1 de 0.9043 con ROC-AUC de 0.9663 en el conjunto de prueba independiente** (23,535 reseñas), muy por encima del requisito de **F1 ≥ 0.85**.

---

# Contexto de Negocio

Film Junky Union, una comunidad de aficionados al cine clásico, está creando un sistema para filtrar y clasificar reseñas de películas. El objetivo es **detectar automáticamente las reseñas negativas**, para que la comunidad no tenga que leer y etiquetar miles de textos a mano. Un modelo confiable ayuda a:

* Clasificar las reseñas nuevas en cuanto se publican.
* Resumir la opinión general sobre una película a partir de muchas reseñas.
* Liberar a los moderadores del etiquetado manual.

El modelo debe predecir una variable objetivo:

| Objetivo | Descripción |
| --- | --- |
| `pos` | 1 si la reseña es positiva, 0 si es negativa |

---

# Conceptos de NLP

| Término | Significado |
| --- | --- |
| Corpus | La colección completa de textos (aquí, las reseñas) |
| Token | Una unidad de texto, normalmente una palabra |
| Lematización | Reducir una palabra a su forma de diccionario (`movies` → `movie`, `were` → `be`) |
| TF-IDF | Frecuencia del término × frecuencia inversa en documentos: una palabra pesa más si es frecuente en una reseña pero rara en el resto del corpus |
| N-grama | Una secuencia de *n* palabras consecutivas. Los unigramas son palabras sueltas; los bigramas son pares como `not good` |
| Matriz dispersa | Una matriz donde la mayoría de los valores son cero. Cada reseña usa solo unos cientos de los cientos de miles de términos posibles |
| `sublinear_tf` | Usa `1 + log(tf)` en lugar del conteo directo, para que repetir una palabra muchas veces no domine la reseña |
| `min_df` / `max_df` | Ignoran las palabras que aparecen en menos de 2 reseñas o en más del 95% de ellas |

## Representación del Texto

```text
"This movie wasn't very good!!!"
        │  limpieza básica
        ▼
"this movie wasn't very good"
        │  TF-IDF (unigramas + bigramas)
        ▼
[ this, movie, wasn't, very, good, this movie, movie wasn't, wasn't very, very good ]
        │
        ▼
Vector disperso con ~417,000 características posibles → clasificador lineal
```

---

# Métrica de Evaluación

La métrica principal es el **F1**, la media armónica de Precision y Recall:

$$
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

El F1 solo es alto cuando el modelo **encuentra la mayoría de las reseñas de una clase** (Recall) y **acierta cuando las marca** (Precision). Además, deja en evidencia a los modelos triviales: un `DummyClassifier` que siempre predice la clase mayoritaria obtiene cerca de 50% de Accuracy, pero **F1 = 0**.

**ROC-AUC** y **Accuracy** se registraron como métricas secundarias. En LinearSVC, el ROC-AUC se calcula con los puntajes de `decision_function`, porque el modelo no entrega probabilidades.

El requisito del proyecto era **F1 ≥ 0.85** en el conjunto de prueba.

---

# Dataset

| Archivo | Filas | Columnas | Descripción |
| --- | ---: | ---: | --- |
| `data/imdb_reviews.tsv` | 47,331 | 17 | Texto de la reseña, etiqueta de sentimiento, metadatos de la película y la partición original train/test |

Columnas principales:

| Columna | Descripción |
| --- | --- |
| `review` | Texto de la reseña (la única variable que usan los modelos) |
| `pos` | Objetivo: 0 = negativa, 1 = positiva |
| `ds_part` | Partición original: `train` o `test` |

* Las clases están balanceadas: **23,715 reseñas negativas (50.1%)** y **23,616 positivas (49.9%)**.
* Solo `average_rating` y `votes` tienen valores faltantes (2 cada una); ninguna se usa en los modelos.

**Fuente de los datos.** El dataset fue proporcionado por **TripleTen** como parte de su Bootcamp de Data Science. Está basado en el dataset de reseñas de IMDB publicado por Maas et al. (2011), *Learning Word Vectors for Sentiment Analysis*, y se incluye en este repositorio con fines educativos y de reproducibilidad.

---

# Flujo del Proyecto

```text
Reseñas de IMDB (47,331)
            │
            ▼
1. Carga de datos y revisión de calidad
            │
            ▼
2. Análisis exploratorio: balance de clases y longitud de las reseñas
            │
            ▼
3. División: test original aparte; train original → 80% train / 20% validación
            │
            ▼
4. Preprocesamiento del texto
   ├── Limpieza básica (minúsculas, solo letras y apóstrofos)
   ├── + Lematización con NLTK
   └── + Lematización con spaCy
            │
            ▼
5. Vectorización TF-IDF (unigramas y después unigramas + bigramas)
            │
            ▼
6. Modelado en validación
   ├── DummyClassifier (línea base)
   ├── Regresión Logística × 3 estrategias de preprocesamiento
   ├── LinearSVC
   ├── Regresión Logística y LinearSVC con bigramas
   └── LightGBM con bigramas
            │
            ▼
7. Reentrenar el mejor modelo con todo el train original y evaluarlo una vez en test
            │
            ▼
8. Clasificar 14 reseñas escritas a mano
```

---

# 1. Preparación de los Datos

* El archivo TSV se cargó con `votes` como entero que admite nulos (`Int64`).
* No se eliminaron filas ni se imputaron valores: los modelos solo usan `review` → `pos`, y ambas columnas están completas.
* Se respetó la partición original de `ds_part`. El conjunto de prueba (**23,535 reseñas**) se mantuvo aparte hasta la evaluación final.
* El train original (**23,796 reseñas**) se dividió en train interno y validación con `stratify`:

```text
Dataset completo (47,331)
│
├── ds_part = train (23,796)
│       ├── Train interno ──→ 19,036 (80%) → entrenamiento de los modelos
│       └── Validación ─────→  4,760 (20%) → selección de modelo e hiperparámetros
│
└── ds_part = test (23,535) ──→ evaluación final (se usa una sola vez)
```

Los tres conjuntos conservan la misma proporción de clases (cerca de 50.1% negativas / 49.9% positivas).

---

# 2. Análisis Exploratorio

## Balance de Clases

| Partición | Negativas | Positivas |
| --- | ---: | ---: |
| Train | 11,912 (50.06%) | 11,884 (49.94%) |
| Test | 11,803 (50.15%) | 11,732 (49.85%) |

La diferencia entre clases es menor al 1%, así que no fue necesario aplicar técnicas de balanceo (oversampling, undersampling o pesos de clase).

## Longitud de las Reseñas

| Clase | Media de palabras | Mediana de palabras | Máximo de palabras |
| --- | ---: | ---: | ---: |
| Negativa | 227.5 | 173 | 1,491 |
| Positiva | 231.3 | 171 | 2,459 |

* Las reseñas son largas: una mediana de **172 palabras**, y el 50% central está entre unas 126 y 278 palabras.
* Las distribuciones de longitud de ambas clases se superponen casi por completo, así que **la longitud no sirve para distinguirlas**; el sentimiento se tiene que aprender de las palabras.

---

# 3. Preprocesamiento y Vectorización del Texto

## Estrategias de Preprocesamiento

| Estrategia | Pasos | Características TF-IDF (unigramas) |
| --- | --- | ---: |
| Limpieza básica | Minúsculas, solo letras `a-z` y apóstrofos, espacios colapsados | 39,531 |
| Lematización con NLTK | Limpieza básica + `word_tokenize` + `WordNetLemmatizer` | 35,027 |
| Lematización con spaCy | Limpieza básica + lemas de `en_core_web_sm` (parser y NER desactivados) | — |

Se conservaron los apóstrofos para que negaciones como `wasn't` o `didn't` sigan siendo un solo token.

Ejemplo de lematización:

```text
Original: The movies were amazing and the actors were running beautifully.
NLTK:     The movie were amazing and the actor were running beautifully .
spaCy:    the movie be amazing and the actor be run beautifully .
```

El `WordNetLemmatizer` de NLTK trata cada palabra como sustantivo si no se le indica la categoría gramatical, por eso solo cambió los plurales. spaCy también lematizó los verbos, pero tardó cerca de **3.5 minutos** en procesar el conjunto de entrenamiento.

## Configuración de TF-IDF

```python
TfidfVectorizer(ngram_range=(1, 1),   # o (1, 2) para bigramas
                min_df=2, max_df=0.95, sublinear_tf=True)
```

El vectorizador siempre se **ajustó solo con los textos de entrenamiento** y después se aplicó a validación y test. Agregar bigramas aumentó el espacio de características de 39,531 a **350,997** columnas en el train interno (416,551 con todo el train).

---

# 4. Modelado

Cada modelo se probó con una pequeña rejilla de hiperparámetros (`C` en los modelos lineales; árboles, tasa de aprendizaje y hojas en LightGBM), y se conservó la mejor configuración según el F1 de validación.

## Experimento 1: Limpieza Básica vs Lematización (Regresión Logística, unigramas)

| Preprocesamiento | Mejor `C` | F1 | ROC-AUC |
| --- | ---: | ---: | ---: |
| **Limpieza básica** | 2 | **0.8989** | 0.9606 |
| Lematización con NLTK | 2 | 0.8973 | 0.9595 |
| Lematización con spaCy | 5 | 0.8932 | 0.9578 |

La lematización redujo el vocabulario, pero no mejoró los resultados, y spaCy fue mucho más lento. **Se conservó la limpieza básica** para el resto del proyecto.

## Experimento 2: Regresión Logística vs LinearSVC (unigramas)

| Modelo | Mejor `C` | F1 | Accuracy | ROC-AUC | Tiempo de entrenamiento |
| --- | ---: | ---: | ---: | ---: | ---: |
| Regresión Logística | 2 | **0.8989** | 0.8979 | 0.9606 | 1.78 s |
| LinearSVC | 0.5 | 0.8988 | **0.8983** | **0.9614** | **0.21 s** |

Los dos modelos lineales dieron casi el mismo resultado, pero LinearSVC entrenó unas **8 veces más rápido**.

## Experimento 3: Impacto de los Bigramas

| Modelo | Unigramas | Unigramas + Bigramas | Mejora |
| --- | ---: | ---: | ---: |
| Regresión Logística | 0.8989 | 0.9076 | +0.0086 |
| LinearSVC | 0.8988 | **0.9139** | **+0.0150** |

Los bigramas dieron **la mayor mejora del proyecto**, más que cualquier cambio de preprocesamiento. Permiten que el modelo capture expresiones cortas como `not good`, `not bad` o `waste of`, donde una palabra sola da la señal equivocada.

## Experimento 4: LightGBM (unigramas + bigramas)

| Configuración | F1 | ROC-AUC | Tiempo de entrenamiento |
| --- | ---: | ---: | ---: |
| 200 árboles, `learning_rate=0.1`, 31 hojas | **0.8785** | 0.9506 | 93 s |
| 300 árboles, `learning_rate=0.05`, 31 hojas | 0.8763 | 0.9473 | 141 s |
| 200 árboles, `learning_rate=0.05`, 15 hojas | 0.8530 | 0.9306 | 58 s |

LightGBM cumplió el requisito, pero fue **menos preciso y unas 200 veces más lento** que LinearSVC. Con cientos de miles de características dispersas, una frontera lineal se ajusta mejor al problema que los árboles.

## Resumen de Validación

| Modelo | F1 | Accuracy | ROC-AUC | Tiempo de entrenamiento |
| --- | ---: | ---: | ---: | ---: |
| **LinearSVC + Bigramas** (`C=0.5`) | **0.9139** | **0.9137** | **0.9677** | 0.47 s |
| Regresión Logística + Bigramas (`C=5`) | 0.9076 | 0.9074 | 0.9662 | 4.50 s |
| Regresión Logística (`C=2`) | 0.8989 | 0.8979 | 0.9606 | 1.78 s |
| LinearSVC (`C=0.5`) | 0.8988 | 0.8983 | 0.9614 | 0.21 s |
| Regresión Logística + NLTK (`C=2`) | 0.8973 | 0.8966 | 0.9595 | 1.51 s |
| Regresión Logística + spaCy (`C=5`) | 0.8932 | 0.8924 | 0.9578 | 2.48 s |
| LightGBM + Bigramas | 0.8785 | 0.8782 | 0.9506 | 92.95 s |
| DummyClassifier (más frecuente) | 0.0000 | 0.5006 | 0.5000 | 0.001 s |

## Resultado Final

El vectorizador TF-IDF y el modelo se volvieron a ajustar con **todo el train original (23,796 reseñas)** y se evaluaron **una sola vez** en el conjunto de prueba:

```text
Modelo:          LinearSVC (C=0.5, max_iter=5000) + TF-IDF unigramas + bigramas
F1 validación:   0.9139
F1 test:         0.9043
Accuracy test:   0.9046
ROC-AUC test:    0.9663
Requisito:       F1 ≥ 0.85   ✓
```

| Clase | Precision | Recall | F1 | Soporte |
| --- | ---: | ---: | ---: | ---: |
| Negativa | 0.91 | 0.90 | 0.90 | 11,803 |
| Positiva | 0.90 | 0.90 | 0.90 | 11,732 |

El F1 bajó cerca de 0.01 de validación a test, algo esperable en datos que no participaron en la selección del modelo. El desempeño está **equilibrado entre ambas clases**.

## Reseñas Escritas a Mano

Se usó la mejor configuración de cada modelo para clasificar 14 reseñas nuevas escritas a mano:

* Todos los modelos lineales (Regresión Logística y LinearSVC, con o sin lematización o bigramas) dieron **la misma etiqueta a las 14 reseñas**.
* Las reseñas claras, como *"One of the worst movies I have ever seen…"* o *"I loved this film…"*, fueron bien clasificadas por todos los modelos.
* *"I didn't expect the reboot to be so good!"* fue clasificada como **negativa** por los modelos lineales, probablemente porque palabras como `didn't` son frecuentes en reseñas negativas. LightGBM la clasificó bien, pero también marcó como positivas dos reseñas claramente negativas (*"I did not simply like it…"* y *"Well, I was bored and felt asleep…"*).

---

# Conclusiones

1. **El dataset estaba balanceado**, así que no hizo falta ninguna técnica de balanceo y el F1 fue una métrica justa para ambas clases.

2. **La limpieza básica funcionó mejor que la lematización.** Ni NLTK ni spaCy mejoraron el F1, y spaCy agregó varios minutos de procesamiento.

3. **La Regresión Logística y LinearSVC tuvieron un desempeño casi idéntico** con unigramas, pero LinearSVC fue unas 8 veces más rápido.

4. **Los bigramas fueron la mejora más importante** (+0.015 de F1 en LinearSVC). Importó más cómo se representa el texto que cómo se normaliza.

5. **LightGBM fue menos preciso y mucho más lento** con estos datos dispersos de alta dimensión.

6. **El modelo final supera el requisito**: F1 de 0.9043 y ROC-AUC de 0.9663 en 23,535 reseñas nuevas, con resultados equilibrados en ambas clases.

## Interpretación de Negocio

El modelo clasifica bien cerca de 9 de cada 10 reseñas y se entrena en menos de un segundo, así que puede reentrenarse seguido y usarse para etiquetar automáticamente las reseñas nuevas. Las reseñas con sarcasmo, contraste (*"not bad, but…"*) o elogios inesperados son los errores más probables, así que los casos dudosos podrían enviarse a un moderador.

## Siguientes Pasos

* Usar el puntaje de `decision_function` de LinearSVC para enviar a revisión manual las reseñas con poca confianza.
* Revisar los coeficientes positivos y negativos más grandes para explicar qué palabras y bigramas determinan cada predicción.
* Probar trigramas o n-gramas de caracteres para capturar negaciones más largas.
* Comparar con un modelo transformer preentrenado (por ejemplo, BERT) para medir la mejora frente a su mayor costo computacional.

---

# Estructura del Proyecto

```text
movie-review-sentiment-classification/
│
├── data/
│   └── imdb_reviews.tsv
│
├── notebooks/
│   └── movie_reviews_classification.ipynb
│
├── src/
│   └── funciones_personales.py
│
├── .gitignore
├── environment.yml
├── requirements.txt
├── LEEME.md
└── README.md
```

---

# Instalación

Clona el repositorio:

```bash
git clone https://github.com/pablo-vc87/movie-review-sentiment-classification.git

cd movie-review-sentiment-classification
```

## Opción 1: Conda

```bash
conda env create -f environment.yml
conda activate nlp_movie_reviews
```

## Opción 2: pip

Crea y activa un entorno virtual y después:

```bash
pip install -r requirements.txt
```

## Recursos de Lenguaje

Los experimentos de lematización necesitan el modelo de inglés de spaCy y los datos de NLTK:

```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet')"
```

Después inicia Jupyter, abre `notebooks/movie_reviews_classification.ipynb` y ejecuta todas las celdas:

```bash
jupyter notebook
```

---

# Tecnologías

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* scikit-learn (TF-IDF, Regresión Logística, LinearSVC, DummyClassifier)
* LightGBM
* NLTK
* spaCy
* tqdm
* Jupyter Notebook

---

# Habilidades Demostradas

* Procesamiento de Lenguaje Natural
* Limpieza y normalización de texto
* Lematización con NLTK y spaCy
* Vectorización TF-IDF con unigramas y bigramas
* Clasificación de sentimiento
* Comparación de modelos: Regresión Logística, LinearSVC, LightGBM y una línea base
* Ajuste de hiperparámetros
* Metodología train / validación / test sin fuga de datos
* Evaluación con F1, ROC-AUC y matriz de confusión
* Análisis de desempeño frente a costo computacional

---

## Autor

**Pablo Andrés Vázquez Calva**

Data Scientist

GitHub: https://github.com/pablo-vc87

LinkedIn: https://www.linkedin.com/in/pablo-avc/
