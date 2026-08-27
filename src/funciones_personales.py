#======================================
import time
import pandas as pd
import numpy as np

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score
)


def evaluar_modelo_clasificacion(
    modelo,
    X_train,
    y_train,
    X_valid,
    y_valid,
    nombre_modelo,
    parametros=None
):
    """
        Entrena un modelo de clasificación y devuelve sus métricas de desempeño.
    
        Parámetros
        ----------
        modelo : estimador de sklearn o compatible
        X_train, y_train : datos de entrenamiento
        X_valid, y_valid : datos de validación
        nombre_modelo : str
        parametros : dict, opcional
            Diccionario con los hiperparámetros utilizados.
    
        Retorna
        -------
        DataFrame con una fila de resultados.
        """

    # ==========================================
    # Entrenamiento
    # ==========================================

    inicio_train = time.perf_counter()

    modelo.fit(
        X_train,
        y_train
    )

    tiempo_train = (
        time.perf_counter() - inicio_train
    )


    # ==========================================
    # Predicción
    # ==========================================

    inicio_pred = time.perf_counter()

    predicciones = modelo.predict(X_valid)

    tiempo_pred = (
        time.perf_counter() - inicio_pred
    )


    # ==========================================
    # Métricas principales
    # ==========================================

    f1 = f1_score(
        y_valid,
        predicciones
    )

    accuracy = accuracy_score(
        y_valid,
        predicciones
    )


    # ==========================================
    # Probabilidades o scores
    # ==========================================

    roc_auc = np.nan

    if hasattr(modelo, 'predict_proba'):

        scores = (
            modelo.predict_proba(X_valid)[:, 1]
        )

        roc_auc = roc_auc_score(
            y_valid,
            scores
        )

    elif hasattr(modelo, 'decision_function'):

        scores = (
            modelo.decision_function(X_valid)
        )

        roc_auc = roc_auc_score(
            y_valid,
            scores
        )


    # ==========================================
    # Formato de hiperparámetros
    # ==========================================

    if parametros is None or len(parametros) == 0:

        parametros_txt = 'Default'

    else:

        parametros_txt = ', '.join(
            f'{k}={v}'
            for k, v in parametros.items()
        )


    # ==========================================
    # Dataframe de resultados
    # ==========================================

    resultados = pd.DataFrame({

        'Modelo': [nombre_modelo],

        'Parámetros': [parametros_txt],

        'F1': [f1],

        'Accuracy': [accuracy],

        'ROC-AUC': [roc_auc],

        'Tiempo entrenamiento (s)': [
            tiempo_train
        ],

        'Tiempo predicción (s)': [
            tiempo_pred
        ]

    })


    return resultados, modelo
    
#====================================================
def probar_hiperparametros_clasificacion(
    modelo_base,
    lista_parametros,
    X_train,
    y_train,
    X_valid,
    y_valid,
    nombre_modelo
):
    """
    Prueba diferentes combinaciones de
    hiperparámetros y devuelve los resultados
    ordenados por F1.
    """

    resultados = []

    mejor_modelo = None

    mejor_f1 = -1


    for parametros in lista_parametros:

        # ==========================================
        # Crear modelo
        # ==========================================

        modelo = modelo_base(
            **parametros
        )


        # ==========================================
        # Evaluar modelo
        # ==========================================

        resultado, modelo_entrenado = (
            evaluar_modelo_clasificacion(

                modelo=modelo,

                X_train=X_train,
                y_train=y_train,

                X_valid=X_valid,
                y_valid=y_valid,

                nombre_modelo=nombre_modelo,

                parametros=parametros
            )
        )


        resultados.append(
            resultado
        )


        # ==========================================
        # Guardar mejor modelo
        # ==========================================

        f1_actual = resultado.loc[
            0,
            'F1'
        ]


        if f1_actual > mejor_f1:

            mejor_f1 = f1_actual

            mejor_modelo = (
                modelo_entrenado
            )


    # ==========================================
    # Unir resultados
    # ==========================================

    resultados = (

        pd.concat(
            resultados,
            ignore_index=True
        )

        .sort_values(
            'F1',
            ascending=False
        )

        .reset_index(
            drop=True
        )

    )


    return resultados, mejor_modelo
#=============================================
from sklearn.feature_extraction.text import TfidfVectorizer


def probar_modelos_tfidf(
    modelo_base,
    lista_parametros_modelo,
    lista_parametros_tfidf,
    X_train,
    y_train,
    X_valid,
    y_valid,
    nombre_modelo
):

    resultados = []

    mejor_modelo = None
    mejor_vectorizador = None

    mejor_f1 = -1


    for parametros_tfidf in lista_parametros_tfidf:

        # ==========================================
        # Crear TF-IDF
        # ==========================================

        vectorizador = TfidfVectorizer(
            **parametros_tfidf
        )


        # ==========================================
        # Transformar train
        # ==========================================

        inicio_vectorizacion = (
            time.perf_counter()
        )

        X_train_tfidf = (
            vectorizador.fit_transform(
                X_train
            )
        )


        X_valid_tfidf = (
            vectorizador.transform(
                X_valid
            )
        )


        tiempo_vectorizacion = (
            time.perf_counter()
            -
            inicio_vectorizacion
        )


        # ==========================================
        # Probar modelos
        # ==========================================

        for parametros_modelo in (
            lista_parametros_modelo
        ):

            modelo = modelo_base(
                **parametros_modelo
            )


            resultado, modelo_entrenado = (
                evaluar_modelo_clasificacion(

                    modelo=modelo,

                    X_train=X_train_tfidf,
                    y_train=y_train,

                    X_valid=X_valid_tfidf,
                    y_valid=y_valid,

                    nombre_modelo=nombre_modelo,

                    parametros=parametros_modelo
                )
            )


            # ==========================================
            # Agregar parámetros TF-IDF
            # ==========================================

            resultado[
                'Parámetros TF-IDF'
            ] = ', '.join(

                f'{k}={v}'

                for k, v in (
                    parametros_tfidf.items()
                )

            )


            resultado[
                'Tiempo vectorización (s)'
            ] = tiempo_vectorizacion


            resultados.append(
                resultado
            )


            # ==========================================
            # Guardar mejor
            # ==========================================

            f1_actual = (
                resultado.loc[
                    0,
                    'F1'
                ]
            )


            if f1_actual > mejor_f1:

                mejor_f1 = f1_actual

                mejor_modelo = (
                    modelo_entrenado
                )

                mejor_vectorizador = (
                    vectorizador
                )


    resultados = (

        pd.concat(
            resultados,
            ignore_index=True
        )

        .sort_values(
            'F1',
            ascending=False
        )

        .reset_index(
            drop=True
        )

    )


    return (
        resultados,
        mejor_modelo,
        mejor_vectorizador
    )
#==================================
import re
#import contractions

def limpiar_y_expandir_texto(text):
    # 1. Convertir a minúsculas
    text = text.lower()
    
    # 2. Expandir contracciones (ej: "don't" -> "do not", "it's" -> "it is")
    #text = contractions.fix(text)
    
    # 3. Eliminar caracteres que no sean letras a-z o espacios
    text = re.sub(r"[^a-z'\s]", ' ', text)
    
    # 4. Colapsar espacios múltiples y quitar extremos
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
#======================================================
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def lematizar_texto_nltk(text):

    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    
    tokens_lemmatized = [
        lemmatizer.lemmatize(token)
        for token in tokens
    ]
    
    return ' '.join(tokens_lemmatized)
#=====================================================
# Función que permite utilizar cualquier combinación de vectorizador y modelo
def clasificar_reviews(
    reviews,
    vectorizador,
    modelo,
    nombre_modelo
):
    
    X_reviews = vectorizador.transform(
        reviews
    )
    
    predicciones = modelo.predict(
        X_reviews
    )
    
    return pd.DataFrame({
        'Review': reviews,
        nombre_modelo: predicciones
    })