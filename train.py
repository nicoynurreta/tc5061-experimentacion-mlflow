# Entrenamiento y evaluación de un clasificador de fallas con el dataset AI4I 2020.
# Viene del notebook de la semana 3; el preprocesamiento vive dentro del pipeline.
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERICAS = ["temp_aire_K", "temp_proceso_K", "velocidad_rpm", "par_Nm", "desgaste_min"]
MODOS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
COLUMNAS = {
    "UDI": "udi",
    "Product ID": "id_producto",
    "Type": "tipo",
    "Air temperature [K]": "temp_aire_K",
    "Process temperature [K]": "temp_proceso_K",
    "Rotational speed [rpm]": "velocidad_rpm",
    "Torque [Nm]": "par_Nm",
    "Tool wear [min]": "desgaste_min",
    "Machine failure": "falla",
}


def leer_argumentos():
    parser = argparse.ArgumentParser(
        description="Entrena un clasificador de fallas de máquina con el dataset AI4I 2020 y lo evalúa en prueba."
    )
    parser.add_argument("--modelo", choices=["regresion_logistica", "bosque_aleatorio"], default="regresion_logistica",
                        help="clasificador a entrenar (por defecto %(default)s)")
    parser.add_argument("--C", type=float, default=1.0,
                        help="inverso de la fuerza de regularización de la regresión logística (por defecto %(default)s)")
    parser.add_argument("--class_weight", choices=["balanced", "ninguno"], default="balanced",
                        help="peso de las clases en el clasificador (por defecto %(default)s)")
    parser.add_argument("--max_iter", type=int, default=1000,
                        help="iteraciones máximas del solver de la regresión logística (por defecto %(default)s)")
    parser.add_argument("--n_estimators", type=int, default=200,
                        help="número de árboles del bosque aleatorio (por defecto %(default)s)")
    parser.add_argument("--max_depth", type=int, default=None,
                        help="profundidad máxima de cada árbol del bosque (por defecto sin límite)")
    parser.add_argument("--min_samples_leaf", type=int, default=1,
                        help="mínimo de registros por hoja en el bosque aleatorio (por defecto %(default)s)")
    parser.add_argument("--semilla", type=int, default=42,
                        help="semilla de la división y del bosque aleatorio (por defecto %(default)s)")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="fracción de registros para prueba (por defecto %(default)s)")
    parser.add_argument("--datos", default="datos/ai4i2020.csv",
                        help="ruta del archivo csv (por defecto %(default)s)")
    parser.add_argument("--salida", default="salidas",
                        help="carpeta donde se guardan la matriz de confusión y las métricas (por defecto %(default)s)")
    return parser.parse_args()


def cargar_datos(ruta):
    return pd.read_csv(ruta)


def limpiar(datos):
    # nombres cortos en español, los mismos del notebook
    datos = datos.rename(columns=COLUMNAS)
    # la falla debe coincidir con tener al menos un modo de falla; se quitan las contradictorias
    con_modo = datos[MODOS].sum(axis=1) > 0
    inconsistentes = (datos["falla"] == 1) != con_modo
    datos = datos[~inconsistentes].copy()
    datos["tipo"] = datos["tipo"].astype("category")
    datos[NUMERICAS] = datos[NUMERICAS].astype(float)
    # fuera identificadores y modos de falla, que revelarían la respuesta
    return datos.drop(columns=["udi", "id_producto"] + MODOS)


def dividir(datos, test_size, semilla):
    X = datos.drop(columns="falla")
    y = datos["falla"]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=semilla)


def construir_modelo(args):
    # escalado de las numéricas y one hot del tipo, siempre dentro del pipeline
    preprocesamiento = ColumnTransformer([
        ("numericas", StandardScaler(), NUMERICAS),
        ("tipo", OneHotEncoder(handle_unknown="ignore"), ["tipo"]),
    ])
    peso = None if args.class_weight == "ninguno" else args.class_weight
    if args.modelo == "regresion_logistica":
        clasificador = LogisticRegression(C=args.C, class_weight=peso, max_iter=args.max_iter)
    else:
        clasificador = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight=peso,
            random_state=args.semilla,
        )
    return Pipeline([("preprocesamiento", preprocesamiento), ("clasificador", clasificador)])


def evaluar(modelo, X, y):
    predicciones = modelo.predict(X)
    matriz = confusion_matrix(y, predicciones)
    vn, fp, fn, vp = matriz.ravel()
    metricas = {
        "exactitud": accuracy_score(y, predicciones),
        "precision": precision_score(y, predicciones, zero_division=0),
        "recall": recall_score(y, predicciones),
        "f1": f1_score(y, predicciones, zero_division=0),
        "verdaderos_negativos": int(vn),
        "falsos_positivos": int(fp),
        "falsos_negativos": int(fn),
        "verdaderos_positivos": int(vp),
    }
    return metricas, matriz


def guardar_artefactos(matriz, metricas, carpeta):
    os.makedirs(carpeta, exist_ok=True)
    figura, eje = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(matriz, display_labels=["Sin falla", "Falla"]).plot(ax=eje, cmap="Blues", colorbar=False, values_format="d")
    eje.set_title("Matriz de confusión en prueba")
    eje.set_xlabel("Predicción")
    eje.set_ylabel("Valor real")
    figura.tight_layout()
    ruta_figura = os.path.join(carpeta, "matriz_confusion.png")
    figura.savefig(ruta_figura, dpi=120)
    plt.close(figura)
    ruta_metricas = os.path.join(carpeta, "metricas.json")
    with open(ruta_metricas, "w", encoding="utf-8") as archivo:
        json.dump(metricas, archivo, indent=2)
    return [ruta_figura, ruta_metricas]


def main():
    args = leer_argumentos()
    datos = limpiar(cargar_datos(args.datos))
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = dividir(datos, args.test_size, args.semilla)
    print("Registros de entrenamiento:", len(X_entrenamiento))
    print("Registros de prueba:", len(X_prueba))

    modelo = construir_modelo(args)
    modelo.fit(X_entrenamiento, y_entrenamiento)
    metricas, matriz = evaluar(modelo, X_prueba, y_prueba)
    guardar_artefactos(matriz, metricas, args.salida)
    for nombre in ["exactitud", "precision", "recall", "f1"]:
        print(f"{nombre}: {metricas[nombre]:.4f}")
    print("Falsos positivos:", metricas["falsos_positivos"])
    print("Falsos negativos:", metricas["falsos_negativos"])


if __name__ == "__main__":
    main()
