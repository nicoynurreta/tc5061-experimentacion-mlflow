# Experimentación con MLflow: predicción de fallas de máquina

Script de entrenamiento parametrizado para el dataset AI4I 2020 de mantenimiento predictivo, instrumentado con MLflow Tracking. Es la continuación del notebook de la semana 3 del curso Operaciones de aprendizaje automático (TC5061): el notebook queda solo como análisis exploratorio y `train.py` es la única ruta de entrenamiento.

## Estructura

```
train.py             carga, limpieza, división, modelo, evaluación y registro en MLflow
requirements.txt     dependencias con versiones fijas
datos/ai4i2020.csv   dataset AI4I 2020
notebooks/           notebook de exploración de la semana 3 (EDA y limpieza)
```

## Instalación

Probado con Python 3.14.2.

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

En Linux o macOS el entorno se activa con `source venv/bin/activate`.

## Registro de MLflow

Por defecto cada corrida se registra en `mlflow.db` (SQLite) y sus artefactos en `mlruns/`, los dos dentro de la carpeta del repositorio, sin necesidad de levantar un servidor. Para verlas en la interfaz, desde la misma carpeta:

```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

La interfaz queda en http://127.0.0.1:5000. Si se prefiere un servidor de tracking, se lanza en otra terminal `mlflow server --host 127.0.0.1 --port 5000` y se pasa `--tracking_uri http://127.0.0.1:5000` en cada corrida.

## Ejecución

Con los valores por defecto se entrena la regresión logística base del notebook (C 1.0, class_weight balanced, max_iter 1000, semilla 42, 20 % de prueba):

```
python train.py
```

Cualquier hiperparámetro se cambia desde la línea de comandos sin tocar el código, por ejemplo:

```
python train.py --modelo bosque_aleatorio --max_depth 5 --nombre_corrida ba_profundidad5
```

`python train.py --help` describe todos los argumentos:

| Argumento | Por defecto | Qué controla |
|---|---|---|
| `--modelo` | `regresion_logistica` | `regresion_logistica` o `bosque_aleatorio` |
| `--C` | 1.0 | inverso de la regularización de la regresión logística |
| `--class_weight` | `balanced` | `balanced` o `ninguno`, aplica a los dos modelos |
| `--max_iter` | 1000 | iteraciones máximas del solver de la regresión logística |
| `--n_estimators` | 200 | número de árboles del bosque aleatorio |
| `--max_depth` | sin límite | profundidad máxima de cada árbol |
| `--min_samples_leaf` | 1 | mínimo de registros por hoja |
| `--semilla` | 42 | semilla de la división y del bosque aleatorio |
| `--test_size` | 0.2 | fracción de registros para prueba |
| `--datos` | `datos/ai4i2020.csv` | ruta del csv |
| `--experimento` | `experimentacion_mantenimiento_ai4i` | nombre del experimento en MLflow |
| `--nombre_corrida` | lo asigna MLflow | nombre de la corrida |
| `--tracking_uri` | `sqlite:///mlflow.db` | almacén de MLflow: archivo SQLite local o servidor |

Cada corrida registra en MLflow los hiperparámetros con `log_param`; las métricas en prueba `exactitud`, `precision`, `recall`, `f1`, `falsos_positivos` y `falsos_negativos` con `log_metric`; la matriz de confusión (`matriz_confusion.png`) y `metricas.json` con `log_artifact`; y el pipeline entrenado con `mlflow.sklearn.log_model`.

## Corridas del experimento

Nueve corridas en el experimento `experimentacion_mantenimiento_ai4i`. Cada una cambia un solo hiperparámetro respecto a su base (`rl_base` o `ba_base`), para poder atribuir cada diferencia a ese hiperparámetro:

```
python train.py --nombre_corrida rl_base
python train.py --nombre_corrida rl_base_repeticion
python train.py --C 0.1 --nombre_corrida rl_C0.1
python train.py --C 10 --nombre_corrida rl_C10
python train.py --class_weight ninguno --nombre_corrida rl_sin_peso
python train.py --max_iter 5 --nombre_corrida rl_iter5
python train.py --modelo bosque_aleatorio --nombre_corrida ba_base
python train.py --modelo bosque_aleatorio --max_depth 5 --nombre_corrida ba_profundidad5
python train.py --modelo bosque_aleatorio --min_samples_leaf 20 --nombre_corrida ba_hojas20
```

En la interfaz se seleccionan las corridas del experimento y se contrastan con el botón Compare.

## Reproducibilidad

La semilla (`--semilla`, 42 por defecto) fija la división estratificada de entrenamiento y prueba y el muestreo del bosque aleatorio; la regresión logística con el solver lbfgs es determinista. Dos corridas con los mismos argumentos producen las mismas métricas, como se ve en `rl_base` y `rl_base_repeticion`. Las versiones de las librerías están fijas en `requirements.txt`.

## Datos

Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset. UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C. Licencia CC BY 4.0.
