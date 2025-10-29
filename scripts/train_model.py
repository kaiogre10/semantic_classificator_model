# src/train_model.py
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
import glob
import json
from create_features import generate_features

# --- Configuración ---
DATA_FOLDER = '../data/'
MODEL_DIR = '../models'
MODEL_PATH = os.path.join(MODEL_DIR, 'header_classifier.pkl')

# --- 1. Cargar y Combinar TODOS los JSON (Lógica Modificada) ---
print(f"Buscando archivos JSON en la carpeta {DATA_FOLDER}...")
json_files = glob.glob(os.path.join(DATA_FOLDER, '*.json'))

if not json_files:
    print(f"Error: No se encontraron archivos JSON en {DATA_FOLDER}.")
    exit()

list_of_dfs = []
for file_path in json_files:
    print(f"Cargando archivo: {os.path.basename(file_path)}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extraer los datos generales y las palabras
    page_info = data.get('page_info', {})
    words_data = data.get('words', [])

    if not words_data:
        continue

    # Convertir la lista de palabras a un DataFrame
    df_temp = pd.DataFrame(words_data)

    # Añadir los datos generales a CADA fila del DataFrame temporal
    df_temp['page_w'] = page_info.get('page_w', 0)
    df_temp['page_h'] = page_info.get('page_h', 0)

    list_of_dfs.append(df_temp)

print("\nCombinando todos los datos en un único DataFrame...")
if not list_of_dfs:
    print("Error: Ningún archivo contenía datos de palabras válidos.")
    exit()

df = pd.concat(list_of_dfs, ignore_index=True)

print(f"¡Carga completa! Se procesaron {len(json_files)} archivos con un total de {len(df)} palabras.")

# 2. Generar Características
print("\nGenerando características...")
features, labels = generate_features(df, label_column='is_header')

if labels is None:
    print("Error: La columna 'is_header' no se encontró en tus archivos JSON.")
    exit()

# --- 3. Dividir Datos para Entrenamiento (sin cambios aquí) ---
print("Dividiendo los datos...")
X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size=0.3,
    random_state=42,
    stratify=labels
)

# --- 4. Entrenar el Modelo (sin cambios aquí) ---
print("Entrenando el modelo LightGBM...")
lgbm = lgb.LGBMClassifier(objective='binary', random_state=42, is_unbalance=True)
lgbm.fit(X_train, y_train)

# --- 5. Evaluar el Rendimiento (sin cambios aquí) ---
print("\n--- Evaluación del Modelo ---")
y_pred = lgbm.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['No-Encabezado (0)', 'Encabezado (1)']))

# --- 6. Guardar el Modelo (sin cambios aquí) ---
print(f"Guardando el modelo en {MODEL_PATH}...")
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(lgbm, MODEL_PATH)

print("\n¡Entrenamiento completado! Tu modelo, entrenado con todos los JSON, está listo.")