# src/create_features.py
import pandas as pd

def generate_features(df: pd.DataFrame, label_column='is_header'):
    """
    Toma un DataFrame con datos de OCR y genera las características
    para el entrenamiento o la predicción.
    """

    # --- Características Geométricas (Normalizadas) ---
    # Asegúrate de que las columnas de página existan y no sean cero
    df['page_h'] = df['page_h'].replace(0, 1)
    df['page_w'] = df['page_w'].replace(0, 1)

    df['x_center'] = (df['xmin'] + df['xmax']) / 2
    df['y_center'] = (df['ymin'] + df['ymax']) / 2
    df['width'] = df['xmax'] - df['xmin']
    df['height'] = df['ymax'] - df['ymin']

    df['rel_y_center'] = df['y_center'] / df['page_h']
    df['rel_x_center'] = df['x_center'] / df['page_w']
    df['rel_width'] = df['width'] / df['page_w']
    df['rel_height'] = df['height'] / df['page_h']

    # --- Selección de Características Finales ---
    feature_columns = [
        'length', 'num_digits', 'num_alpha', 'is_upper',
        'rel_y_center', 'rel_x_center', 'rel_width', 'rel_height'
    ]
    
    # Extraer las características y las etiquetas (si existen)
    features = df[feature_columns]
    labels = df[label_column] if label_column in df.columns else None
    
    return features, labels