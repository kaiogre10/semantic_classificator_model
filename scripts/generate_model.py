import os
import logging
import pickle
import yaml
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Optional, List
from src.train_model import TrainModel

logger = logging.getLogger(__name__)

class ModelGenerator:
    def __init__(self, config_file: str, project_root: str, label_path: str):
        time0 = time.perf_counter()
        self.project_root = project_root
        self.config_file = config_file
        self.label_path = label_path
        logger.info(f"Modelos iniciado en: '{time.perf_counter()-time0:.6f}s'")
            
    def generate_model(self, config_file: str, label_path: str) -> Optional[Dict[str, Any]]:
        """Lee YAML, normaliza variantes, precomputa n-gramas 2-5y guarda un pickle con toda la info necesaria para WordFinder."""
        time1 = time.perf_counter()
        self.label_path = label_path
        self.config_file = config_file
        self.config_dict: Dict[str, Dict[str, Any]] = {}
        try:
            if not os.path.exists(self.config_file):
                raise FileNotFoundError(f"No existe config: {self.config_file}")
            with open(self.config_file, "r", encoding="utf-8") as f:
                if self.config_file:
                    self.config_dict = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error cargando el modelo: {e}", exc_info=True)
            return None
        self.params = self.config_dict.get("params", {})
        self.encoders = self.params.get("encoders", {})

        try:
            self._train = TrainModel(config=self.config_dict, project_root=self.project_root, label_path=self.label_path)
            rows = self._train.generate_features()
            df = pd.DataFrame(rows)
            df.reset_index(drop=True, inplace=True)  # RangeIndex 0..N-1

            # Construir X,y
            feature_cols = [c for c in df.columns if c.startswith("f")]
            X = df[feature_cols].to_numpy(dtype=np.float32)
            y = df["label_mapped"].to_numpy(dtype=np.int32)

            # Split y entrenamiento con params['model_config']
            mc = self.params.get("model_config", {})
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

            now = datetime.now()
            model_gen = now.isoformat()
            
            model = lgb.train(
                mc,
                train_data,
                valid_sets=[valid_data],
                num_boost_round=100,
                callbacks=[lgb.early_stopping(stopping_rounds=10)],
            )

        except Exception as e:
            logger.error(f"Error generadndo modelo: {e}", exc_info=True)

        try:
        # Evaluación en etiquetas originales (opcional)
            inv_map: List[Dict[str, int]] = self.encoders.get("conversion_map", [])
            y_pred = model.predict(X_test)
            y_pred_conv = np.argmax(y_pred, axis=1)
            y_pred_orig = np.array([inv_map[int(v)] for v in y_pred_conv])

        except Exception as e:
            logger.error(f"Error evaluando modelo: {e}", exc_info=True)

        logger.info(f"Modelo generado en: {time.perf_counter()-time1}s")

        output_path = os.path.join(self.project_root, "models", "sc_model.pkl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            with open(output_path, "wb") as f:
                pickle.dump(model, f)
                
            logger.critical(f"Modelo 'CLASSIFICADOR' generado el {model_gen} guardado en: %s", output_path)
            return model
            
        except AttributeError as e:
            logger.info(f"Error costruyendo Modelo: {e}", exc_info=True)
