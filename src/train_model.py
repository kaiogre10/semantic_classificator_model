
import logging
import pandas as pd
import lightgbm as lgb
import time
import joblib
import glob
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from calculate_features import calculate_features
from typing import List, Dict, Any, Tuple
from fuzzywuzzy import utils #type: ignore

logger = logging.getLogger(__name__)

class TrainModel:
    def __init__(self, config: Dict[str, Dict[str, Any]], project_root: str, label_path: str):
        self.project_root = project_root
        self.label_path = label_path
        self.config = config
        self.params = config.get("params", {})
        self.char_num: List[str] = self.params["char_num"]

    def generate_feaures(self) -> Dict[str, Tuple[List[str], List[int]]]:
        """
        Lee todos los JSON de self.label_path y devuelve un dict por archivo:
        { filename: (texts, classifications) }
        donde texts es List[str] y classifications es List[int]
        """
        json_files = glob.glob(os.path.join(self.label_path, '*.json'))
        if not json_files:
            logger.error(f"No se encontraron archivos JSON en la ruta: {self.label_path}")
            return {}

        all_data: Dict[str, Tuple[List[str], List[int]]] = {}
        for file_path in json_files:
            file_name = os.path.basename(file_path)
            logger.info(f"Procesando archivo: {file_name}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    classified_words: Dict[str, Dict[str, Any]] = json.load(f)
                    
                    texts: List[str] = []
                    classifications_labels: List[int] = []
                    
                    for poly_id, poly_data in classified_words.items():
                        if poly_data is None:
                            continue
                        
                        text = poly_data.get("text", "")
                        if not utils.validate_string(text): #type: ignore
                            continue

                        classification_label = poly_data.get("semantic_clasification", 0)

                        df_temp = pd.DataFrame(text)

                        features, labels = calculate_features(df, label_column=classification_label, self.params)
                        
                        if text and classification_label is not None:
                            texts.append(text)
                            classifications_labels.append(classification_label)
                    
                    all_data[file_name] = (texts, classifications_labels)
                    logger.info(f"Procesado {file_name}: {all_data}")
                    
            except Exception as e:
                logger.exception(f"Error leyendo {file_name}: {e}")
        return all_data

    