import os
import logging
import pickle
import yaml
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from src.train_model import TrainModel

logger = logging.getLogger(__name__)

class ModelGenerator:
    def __init__(self, config_file: str, project_root: str, label_path: str):
        time0 = time.perf_counter()
        self.project_root = project_root
        self.config_file = config_file
        self.label_path = label_path
        logger.info(f"Modelos iniciado en: {time.perf_counter() - time0:.6f}s")

    def _ngrams(self, s: str, n: int) -> List[str]:
        if n <= 0 or not s:
            return []
        if len(s) < n:
            return []
        return [s[i:i+n] for i in range(len(s) - n + 1)]
            
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
        
        self._train = TrainModel(config=self.config_dict, project_root=self.project_root, label_path=self.label_path)
        self.params = self.config_dict.get("params", {})
                
        features = self._train.generate_feaures()

        now = datetime.now()
        model_time = now.isoformat()
                            
        model: Dict[str, Any] = {
            "params": self.params,
            "model_time": model_time,
        }

        logger.info(f"Modelo generado en: {time.perf_counter()-time1}s")

        output_path = os.path.join(self.project_root, "models", "sc_model.pkl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            with open(output_path, "wb") as f:
                pickle.dump(model, f)
                
            logger.critical(f"Modelo 'WORD_FINDER' generado el {model_time} guardado en: %s", output_path)
            return model
            
        except AttributeError as e:
            logger.info(f"Error costruyendo Modelo: {e}", exc_info=True)
