import logging
import glob
import json
import os
from src.calculate_features import calculate_features
from typing import List, Dict, Any
from fuzzywuzzy import utils #type: ignore

logger = logging.getLogger(__name__)

class TrainModel:
    def __init__(self, config: Dict[str, Dict[str, Any]], project_root: str, label_path: str):
        self.project_root = project_root
        self.label_path = label_path
        self.config = config
        self.params = self.config.get("params", {})
        self.encoders = self.params.get("encoders", {})
        self.char_num: List[str] = self.encoders["char_num"]
        self._conversion_map: Dict[int, int] = self._build_conversion_map()

    def generate_features(self) -> List[Dict[str, Any]]:
        
        rows: List[Dict[str, Any]] = []
        json_files = glob.glob(os.path.join(self.label_path, '*.json'))

        for file_path in json_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                classified: Dict[str, Dict[str, Any]] = json.load(f)

            for _, poly_data in classified.items():
                if not poly_data:
                    continue

                text = poly_data.get("text", "")
                if not utils.validate_string(text):  # type: ignore
                    continue

                y_orig = int(poly_data.get("semantic_clasification", 0))
                feats = calculate_features(text, self.encoders)
                y_map = self._convert_label(y_orig)
                rows.append({
                    "text": text,
                    "label_original": y_orig,
                    "label_mapped": y_map,
                    **{f"f{i}": float(feats[i]) for i in range(len(feats))}
                })
        return rows

    def _build_conversion_map(self) -> Dict[int, int]:
        """Build the label conversion map once during initialization."""
        conv: Dict[int, int] = {}
        for item in self.encoders.get("conversion_map", []):
            for k, v in item.items():
                conv[int(k)] = int(v)
        return conv

    def _convert_label(self, y: int) -> int:
        """Convert label using pre-computed conversion map."""
        return self._conversion_map.get(y, 0)
