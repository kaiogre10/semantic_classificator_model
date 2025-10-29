from fuzzywuzzy import utils  # type:ignore
import logging
import json
import numpy as np
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class TrainModel:
    def __init__(self, config: Dict[str, List[Tuple[str, float]]], project_root: str, label_path: str):
        self.project_root = project_root
        self.label_path = label_path
        self.config = config
        self.params = config.get("params", {})

    def generate_feaures(self):
        json_files = glob.glob(os.path.join(self.label_path, '*.json'))
        if not json_files:
            logger.error(f"No se encontraron archivos JSON en la ruta: {self.label_path}")
            return {}

        for file_path in json_files:
            file_name = os.path.basename(file_path)
            logger.info(f"\n--- Procesando archivo: {file_name} ---")

            with open(file_path, 'r', encoding='utf-8') as f:
                classified_words: Dict[str, Dict[str, Any]] = json.load(f)
                time1 = time.perf_counter()
                for poly_id, poly_data in polygons.items():
                    if not poly_data:
                        continue
                    sc_word: int = poly_data.semantic_clasification[1]

        return features

    def encode_text(text: str, encoder: Dict[str, float]) -> List[float]:
        try:
            if not utils.validate_string(text):  # type:ignore
                return []

            if text:
                minus_text = text.lower()
                compact_text = ''.join(minus_text.split())
                encoded_poly = [encoder.get(char, 0) for char in compact_text]

                return encoded_poly

        except Exception as e:
            logger.warning(f"Error codificando polígonos: {e}", exc_info=True)
        return []

    def get_morphological_map() -> List[str]:
        self.char_num: List[str] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", ",", "$"]
        return char_num

    def get_morphological_encode(text: str) -> List[float]:
        try:
            char_num = get_morphological_map()
            result: List[float] = []
            if not utils.validate_string(text):  # type:ignore
                return []

            compact_text = ''.join(text.split())
            for ch in compact_text:
                if ch in char_num:
                    result.append(1.0)
                elif ch.isalpha():
                    result.append(-1.0)
                else:
                    result.append(0.0)
            return result

        except Exception as e:
            logger.warning(f"Error codificando polígonos: {e}", exc_info=True)
        return []

    def _clasify_words(self, polygons: Dict[str, Polygons], encoder: Dict[str, float], inv_encoder: Dict[str, float]) -> \
    Dict[str, int]:
        semantic_range: Tuple[float, float] = self.worker_config.get("semantic_range", [])
        encode_mean: Tuple[float, float] = self.worker_config.get("encode_mean", [])
        morph_mean: Tuple[float, float] = self.worker_config.get("morph_mean", [])

        texts: Dict[str, str] = {poly_id: (polygon.ocr_text or "") for poly_id, polygon in polygons.items()}
        final_results: Dict[str, int] = {}

        for pid, s in texts.items():
            chars = [ch for ch in s if not ch.isspace()]
            total = len(chars)
            char_num = get_morphological_map()
            pct = (sum(1 for ch in chars if ch in char_num) / total) * 100.0 if total else 0.0

            encoded_poly = encode_text(s, encoder)
            poly_mean = np.mean(encoded_poly)

            inv_encoded_poly = encode_text(s, inv_encoder)
            inv_poly_mean = np.mean(inv_encoded_poly)

            morph_text = get_morphological_encode(s)
            poly_morph_mean = np.mean(morph_text) if morph_text else - 1.0

            # Lógica de clasificación simplificada a enteros
            semantic_type = 0  # descriptive por defecto

            if contains_quantitative(s):
                semantic_type = 2  # quantitative
            elif find_umd(s):
                semantic_type = -2  # umd
            elif morph_mean[1] < poly_morph_mean and poly_mean < encode_mean[0] and encode_mean[1] < inv_poly_mean and \
                    semantic_range[1] < pct:
                has_quantitative = find_quantitative(s)
                if has_quantitative:
                    semantic_type = 2  # quantitative
                else:
                    semantic_type = 1  # numeric
            elif semantic_range[0] < pct < semantic_range[1] and morph_mean[0] < poly_morph_mean < morph_mean[1]:
                semantic_type = -1  # code
            else:
                # pct < semantic_range[0] and poly_morph_mean < morph_mean[0]
                semantic_type = 0  # Descriptive

            logger.debug(
                f"{pid}: '{s}'| mean: {poly_mean:.4f}, inv_mean: {inv_poly_mean}, morph: {poly_morph_mean}, {pct}% | sc: {semantic_type}")

            final_results[pid] = semantic_type

        return final_results
