
import logging
import numpy as np
from fuzzywuzzy import utils #type: ignore
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

def calculate_features(s: str, encoders: Dict[str, Any]):
    char_num: List[str] = encoders["char_num"]
    density_encoder: Dict[str, float] = encoders.get("density_encoder", {})
    inv_encoder_density: Dict[str, float] = encoders.get("inv_encoder_density", {})

    encoded_poly = encode_text(s, density_encoder)
    mean_encoded, std_encoded, var_encoded = vectorice_values(encoded_poly)

    inv_encoded_poly = encode_text(s, inv_encoder_density)
    inv_mean_encoded, inv_std_encoded, inv_var_encoded = vectorice_values(inv_encoded_poly)

    morph_poly = get_morphological_encode(s, char_num)
    mean_morph, std_morph, var_morph = vectorice_values(morph_poly)

    num_pct, alpha_pct, spc_pct, word_len = calculate_percentages(s, char_num)

    money = calculate_textual_features(s)

    features = np.array([
        mean_encoded, 
        std_encoded, 
        var_encoded, 
        inv_mean_encoded, 
        inv_std_encoded, 
        inv_var_encoded, 
        mean_morph, 
        std_morph, 
        var_morph, 
        num_pct, 
        alpha_pct, 
        spc_pct,
        word_len,
        money
        ],
        dtype=np.float32
    )

    feature_names = [
        "mean_encoded",
        "std_encoded",
        "var_encoded",
        "inv_mean_encoded",
        "inv_std_encoded",
        "inv_var_encoded",
        "mean_morph",
        "std_morph",
        "var_morph",
        "num_pct",
        "alpha_pct",
        "spc_pct",
        "word_len",
        "money"
    ]

    feature_dict = {name: float(value) for name, value in zip(feature_names, features)}
    log_message = "Features: " + ", ".join([f"{k}: {v:.4f}" for k, v in feature_dict.items()])
    logger.debug(log_message)

    return features
    
def calculate_textual_features(s: str) -> float:
    point_c = float(s.count('.'))
    money_c = float(s.count('$'))
    coma_c = float(s.count(','))
    return coma_c + money_c + point_c

def calculate_percentages(s: str, char_num: List[str]):
    chars = [ch for ch in s if not ch.isspace()]
    word_len = len(chars)
    num_pct = (sum(1 for ch in chars if ch in char_num) / word_len) * 100.0 if word_len else 0.0
    alpha_pct = (sum(1 for ch in chars if ch.isalpha() and ch not in char_num) / word_len) * 100.0 if word_len else 0.0

    alphanum =  num_pct + alpha_pct
    if alphanum == 100.0:
        spc_pct = 0.0
    else:
        spc_pct = 100.0 - alphanum
    
    return num_pct, alpha_pct, spc_pct, float(word_len)

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

def get_morphological_encode(text: str, char_num: List[str]) -> List[float]:
    try:
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

def vectorice_values(data_list: List[float]) -> List[float]:
    """
    Calcula estadísticas vectorizadas (media, desviación estándar, varianza) de una lista de valores.
    """
    if not data_list:
        return [0.0, 0.0, 0.0]

    value_array = np.array(data_list, dtype=np.float32)
    poly_mean = np.mean(value_array)
    poly_std = np.std(value_array)
    poly_var = np.var(value_array)
    
    return [float(poly_mean), float(poly_std), float(poly_var)]
