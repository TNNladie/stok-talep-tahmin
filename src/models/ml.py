"""Global ağaç modelleri: tüm ürünler tek modelde eğitilir."""
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Sıfır yığılmalı talep için varsayılan kayıp fonksiyonları. LightGBM ve XGBoost aynı
# (tweedie) kaybı kullanır ki karşılaştırma adil olsun; config ile değiştirilebilir.
DEFAULT_OBJECTIVE = {"xgboost": "reg:tweedie", "lightgbm": "tweedie"}


def make_model(name: str, params: dict):
    params = dict(params or {})
    if name == "random_forest":
        return RandomForestRegressor(random_state=42, **params)
    if name == "xgboost":
        params.setdefault("objective", DEFAULT_OBJECTIVE["xgboost"])
        return XGBRegressor(random_state=42, **params)
    if name == "lightgbm":
        params.setdefault("objective", DEFAULT_OBJECTIVE["lightgbm"])
        return LGBMRegressor(random_state=42, **params)
    raise ValueError(f"Bilinmeyen model: {name}")


ML_MODELS = ["random_forest", "xgboost", "lightgbm"]
