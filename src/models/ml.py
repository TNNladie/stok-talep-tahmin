"""Global ağaç modelleri: tüm ürünler tek modelde eğitilir."""
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def make_model(name: str, params: dict):
    params = dict(params or {})
    if name == "random_forest":
        return RandomForestRegressor(random_state=42, **params)
    if name == "xgboost":
        return XGBRegressor(random_state=42, objective="reg:squarederror", **params)
    if name == "lightgbm":
        return LGBMRegressor(random_state=42, objective="tweedie", **params)  # sıfırlı talep için tweedie
    raise ValueError(f"Bilinmeyen model: {name}")


ML_MODELS = ["random_forest", "xgboost", "lightgbm"]
