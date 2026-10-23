"""Gecikme, kayan istatistik ve takvim öznitelikleri.

Gecikmelerin hepsi >= tahmin ufku (28) olduğundan model, 28 gün ilerisini
gelecekteki bilinmeyen değerlere ihtiyaç duymadan doğrudan tahmin eder.
"""
import pandas as pd

try:
    import holidays
    _UK_HOLIDAYS = holidays.UK(years=range(2009, 2031))
except Exception:  # paket yoksa tatil özniteliği 0 olur
    _UK_HOLIDAYS = {}


def add_features(df: pd.DataFrame, lags=(28, 35, 42, 56), windows=(7, 28), horizon: int = 28) -> pd.DataFrame:
    df = df.sort_values(["item_id", "date"]).copy()
    g = df.groupby("item_id")["qty"]
    for lag in lags:
        df[f"lag_{lag}"] = g.shift(lag)
    shifted = g.shift(horizon)
    for w in windows:
        df[f"roll_mean_{w}"] = shifted.groupby(df["item_id"]).transform(lambda s: s.rolling(w, min_periods=1).mean())
    df["roll_std_28"] = shifted.groupby(df["item_id"]).transform(lambda s: s.rolling(28, min_periods=2).std())

    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["dom"] = df["date"].dt.day
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_holiday"] = df["date"].dt.date.map(lambda d: int(d in _UK_HOLIDAYS))
    df["item_code"] = df["item_id"].astype("category").cat.codes
    if "price" in df and df["price"].notna().any():
        df["price_rel"] = df["price"] / df.groupby("item_id")["price"].transform("mean")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    skip = {"date", "item_id", "qty", "price"}
    return [c for c in df.columns if c not in skip]
