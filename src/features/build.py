"""Gecikme, kayan istatistik ve takvim öznitelikleri.

Gecikmelerin hepsi >= tahmin ufku (28) olduğundan model, 28 gün ilerisini
gelecekteki bilinmeyen değerlere ihtiyaç duymadan doğrudan tahmin eder.
Fiyat özniteliği de aynı nedenle ufuk kadar geriye kaydırılır: tahmin anında
gelecekteki fiyat bilinmez, eğitimde de kullanılmamalıdır.
"""
import numpy as np
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
    # Satış yapılan gün oranı (aralıklı talep göstergesi)
    df["nz_rate_28"] = (shifted > 0).astype(float).where(shifted.notna()).groupby(df["item_id"]).transform(
        lambda s: s.rolling(28, min_periods=1).mean())
    # Geçen yılın aynı dönemi: t-364 etrafında 28 günlük ortalama (en yakın gün t-350 >= ufuk)
    s364 = g.shift(364)
    df["yoy_mean_28"] = s364.groupby(df["item_id"]).transform(lambda s: s.rolling(28, min_periods=1, center=True).mean())
    # Mağaza geneli talep seviyesi ve ürünün bu seviyeye oranı
    store = df.groupby("date")["qty"].sum(min_count=1).sort_index()
    store = store.rolling("28D", min_periods=1).mean().shift(horizon, freq="D")
    df["store_roll_28"] = df["date"].map(store)
    df["store_rel"] = df["roll_mean_28"] / df["store_roll_28"].replace(0, np.nan) if "roll_mean_28" in df else np.nan

    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["dom"] = df["date"].dt.day
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_holiday"] = df["date"].dt.date.map(lambda d: int(d in _UK_HOLIDAYS))
    df["is_open"] = df["is_open"].fillna(1).astype(int) if "is_open" in df else 1
    df["item_code"] = df["item_id"].astype("category").cat.codes
    if "price" in df and df["price"].notna().any():
        p_lag = df.groupby("item_id")["price"].shift(horizon)
        df["price_rel"] = p_lag / p_lag.groupby(df["item_id"]).transform("mean")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    skip = {"date", "item_id", "qty", "price"}
    return [c for c in df.columns if c not in skip]


def future_open_days(df: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    """Gelecek günlerin açık/kapalı tahmini: geçmişte o haftanın günü ve takvim günü (ay-gün)
    çoğunlukla açıksa açık kabul edilir (ör. cumartesi ve Noel–Yılbaşı kapanışı)."""
    if "is_open" not in df:
        return np.ones(len(dates), dtype=int)
    days = df.drop_duplicates("date")[["date", "is_open"]]
    by_dow = days.groupby(days["date"].dt.dayofweek)["is_open"].mean()
    by_md = days.groupby(days["date"].dt.strftime("%m-%d"))["is_open"].mean()
    dow_ok = dates.dayofweek.map(by_dow).fillna(1).to_numpy() >= 0.5
    md_ok = pd.Index(dates.strftime("%m-%d")).map(by_md).fillna(1).to_numpy() >= 0.5
    return (dow_ok & md_ok).astype(int)


def make_future_frame(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Son tarihten sonraki `horizon` gün için boş (qty=NaN) satırlar ekler."""
    last = df["date"].max()
    dates = pd.date_range(last + pd.Timedelta(days=1), periods=horizon, freq="D")
    items = df["item_id"].unique()
    future = pd.DataFrame({"date": np.tile(dates, len(items)), "item_id": np.repeat(items, len(dates))})
    future["qty"], future["price"] = np.nan, np.nan
    future["is_open"] = np.tile(future_open_days(df, dates), len(items))
    return pd.concat([df, future[df.columns.intersection(future.columns)]], ignore_index=True)
