"""Zaman bazlı genişleyen pencere doğrulaması.

Kat k için: eğitim = test başlangıcından önceki tüm günler, test = sonraki `horizon` gün.
Katlar sondan geriye doğru dizilir; rastgele bölme YOKTUR.
`offset` > 0 verilirse son `offset` kat atlanır — hiperparametre ayarı bu "doğrulama
katlarında" yapılır, böylece raporlanan test katları ayar sırasında hiç görülmez.
"""
import numpy as np
import pandas as pd

from src.evaluation.metrics import all_metrics
from src.features.build import add_features, feature_columns
from src.models.baselines import BASELINES
from src.models.ml import ML_MODELS, make_model


def fold_cutoffs(dates: pd.Series, horizon: int, n_folds: int, offset: int = 0) -> list[pd.Timestamp]:
    last = dates.max()
    return [last - pd.Timedelta(days=horizon * (k + offset)) for k in range(n_folds, 0, -1)]


def clip_target(train: pd.DataFrame, q: float | None) -> pd.Series:
    """Eğitim hedefini ürün bazında q yüzdeliğinde kırpar (tek seferlik toptan siparişler).
    Yüzdelik yalnızca eğitim verisinden hesaplanır; test hedefi asla kırpılmaz."""
    if not q:
        return train["qty"]
    cap = train.groupby("item_id")["qty"].transform(lambda s: s.quantile(q))
    return train["qty"].clip(upper=cap)


def run_baselines(df: pd.DataFrame, cutoff: pd.Timestamp, horizon: int) -> pd.DataFrame:
    rows = []
    for item, g in df.groupby("item_id"):
        g = g.sort_values("date")
        hist = g[g["date"] <= cutoff]["qty"].to_numpy()
        test = g[(g["date"] > cutoff) & (g["date"] <= cutoff + pd.Timedelta(days=horizon))]
        for name, fn in BASELINES.items():
            rows.append(pd.DataFrame({"item_id": item, "date": test["date"].values, "y": test["qty"].values,
                                      "pred": fn(hist, len(test)), "model": name}))
    return pd.concat(rows, ignore_index=True)


def train_matrix(feat: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    cols = feature_columns(feat)
    return feat[(feat["date"] <= cutoff) & feat["qty"].notna()].dropna(subset=[c for c in cols if c.startswith("lag_")])


def run_ml(feat: pd.DataFrame, cutoff: pd.Timestamp, horizon: int, model_cfg: dict, models=ML_MODELS,
           clip_q: float | None = None) -> pd.DataFrame:
    cols = feature_columns(feat)
    train = train_matrix(feat, cutoff)
    test = feat[(feat["date"] > cutoff) & (feat["date"] <= cutoff + pd.Timedelta(days=horizon))]
    y_train = clip_target(train, clip_q)
    out = []
    for name in models:
        m = make_model(name, model_cfg.get(name, {}))
        m.fit(train[cols].fillna(0), y_train)
        pred = np.clip(m.predict(test[cols].fillna(0)), 0, None)
        if "is_open" in test:                     # mağaza takvimi önceden bilinir: kapalı gün = 0
            pred = pred * test["is_open"].to_numpy()
        out.append(pd.DataFrame({"item_id": test["item_id"].values, "date": test["date"].values,
                                 "y": test["qty"].values, "pred": pred, "model": name}))
    return pd.concat(out, ignore_index=True)


def summarize(preds: pd.DataFrame) -> pd.DataFrame:
    rows = [{"model": m, "fold": f, **all_metrics(g["y"], g["pred"])} for (m, f), g in preds.groupby(["model", "fold"])]
    return pd.DataFrame(rows).drop(columns="fold").groupby("model").mean().sort_values("MAE")


def backtest(df: pd.DataFrame, cfg: dict, models=ML_MODELS, offset: int = 0,
             baselines: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tüm katlarda tahminleri ve model bazında ortalama metrikleri döndürür."""
    fc = cfg["forecast"]
    h, k = fc["horizon"], fc["n_folds"]
    feat = add_features(df, fc["lags"], fc["rolling_windows"], h)
    preds = []
    for i, cut in enumerate(fold_cutoffs(df["date"], h, k, offset)):
        parts = [run_ml(feat, cut, h, cfg["models"], models, fc.get("target_clip_quantile"))]
        if baselines:
            parts.insert(0, run_baselines(df, cut, h))
        p = pd.concat(parts)
        p["fold"] = i
        preds.append(p)
    preds = pd.concat(preds, ignore_index=True)
    return preds, summarize(preds)
