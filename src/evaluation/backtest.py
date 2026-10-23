"""Zaman bazlı genişleyen pencere doğrulaması.

Kat k için: eğitim = test başlangıcından önceki tüm günler, test = sonraki `horizon` gün.
Katlar sondan geriye doğru dizilir; rastgele bölme YOKTUR.
"""
import numpy as np
import pandas as pd

from src.evaluation.metrics import all_metrics
from src.features.build import add_features, feature_columns
from src.models.baselines import BASELINES
from src.models.ml import ML_MODELS, make_model


def fold_cutoffs(dates: pd.Series, horizon: int, n_folds: int) -> list[pd.Timestamp]:
    last = dates.max()
    return [last - pd.Timedelta(days=horizon * k) for k in range(n_folds, 0, -1)]


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


def run_ml(feat: pd.DataFrame, cutoff: pd.Timestamp, horizon: int, model_cfg: dict, models=ML_MODELS) -> pd.DataFrame:
    cols = feature_columns(feat)
    train = feat[feat["date"] <= cutoff].dropna(subset=[c for c in cols if c.startswith("lag_")])
    test = feat[(feat["date"] > cutoff) & (feat["date"] <= cutoff + pd.Timedelta(days=horizon))]
    out = []
    for name in models:
        m = make_model(name, model_cfg.get(name, {}))
        m.fit(train[cols].fillna(0), train["qty"])
        pred = np.clip(m.predict(test[cols].fillna(0)), 0, None)
        out.append(pd.DataFrame({"item_id": test["item_id"].values, "date": test["date"].values,
                                 "y": test["qty"].values, "pred": pred, "model": name}))
    return pd.concat(out, ignore_index=True)


def backtest(df: pd.DataFrame, cfg: dict, models=ML_MODELS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tüm katlarda tahminleri ve model bazında ortalama metrikleri döndürür."""
    h, k = cfg["forecast"]["horizon"], cfg["forecast"]["n_folds"]
    feat = add_features(df, cfg["forecast"]["lags"], cfg["forecast"]["rolling_windows"], h)
    preds = []
    for i, cut in enumerate(fold_cutoffs(df["date"], h, k)):
        p = pd.concat([run_baselines(df, cut, h), run_ml(feat, cut, h, cfg["models"], models)])
        p["fold"] = i
        preds.append(p)
    preds = pd.concat(preds, ignore_index=True)
    summary = (preds.groupby(["model", "fold"]).apply(lambda g: pd.Series(all_metrics(g["y"], g["pred"])))
                    .groupby("model").mean().sort_values("MAE"))
    return preds, summary
