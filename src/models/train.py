"""Final model eğitimi ve kaydı.

Seçilen model (varsayılan: LightGBM) TÜM geçmiş veriyle yeniden eğitilir ve
models/ altına kaydedilir. Güvenlik stoku için gereken ürün bazlı artık standart
sapması (sigma_e), backtest test katlarındaki tahmin hatalarından hesaplanır.

    python -m src.models.train                  # results/predictions.parquet'ten sigma_e alır
    python -m src.models.train --model xgboost
Çıktı: models/<model>.pkl  +  models/<model>_meta.json
"""
import argparse
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from src.config import ROOT, load_config
from src.evaluation.backtest import clip_target, train_matrix
from src.features.build import add_features, feature_columns, make_future_frame
from src.models.ml import make_model

MODEL_DIR = ROOT / "models"


def residual_std(preds: pd.DataFrame, model: str) -> dict[str, float]:
    p = preds[preds["model"] == model]
    return {str(k): float(v) for k, v in (p["y"] - p["pred"]).groupby(p["item_id"]).std().fillna(0).items()}


def fit_final(df: pd.DataFrame, cfg: dict, model: str = "lightgbm"):
    fc = cfg["forecast"]
    feat = add_features(df, fc["lags"], fc["rolling_windows"], fc["horizon"])
    cols = feature_columns(feat)
    train = train_matrix(feat, df["date"].max())
    m = make_model(model, cfg["models"].get(model, {}))
    m.fit(train[cols].fillna(0), clip_target(train, fc.get("target_clip_quantile")))
    return m, cols


def predict_future(m, df: pd.DataFrame, cfg: dict, cols: list[str]) -> pd.DataFrame:
    fc = cfg["forecast"]
    full = make_future_frame(df, fc["horizon"])
    feat = add_features(full, fc["lags"], fc["rolling_windows"], fc["horizon"])
    fut = feat[feat["date"] > df["date"].max()]
    pred = np.clip(m.predict(fut[cols].fillna(0)), 0, None) * fut["is_open"].to_numpy()
    return pd.DataFrame({"date": fut["date"].values, "item_id": fut["item_id"].values, "pred": pred})


def save(m, meta: dict, model: str) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(m, MODEL_DIR / f"{model}.pkl")
    (MODEL_DIR / f"{model}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load(model: str = "lightgbm"):
    path = MODEL_DIR / f"{model}.pkl"
    if not path.exists():
        return None, None
    meta = json.loads((MODEL_DIR / f"{model}_meta.json").read_text(encoding="utf-8"))
    return joblib.load(path), meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lightgbm")
    args = ap.parse_args()
    cfg = load_config()
    df = pd.read_parquet(ROOT / cfg["data"]["processed_path"])
    df["item_id"] = df["item_id"].astype(str)

    m, cols = fit_final(df, cfg, args.model)
    preds_path = ROOT / "results" / "predictions.parquet"
    sig = residual_std(pd.read_parquet(preds_path), args.model) if preds_path.exists() else {}
    meta = {
        "model": args.model, "trained_at": datetime.now().isoformat(timespec="seconds"),
        "data_source": cfg["data"]["source"], "train_start": str(df["date"].min().date()),
        "train_end": str(df["date"].max().date()), "n_items": int(df["item_id"].nunique()),
        "items": sorted(df["item_id"].unique().tolist()), "features": cols,
        "params": cfg["models"].get(args.model, {}), "forecast": cfg["forecast"], "residual_std": sig,
    }
    save(m, meta, args.model)
    fut = predict_future(m, df, cfg, cols)
    fut.to_csv(ROOT / "results" / f"forecast_next_{cfg['forecast']['horizon']}d.csv", index=False)
    print(f"Kaydedildi: models/{args.model}.pkl  ({len(cols)} öznitelik, {meta['n_items']} ürün, "
          f"{meta['train_start']} → {meta['train_end']})")
    print(f"Gelecek {cfg['forecast']['horizon']} gün tahmini: results/forecast_next_{cfg['forecast']['horizon']}d.csv")


if __name__ == "__main__":
    main()
