"""Uçtan uca akış: veri -> temizleme -> backtest -> sonuç tabloları.

    python -m src.pipeline            # config.yaml'daki kaynak
    python -m src.pipeline --demo     # internetsiz sentetik veri
"""
import argparse

import pandas as pd

from src.config import ROOT, load_config
from src.data.clean import clean_online_retail, clean_store_item, make_demo_data
from src.evaluation.backtest import backtest


def load_daily(cfg: dict, demo: bool) -> pd.DataFrame:
    if demo or cfg["data"]["source"] == "demo":
        return make_demo_data()
    raw_dir = ROOT / cfg["data"]["raw_dir"]
    if cfg["data"]["source"] == "online_retail_ii":
        raw = pd.read_parquet(raw_dir / "online_retail_ii.parquet")
        return clean_online_retail(raw, cfg["data"]["country"], cfg["data"]["top_n_items"])
    return clean_store_item(pd.read_csv(raw_dir / "train.csv"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    cfg = load_config()

    daily = load_daily(cfg, args.demo)
    out = ROOT / cfg["data"]["processed_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out, index=False)
    print(f"Günlük tablo: {len(daily):,} satır, {daily['item_id'].nunique()} ürün")

    preds, summary = backtest(daily, cfg)
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    preds.to_parquet(res / "predictions.parquet", index=False)
    summary.round(3).to_csv(res / "model_comparison.csv")
    print("\nModel karşılaştırması (katların ortalaması):\n", summary.round(3))


if __name__ == "__main__":
    main()
