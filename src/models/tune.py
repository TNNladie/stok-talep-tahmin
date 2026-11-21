"""Hiperparametre ayarı (rastgele arama) — yalnızca DOĞRULAMA katlarında.

Doğrulama katları, raporlanan test katlarından hemen önceki `n_folds` kattır
(backtest offset=n_folds). Böylece test katları ayar sırasında hiç görülmez.

    python -m src.models.tune                 # tüm ML modelleri
    python -m src.models.tune --trials 40 --models lightgbm
Çıktı: configs/tuned_params.yaml  +  results/tuning_trials.csv
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import yaml

from src.config import ROOT, load_config
from src.evaluation.backtest import backtest

warnings.filterwarnings("ignore")

SPACE = {
    "lightgbm": {
        "n_estimators": [300, 600, 1000, 1500], "learning_rate": [0.01, 0.02, 0.05, 0.1],
        "num_leaves": [15, 31, 63, 127], "min_child_samples": [10, 20, 50, 100],
        "subsample": [0.7, 0.8, 1.0], "subsample_freq": [1], "colsample_bytree": [0.5, 0.7, 0.9],
        "reg_lambda": [0.0, 1.0, 5.0], "tweedie_variance_power": [1.1, 1.3, 1.5, 1.7], "verbose": [-1],
    },
    "xgboost": {
        "n_estimators": [300, 600, 1000], "learning_rate": [0.02, 0.05, 0.1],
        "max_depth": [4, 6, 8], "min_child_weight": [1, 5, 10], "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.5, 0.7, 0.9], "reg_lambda": [1.0, 5.0], "tweedie_variance_power": [1.1, 1.3, 1.5, 1.7],
    },
    "random_forest": {
        "n_estimators": [300], "max_depth": [8, 12, 16, None], "min_samples_leaf": [1, 5, 10, 20],
        "max_features": [0.33, 0.5, 1.0], "n_jobs": [-1],
    },
}
DEFAULT_TRIALS = {"lightgbm": 30, "xgboost": 20, "random_forest": 8}


def sample(space: dict, rng: np.random.Generator) -> dict:
    return {k: (v[rng.integers(len(v))]) for k, v in space.items()}


def _py(v):
    return v.item() if hasattr(v, "item") else v


def tune(df: pd.DataFrame, cfg: dict, models: list[str], trials: dict, seed: int = 42) -> tuple[dict, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    k = cfg["forecast"]["n_folds"]
    best, log = {}, []
    for name in models:
        candidates = [cfg["models"].get(name, {})] + [sample(SPACE[name], rng) for _ in range(trials[name])]
        scores = []
        for i, params in enumerate(candidates):
            params = {p: _py(v) for p, v in params.items()}
            c = {**cfg, "models": {**cfg["models"], name: params}}
            _, s = backtest(df, c, models=[name], offset=k, baselines=False)
            mae, wape = float(s.loc[name, "MAE"]), float(s.loc[name, "WAPE"])
            scores.append((mae, params))
            log.append({"model": name, "trial": i, "MAE": mae, "WAPE": wape, "params": params})
            print(f"{name:14s} deneme {i:2d}  MAE={mae:7.3f}  WAPE={wape:6.2f}", flush=True)
        best[name] = min(scores, key=lambda t: t[0])[1]
    return best, pd.DataFrame(log)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(SPACE))
    ap.add_argument("--trials", type=int, default=None, help="tüm modeller için deneme sayısı")
    args = ap.parse_args()
    cfg = load_config(tuned=False)
    df = pd.read_parquet(ROOT / cfg["data"]["processed_path"])
    trials = {m: args.trials or DEFAULT_TRIALS[m] for m in args.models}
    best, log = tune(df, cfg, args.models, trials)

    out = ROOT / "configs" / "tuned_params.yaml"
    existing = yaml.safe_load(out.read_text(encoding="utf-8")) if out.exists() else {}
    existing = (existing or {}) | best
    out.write_text("# python -m src.models.tune ile doğrulama katlarında seçildi — elle düzenlemeyin\n"
                   + yaml.safe_dump(existing, allow_unicode=True, sort_keys=True), encoding="utf-8")
    (ROOT / "results").mkdir(exist_ok=True)
    log.to_csv(ROOT / "results" / "tuning_trials.csv", index=False)
    print("\nEn iyi parametreler ->", out)


if __name__ == "__main__":
    main()
