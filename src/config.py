"""Ayar dosyasını okur.

configs/tuned_params.yaml varsa (python -m src.models.tune üretir), oradaki model
parametreleri config.yaml'daki varsayılanların yerine geçer.
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TUNED_PATH = ROOT / "configs" / "tuned_params.yaml"


def load_config(path: str | Path = ROOT / "configs" / "config.yaml", tuned: bool = True) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if tuned and TUNED_PATH.exists():
        best = yaml.safe_load(TUNED_PATH.read_text(encoding="utf-8")) or {}
        for name, params in best.items():
            cfg["models"][name] = params
    return cfg
