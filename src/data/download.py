"""Ham veri setlerini indirir.

Kullanım:
    python -m src.data.download --source online_retail_ii
    python -m src.data.download --source store_item   # Kaggle CLI + API anahtarı gerekir
"""
import argparse
import subprocess
from pathlib import Path

from src.config import ROOT, load_config


def download_online_retail_ii(raw_dir: Path) -> Path:
    """UCI Online Retail II (id=502) → data/raw/online_retail_ii.parquet"""
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=502)
    df = ds.data.original if ds.data.original is not None else ds.data.features
    out = raw_dir / "online_retail_ii.parquet"
    df.to_parquet(out, index=False)
    return out


def download_store_item(raw_dir: Path) -> Path:
    """Kaggle Store Item Demand Forecasting Challenge."""
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", "demand-forecasting-kernels-only", "-p", str(raw_dir)],
        check=True,
    )
    subprocess.run(["unzip", "-o", str(raw_dir / "demand-forecasting-kernels-only.zip"), "-d", str(raw_dir)], check=True)
    return raw_dir / "train.csv"


def main() -> None:
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=cfg["data"]["source"])
    args = p.parse_args()
    raw_dir = ROOT / cfg["data"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    if args.source == "online_retail_ii":
        print("İndirildi:", download_online_retail_ii(raw_dir))
    elif args.source == "store_item":
        print("İndirildi:", download_store_item(raw_dir))
    else:
        print("demo kaynağı indirme gerektirmez.")


if __name__ == "__main__":
    main()
