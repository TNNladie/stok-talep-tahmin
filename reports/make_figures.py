"""Rapor şekillerini results/ tablolarından üretir:  python reports/make_figures.py"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BLUE, GRAY, INK, MUTED, SURF = "#2a78d6", "#b9b8b3", "#1f1f1e", "#6b6a66", "#fcfcfb"
LABELS = {"lightgbm": "LightGBM", "xgboost": "XGBoost", "random_forest": "Random Forest", "ma_28": "MA-28",
          "ma_7": "MA-7", "persistence": "Persistence", "seasonal_naive_7": "Mevsimsel Naive (7)"}


def main() -> None:
    s = pd.read_csv(ROOT / "results" / "model_comparison.csv").sort_values("MAE", ascending=False)
    ml = {"lightgbm", "xgboost", "random_forest"}
    best_base = s[~s["model"].isin(ml)]["MAE"].min()
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURF)
    ax.set_facecolor(SURF)
    colors = [BLUE if m in ml else GRAY for m in s["model"]]
    ax.barh([LABELS[m] for m in s["model"]], s["MAE"], color=colors, height=0.6)
    for y, v in enumerate(s["MAE"]):
        ax.text(v + 0.6, y, f"{v:.1f}", va="center", fontsize=9, color=INK)
    ax.axvline(best_base * 0.9, color=INK, lw=1, ls="--")
    ax.text(best_base * 0.9, len(s) - 0.4, "  başarı eşiği (MA-28 − %10)", fontsize=8, color=MUTED, va="center")
    ax.set_xlabel("MAE (adet/gün) — düşük daha iyi", color=MUTED, fontsize=9)
    ax.set_title("Online Retail II · 3 test katı × 28 gün · mavi = ML, gri = baseline",
                 loc="left", fontsize=10, color=INK)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(GRAY)
    ax.spines["bottom"].set_color(GRAY)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color="#e8e7e3", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = ROOT / "reports" / "figures" / "model_comparison_online_retail.png"
    fig.savefig(out, dpi=150)
    print("Kaydedildi:", out)


if __name__ == "__main__":
    main()
