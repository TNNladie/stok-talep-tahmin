"""Syntetos–Boylan talep sınıflandırması (ADI / CV²) ve segment bazında metrikler."""
import numpy as np
import pandas as pd

from src.evaluation.metrics import all_metrics


def classify(df: pd.DataFrame, adi_cut: float = 1.32, cv2_cut: float = 0.49) -> pd.DataFrame:
    """Ürün başına ADI (satışlar arası ort. gün) ve CV² (sıfır olmayan talebin varyasyonu)."""
    rows = []
    for item, g in df.groupby("item_id"):
        q = g.loc[g["is_open"] == 1, "qty"] if "is_open" in g else g["qty"]   # kapalı günler sayılmaz
        nz = q[q > 0]
        adi = len(q) / len(nz) if len(nz) else np.inf
        cv2 = (nz.std() / nz.mean()) ** 2 if len(nz) > 1 else 0.0
        if adi < adi_cut:
            seg = "smooth" if cv2 < cv2_cut else "erratic"
        else:
            seg = "intermittent" if cv2 < cv2_cut else "lumpy"
        rows.append({"item_id": item, "ADI": adi, "CV2": cv2, "segment": seg})
    return pd.DataFrame(rows)


def segment_metrics(preds: pd.DataFrame, seg: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    p = preds[preds["model"].isin(models)].merge(seg[["item_id", "segment"]], on="item_id")
    rows = [{"segment": s, "model": m, "n_items": g["item_id"].nunique(), **all_metrics(g["y"], g["pred"])}
            for (s, m), g in p.groupby(["segment", "model"])]
    return pd.DataFrame(rows)
