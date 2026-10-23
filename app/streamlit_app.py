"""Stok karar destek ekranı.  Çalıştırma: streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import ROOT, load_config  # noqa: E402
from src.data.clean import make_demo_data  # noqa: E402
from src.evaluation.backtest import run_ml  # noqa: E402
from src.features.build import add_features  # noqa: E402
from src.inventory.policy import recommend  # noqa: E402

st.set_page_config(page_title="Stok Talep Tahmini", layout="wide")
cfg = load_config()
H = cfg["forecast"]["horizon"]

st.title("Stok Talep Tahmin ve Sipariş Önerisi")
up = st.sidebar.file_uploader("Satış verisi (CSV: date,item_id,qty,price)", type="csv")
processed = ROOT / cfg["data"]["processed_path"]
if up is not None:
    daily = pd.read_csv(up, parse_dates=["date"])
elif processed.exists():
    daily = pd.read_parquet(processed)
else:
    daily = make_demo_data()
    st.sidebar.info("Demo verisi kullanılıyor.")
daily["item_id"] = daily["item_id"].astype(str)

item = st.sidebar.selectbox("Ürün", sorted(daily["item_id"].unique()))
on_hand = st.sidebar.number_input("Eldeki stok (adet)", min_value=0, value=50)
L = st.sidebar.slider("Tedarik süresi (gün)", 1, 21, cfg["inventory"]["lead_time_days"])
sl = st.sidebar.slider("Hizmet düzeyi", 0.80, 0.99, cfg["inventory"]["service_level"])


@st.cache_data
def forecast_all(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Son tarihten sonraki H gün için LightGBM tahmini + doğrulama artıkları."""
    last = df["date"].max()
    future = pd.DataFrame([(last + pd.Timedelta(days=i), it) for it in df["item_id"].unique() for i in range(1, H + 1)],
                          columns=["date", "item_id"])
    future["qty"], future["price"] = np.nan, np.nan
    full = pd.concat([df, future], ignore_index=True)
    feat = add_features(full, cfg["forecast"]["lags"], cfg["forecast"]["rolling_windows"], H)
    fc = run_ml(feat, last, H, cfg["models"], ["lightgbm"])
    val = run_ml(add_features(df, cfg["forecast"]["lags"], cfg["forecast"]["rolling_windows"], H),
                 last - pd.Timedelta(days=H), H, cfg["models"], ["lightgbm"])
    return fc, val


fc, val = forecast_all(daily)
f_item = fc[fc["item_id"] == item].sort_values("date")
resid_std = float((val[val["item_id"] == item]["y"] - val[val["item_id"] == item]["pred"]).std())
rec = recommend(f_item["pred"].to_numpy(), resid_std, on_hand, L, cfg["inventory"]["review_period_days"], sl)

c1, c2, c3, c4 = st.columns(4)
c1.metric("28 gün tahmini talep", f"{f_item['pred'].sum():.0f}")
c2.metric("Güvenlik stoku", f"{rec.safety_stock:.0f}")
c3.metric("Yeniden sipariş noktası", f"{rec.reorder_point:.0f}")
c4.metric("Önerilen sipariş", f"{rec.order_qty:.0f}" if rec.should_order else "Gerek yok")

hist = daily[daily["item_id"] == item].tail(120)
fig = go.Figure()
fig.add_scatter(x=hist["date"], y=hist["qty"], name="Gerçek satış")
fig.add_scatter(x=f_item["date"], y=f_item["pred"], name="Tahmin (LightGBM)", line=dict(dash="dash"))
fig.update_layout(height=420, xaxis_title="Tarih", yaxis_title="Adet")
st.plotly_chart(fig, use_container_width=True)
if rec.should_order:
    st.success(f"Eldeki stok ({on_hand}) yeniden sipariş noktasının ({rec.reorder_point:.0f}) altında: "
               f"**{rec.order_qty:.0f} adet sipariş verin.**")
else:
    st.info("Stok yeterli; bu gözden geçirmede sipariş gerekmiyor.")
