"""Ham veriyi ürün × gün talep tablosuna çevirir.

Çıktı şeması: date | item_id | qty | price
"""
import numpy as np
import pandas as pd


def clean_online_retail(raw: pd.DataFrame, country: str = "United Kingdom", top_n: int = 50) -> pd.DataFrame:
    df = raw.rename(columns={"Invoice": "invoice", "InvoiceNo": "invoice", "StockCode": "item_id",
                             "Quantity": "qty", "InvoiceDate": "date", "Price": "price",
                             "UnitPrice": "price", "Country": "country"}).copy()
    df = df.drop_duplicates()                                  # YENİ: tekrar eden satırlar
    df["invoice"] = df["invoice"].astype(str)
    df["item_id"] = df["item_id"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # YENİ: iptal edilen satışı, iptal faturasıyla birlikte çıkar
    is_cancel = df["invoice"].str.startswith("C")
    cancels = df[is_cancel].copy()
    cancels["qty"] = -cancels["qty"]
    sales = df[~is_cancel].copy()
    key = ["item_id", "date", "qty"]
    cancels["_n"] = cancels.groupby(key).cumcount()
    sales["_n"] = sales.groupby(key).cumcount()
    merged = sales.merge(cancels[key + ["_n"]], on=key + ["_n"], how="left", indicator=True)
    df = merged[merged["_merge"] == "left_only"].drop(columns=["_n", "_merge"])

    df = df[(df["qty"] > 0) & (df["price"] > 0)]
    if country and "country" in df:
        df = df[df["country"] == country]

    daily = (df.groupby(["item_id", "date"])
               .agg(qty=("qty", "sum"), price=("price", "mean"))
               .reset_index())
    top = daily.groupby("item_id")["qty"].sum().nlargest(top_n).index
    return fill_missing_days(daily[daily["item_id"].isin(top)])


def clean_store_item(raw: pd.DataFrame, store: int = 1) -> pd.DataFrame:
    df = raw[raw["store"] == store].rename(columns={"item": "item_id", "sales": "qty"})
    df["date"] = pd.to_datetime(df["date"])
    df["item_id"] = df["item_id"].astype(str)
    df["price"] = np.nan
    return fill_missing_days(df[["item_id", "date", "qty", "price"]])


def fill_missing_days(daily: pd.DataFrame) -> pd.DataFrame:
    """Satış olmayan günleri qty=0 ile doldurur; fiyatı ileri taşır."""
    full_dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    out = []
    for item, g in daily.groupby("item_id"):
        g = g.set_index("date").reindex(full_dates)
        g["item_id"] = item
        g["qty"] = g["qty"].fillna(0)
        g["price"] = g["price"].ffill().bfill()
        out.append(g.rename_axis("date").reset_index())
    return pd.concat(out, ignore_index=True)[["date", "item_id", "qty", "price"]]


def make_demo_data(n_items: int = 10, days: int = 730, seed: int = 42) -> pd.DataFrame:
    """İnternet olmadan test için sentetik, mevsimsel ve aralıklı talep verisi."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    t = np.arange(days)
    rows = []
    for i in range(n_items):
        base = rng.uniform(3, 30)
        weekly = 1 + 0.3 * np.sin(2 * np.pi * (t % 7) / 7 + i)
        yearly = 1 + 0.25 * np.sin(2 * np.pi * t / 365)
        trend = 1 + rng.uniform(-0.2, 0.4) * t / days
        lam = base * weekly * yearly * trend
        qty = rng.poisson(lam)
        if i % 3 == 0:                                   # bazı ürünler aralıklı
            qty = qty * (rng.random(days) > 0.6)
        price = round(rng.uniform(1, 20), 2)
        rows.append(pd.DataFrame({"date": dates, "item_id": f"P{i:03d}", "qty": qty.astype(float), "price": price}))
    return pd.concat(rows, ignore_index=True)
