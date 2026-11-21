"""Ham veriyi ürün × gün talep tablosuna çevirir.

Çıktı şeması: date | item_id | qty | price | is_open
"""
import numpy as np
import pandas as pd

CANCEL_WINDOW_DAYS = 90   # iptal, en fazla bu kadar gün önceki satışla eşleştirilir


def _match_cancellations(df: pd.DataFrame, window_days: int = CANCEL_WINDOW_DAYS) -> pd.DataFrame:
    """İptal faturalarını (C...) kendi satışlarıyla eşleştirip ikisini birlikte çıkarır.

    Eşleştirme: aynı müşteri + aynı ürün + aynı adet, satış iptalden ÖNCE ve en fazla
    `window_days` gün önce. Online Retail II'de iptal ile satış arası medyan ~11 gün olduğundan
    aynı gün şartı iptallerin çoğunu kaçırıyordu. Müşteri numarası olmayan iptaller için
    eski kural (aynı gün + ürün + adet) yedek olarak kullanılır.
    """
    df = df.reset_index(drop=True)
    df["_row"] = np.arange(len(df))
    is_cancel = df["invoice"].str.startswith("C")
    cancels = df[is_cancel].copy()
    cancels["qty"] = -cancels["qty"]
    sales = df[~is_cancel & (df["qty"] > 0)]
    drop_rows: set[int] = set()

    has_cust = "customer_id" in df and df["customer_id"].notna().any()
    if has_cust:
        c = cancels.dropna(subset=["customer_id"]).sort_values("ts")
        s = (sales.dropna(subset=["customer_id"]).sort_values("ts")
                  [["ts", "item_id", "customer_id", "qty", "_row"]].rename(columns={"ts": "sale_ts", "_row": "sale_row"}))
        m = pd.merge_asof(c, s, left_on="ts", right_on="sale_ts", by=["item_id", "customer_id", "qty"],
                          direction="backward", tolerance=pd.Timedelta(days=window_days))
        m = m.dropna(subset=["sale_row"]).drop_duplicates("sale_row")   # bir satış en fazla bir kez silinir
        drop_rows |= set(m["sale_row"].astype(int))
        rest = cancels[cancels["customer_id"].isna()]
    else:
        rest = cancels

    # Yedek: müşterisiz iptaller için aynı gün + ürün + adet
    key = ["item_id", "date", "qty"]
    remaining = sales[~sales["_row"].isin(drop_rows)].copy()
    rest = rest.copy()
    rest["_n"] = rest.groupby(key).cumcount()
    remaining["_n"] = remaining.groupby(key).cumcount()
    fb = remaining.merge(rest[key + ["_n"]], on=key + ["_n"], how="inner")
    drop_rows |= set(fb["_row"])

    keep = ~is_cancel & ~df["_row"].isin(drop_rows)
    return df[keep].drop(columns="_row")


def clean_online_retail(raw: pd.DataFrame, country: str = "United Kingdom", top_n: int = 50,
                        cancel_window_days: int = CANCEL_WINDOW_DAYS) -> pd.DataFrame:
    df = raw.rename(columns={"Invoice": "invoice", "InvoiceNo": "invoice", "StockCode": "item_id",
                             "Quantity": "qty", "InvoiceDate": "ts", "Price": "price",
                             "UnitPrice": "price", "Country": "country",
                             "Customer ID": "customer_id", "CustomerID": "customer_id"}).copy()
    df = df.drop_duplicates()                                  # tekrar eden satırlar
    df["invoice"] = df["invoice"].astype(str)
    df["item_id"] = df["item_id"].astype(str).str.strip()
    df["ts"] = pd.to_datetime(df["ts"])
    df["date"] = df["ts"].dt.normalize()

    # Mağazanın açık olduğu günler: o gün en az bir satış faturası kesilmiş mi (tüm ürün/ülkeler)
    open_days = pd.DatetimeIndex(df.loc[~df["invoice"].str.startswith("C"), "date"].unique())

    df = _match_cancellations(df, cancel_window_days)
    df = df[(df["qty"] > 0) & (df["price"] > 0)]
    if country and "country" in df:
        df = df[df["country"] == country]

    daily = (df.groupby(["item_id", "date"])
               .agg(qty=("qty", "sum"), price=("price", "mean"))
               .reset_index())
    top = daily.groupby("item_id")["qty"].sum().nlargest(top_n).index
    out = fill_missing_days(daily[daily["item_id"].isin(top)])
    out["is_open"] = out["date"].isin(open_days).astype(int)
    return out


def clean_store_item(raw: pd.DataFrame, store: int = 1) -> pd.DataFrame:
    df = raw[raw["store"] == store].rename(columns={"item": "item_id", "sales": "qty"})
    df["date"] = pd.to_datetime(df["date"])
    df["item_id"] = df["item_id"].astype(str)
    df["price"] = np.nan
    out = fill_missing_days(df[["item_id", "date", "qty", "price"]])
    out["is_open"] = 1
    return out


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
        rows.append(pd.DataFrame({"date": dates, "item_id": f"P{i:03d}", "qty": qty.astype(float),
                                  "price": price, "is_open": 1}))
    return pd.concat(rows, ignore_index=True)
