"""Basit günlük stok simülasyonu: politika karşılaştırması (Sprint 9).

Her gün: teslimat gelir -> talep karşılanır -> R günde bir stok gözden geçirilir.
"""
import numpy as np


def simulate(demand: np.ndarray, order_fn, start_stock: float, lead_time: int = 7, review_period: int = 7) -> dict:
    stock, pipeline = start_stock, {}
    stockout_days, lost, levels = 0, 0.0, []
    for t, d in enumerate(demand):
        stock += pipeline.pop(t, 0.0)
        sold = min(stock, d)
        if d > stock:
            stockout_days += 1
            lost += d - stock
        stock -= sold
        if t % review_period == 0:
            q = order_fn(t, stock + sum(pipeline.values()))
            if q > 0:
                pipeline[t + lead_time] = pipeline.get(t + lead_time, 0.0) + q
        levels.append(stock)
    return {"stockout_days": stockout_days, "lost_units": round(lost, 1),
            "avg_stock": round(float(np.mean(levels)), 1)}
