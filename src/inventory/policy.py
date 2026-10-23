"""Tahminden yeniden sipariş önerisine.

SS  = z * sigma_e * sqrt(L)
ROP = D_L + SS
Q   = max(0, D_{L+R} + SS - eldeki_stok)   (yalnızca stok <= ROP ise)
"""
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class Recommendation:
    safety_stock: float
    reorder_point: float
    order_qty: float
    should_order: bool


def recommend(forecast: np.ndarray, residual_std: float, on_hand: float,
              lead_time: int = 7, review_period: int = 7, service_level: float = 0.95) -> Recommendation:
    z = norm.ppf(service_level)
    ss = z * residual_std * np.sqrt(lead_time)
    d_l = float(np.sum(forecast[:lead_time]))
    d_lr = float(np.sum(forecast[:lead_time + review_period]))
    rop = d_l + ss
    order = on_hand <= rop
    q = max(0.0, d_lr + ss - on_hand) if order else 0.0
    return Recommendation(round(ss, 1), round(rop, 1), float(np.ceil(q)), bool(order))
