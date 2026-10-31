import numpy as np
import pandas as pd

from src.config import load_config
from src.data.clean import clean_online_retail, make_demo_data
from src.evaluation.backtest import backtest
from src.evaluation.metrics import mae, mape, wape
from src.inventory.policy import recommend
from src.inventory.simulate import simulate
from src.models.baselines import moving_average, seasonal_naive


def test_metrics():
    assert mae([1, 2, 3], [1, 2, 4]) == 1 / 3
    assert np.isclose(mape([0, 10], [5, 5]), 50.0)   # y=0 günü dışarıda
    assert np.isclose(wape([10, 10], [5, 15]), 50.0)


def test_baselines():
    h = np.array([1, 2, 3, 4, 5, 6, 7, 8.0])
    assert seasonal_naive(h, 3).tolist() == [2, 3, 4]
    assert moving_average(h, 2, window=2).tolist() == [7.5, 7.5]


def test_clean_drops_returns_and_fills_days():
    raw = pd.DataFrame({
        "Invoice": ["0", "1", "C2", "3"], "StockCode": ["A", "A", "A", "A"],
        "Quantity": [3, 5, -5, 2],
        "InvoiceDate": ["2010-01-01 08:00", "2010-01-01 10:00", "2010-01-01 11:00", "2010-01-03 09:00"],
        "Price": [1.0, 1.0, 1.0, 1.0], "Country": ["United Kingdom"] * 4,
    })
    out = clean_online_retail(raw)
    # 5 adetlik satış ve iptali birlikte silinir, 3 kalır; 2 Ocak boş gün 0 ile dolar
    assert out["qty"].tolist() == [3, 0, 2]


def test_policy_orders_when_low():
    r = recommend(np.full(28, 10.0), residual_std=3, on_hand=20, lead_time=7, review_period=7)
    assert r.should_order and r.order_qty > 0
    r2 = recommend(np.full(28, 10.0), residual_std=3, on_hand=500)
    assert not r2.should_order


def test_simulation_runs():
    res = simulate(np.full(60, 5.0), lambda t, pos: 50 if pos < 40 else 0, start_stock=40)
    assert set(res) == {"stockout_days", "lost_units", "avg_stock"}


def test_backtest_end_to_end():
    cfg = load_config()
    cfg["forecast"]["n_folds"] = 1
    cfg["models"]["random_forest"]["n_estimators"] = 20
    cfg["models"]["xgboost"]["n_estimators"] = 30
    cfg["models"]["lightgbm"]["n_estimators"] = 30
    _, summary = backtest(make_demo_data(n_items=3, days=300), cfg)
    assert {"persistence", "lightgbm", "xgboost", "random_forest"} <= set(summary.index)
