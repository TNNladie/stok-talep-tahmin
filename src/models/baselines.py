"""Baseline modeller. Her fonksiyon: (geçmiş seri, ufuk) -> tahmin dizisi."""
import numpy as np


def persistence(history: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(history[-1], horizon).astype(float)


def seasonal_naive(history: np.ndarray, horizon: int, season: int = 7) -> np.ndarray:
    last = history[-season:]
    return np.resize(last, horizon).astype(float)


def moving_average(history: np.ndarray, horizon: int, window: int = 7) -> np.ndarray:
    return np.repeat(history[-window:].mean(), horizon)


BASELINES = {
    "persistence": persistence,
    "seasonal_naive_7": seasonal_naive,
    "ma_7": lambda h, n: moving_average(h, n, 7),
    "ma_28": lambda h, n: moving_average(h, n, 28),
}
