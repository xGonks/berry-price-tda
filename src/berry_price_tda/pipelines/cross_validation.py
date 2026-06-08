"""
Validacion cruzada para series temporales (walk-forward / expanding window).

Usa sklearn.model_selection.TimeSeriesSplit, que respeta el orden temporal:
cada fold entrena con el pasado y valida con el futuro inmediato, sin fugas.

IMPORTANTE: cuando el objetivo es una diferencia (config.difference != "none"),
las metricas se calculan sobre el NIVEL reconstruido, no sobre la diferencia.
Asi las metricas son comparables entre modos (predecir nivel vs diferencia).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from ..features.builder import reconstruct_level
from ..utils.metrics import all_metrics


def cross_validate_model(
    model_factory: Callable[[], object],
    X: np.ndarray,
    y: np.ndarray,
    base: np.ndarray | None = None,
    n_splits: int = 5,
) -> dict[str, float]:
    """
    Evalua un modelo con TimeSeriesSplit y promedia las metricas sobre folds.

    Parameters
    ----------
    model_factory : callable
        Funcion sin argumentos que devuelve un estimador nuevo, no entrenado.
    X, y : np.ndarray
        Matriz de diseno y objetivo (nivel o diferencia segun config).
    base : np.ndarray | None
        Valor base para reconstruir el nivel cuando y es una diferencia.
        Si es None o todo ceros, se asume que y ya es el nivel.
    n_splits : int
        Numero de folds temporales.

    Returns
    -------
    dict con metricas promedio (mae, rmse, mape, r2) y su std entre folds.
    Las metricas se reportan SIEMPRE en la escala del nivel.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if base is None:
        base = np.zeros_like(y)
    base = np.asarray(base, dtype=float)

    max_splits = max(2, min(n_splits, len(X) // 12))
    tscv = TimeSeriesSplit(n_splits=max_splits)

    fold_metrics: list[dict[str, float]] = []

    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        base_te = base[test_idx]

        model = model_factory()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)

        # Reconstruir nivel para comparar de forma justa
        y_te_level = reconstruct_level(y_te, base_te)
        preds_level = reconstruct_level(preds, base_te)

        fold_metrics.append(all_metrics(y_te_level, preds_level))

    keys = fold_metrics[0].keys()
    summary: dict[str, float] = {}
    for k in keys:
        vals = np.array([fm[k] for fm in fold_metrics])
        summary[k] = float(np.mean(vals))
        summary[f"{k}_std"] = float(np.std(vals))

    summary["n_folds"] = len(fold_metrics)
    return summary
