"""
Evaluacion unificada de una configuracion completa de forecasting.

Una "configuracion" combina TODOS los factores:
    modelo, ventana, transformacion, subconjunto de exogenas, TDA si/no,
    estrategia anti-anomalia.

Esta funcion es la base compartida por:
    - la busqueda con Optuna (optimiza sobre VALIDACION)
    - el factorial dirigido (evalua sobre TEST los mejores modelos)

Metodologia:
    - Las features TDA se calculan sobre la ventana en NIVEL (teoria de Takens)
      y se sanean con estadisticos del train.
    - El objetivo se predice en la escala transformada y se reconstruye a nivel
      para medir.
    - Walk-forward con valores reales en el tramo de evaluacion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.loader import EXOGENOUS_COLS, TARGET
from ..features.anomaly import apply_strategy
from ..features.tda import (
    TDA_FEATURE_NAMES, extract_tda_features, gtda_available, sanitize_tda_matrix,
)
from ..features.transforms import Transformer
from ..models.sklearn_models import get_sklearn_model
from ..utils.metrics import all_metrics


@dataclass
class RunConfig:
    """Configuracion completa de un experimento puntual."""
    model_name: str
    window: int
    transform: str = "none"
    exog_cols: list[str] = field(default_factory=list)
    use_tda: bool = False
    strategy: str = "none"
    embedding_dim: int = 6
    time_delay: int = 2

    def label(self) -> str:
        ex = "+".join(c[:4] for c in self.exog_cols) if self.exog_cols else "noexog"
        parts = [self.model_name, f"w{self.window}", self.transform, ex]
        if self.use_tda:
            parts.append("tda")
        if self.strategy != "none":
            parts.append(self.strategy)
        return "|".join(parts)

    def to_row(self) -> dict:
        return {
            "modelo": self.model_name, "window": self.window,
            "transform": self.transform,
            "exog": "+".join(self.exog_cols) if self.exog_cols else "none",
            "n_exog": len(self.exog_cols),
            "use_tda": self.use_tda, "estrategia": self.strategy,
        }


def _build_design(series, exog, window, transform, use_tda,
                  n_eval, embedding_dim, time_delay, n_fit):
    """
    Construye (X_fit, y_fit, X_eval, idx_eval, transformer, offset) para
    forecasting t+1. `n_fit` es cuantas observaciones (en nivel) se usan para
    ajustar; el resto hasta n_eval son el tramo de evaluacion.

    series : serie objetivo COMPLETA en nivel (fit + eval).
    exog   : matriz exogena COMPLETA alineada con series, o None.
    """
    tr = Transformer(transform)
    z = tr.fit_transform(series)
    offset = len(series) - len(z)
    n_z_fit = n_fit - offset

    # Ventanas de ajuste
    Xl, y, idx_fit = [], [], []
    for i in range(n_z_fit - window):
        Xl.append(z[i:i + window]); y.append(z[i + window]); idx_fit.append(i + window)
    if len(Xl) < 10:
        raise ValueError("Pocas muestras de ajuste")
    Xl = np.array(Xl); y = np.array(y)

    blocks_fit = [Xl]
    # Exogenas (valor en el instante del objetivo)
    if exog is not None:
        ex_fit = np.array([exog[idx + offset] for idx in idx_fit])
        blocks_fit.append(ex_fit)

    # TDA del ajuste (ventana en NIVEL)
    if use_tda:
        tda_fit = np.zeros((len(idx_fit), len(TDA_FEATURE_NAMES)))
        for r, idx in enumerate(idx_fit):
            lvl_end = idx + offset
            win_lvl = series[max(0, lvl_end - window):lvl_end]
            tda_fit[r] = extract_tda_features(win_lvl, embedding_dim, time_delay)

    # Ventanas de evaluacion
    Xl_e, idx_eval = [], []
    for k in range(n_eval - n_fit):
        pos = n_z_fit + k
        if pos - window < 0 or pos >= len(z):
            continue
        Xl_e.append(z[pos - window:pos]); idx_eval.append(pos)
    Xl_e = np.array(Xl_e)

    blocks_eval = [Xl_e]
    if exog is not None:
        ex_ev = np.array([exog[pos + offset] for pos in idx_eval])
        blocks_eval.append(ex_ev)

    if use_tda:
        tda_ev = np.zeros((len(idx_eval), len(TDA_FEATURE_NAMES)))
        for r, pos in enumerate(idx_eval):
            lvl_end = pos + offset
            win_lvl = series[max(0, lvl_end - window):lvl_end]
            tda_ev[r] = extract_tda_features(win_lvl, embedding_dim, time_delay)
        tda_fit_s, tda_ev_s, kept = sanitize_tda_matrix(tda_fit, tda_ev)
        if kept:
            blocks_fit.append(tda_fit_s); blocks_eval.append(tda_ev_s)

    X_fit = np.hstack(blocks_fit)
    X_eval = np.hstack(blocks_eval)
    return X_fit, y, X_eval, idx_eval, tr, offset, z


def evaluate_config(
    df_fit: pd.DataFrame,
    df_eval: pd.DataFrame,
    config: RunConfig,
) -> dict[str, float]:
    """
    Ajusta con df_fit y evalua sobre df_eval (validacion o test).

    df_fit y df_eval deben ser tramos CONTIGUOS en el tiempo (df_eval va
    inmediatamente despues de df_fit), porque las ventanas de evaluacion usan
    las ultimas observaciones de df_fit.
    """
    if config.use_tda and not gtda_available():
        raise RuntimeError("use_tda=True pero giotto-tda no esta disponible.")

    # Aplicar estrategia anti-anomalia al tramo de ajuste
    df_fit_t, _dummy = apply_strategy(df_fit, config.strategy, TARGET)
    df_full = pd.concat([df_fit_t, df_eval])

    series = df_full[TARGET].to_numpy(dtype=float)
    n_fit = len(df_fit_t)
    n_eval = len(df_full)

    exog = None
    if config.exog_cols:
        present = [c for c in config.exog_cols if c in df_full.columns]
        if present:
            exog = df_full[present].to_numpy(dtype=float)

    X_fit, y_fit, X_eval, idx_eval, tr, offset, z = _build_design(
        series, exog, config.window, config.transform, config.use_tda,
        n_eval, config.embedding_dim, config.time_delay, n_fit,
    )

    model = get_sklearn_model(config.model_name)
    model.fit(X_fit, y_fit)

    preds, acts = [], []
    for r, pos in enumerate(idx_eval):
        z_hat = float(model.predict(X_eval[r:r + 1])[0])
        z_seq = np.append(z[:pos], z_hat)
        rec = tr.inverse_transform(z_seq, history=series[:offset] if offset else None)
        preds.append(rec[-1]); acts.append(series[pos + offset])

    if not preds:
        raise ValueError("Sin predicciones (revisar tamanos de ventana / tramos).")

    return all_metrics(np.array(acts), np.array(preds))


def count_features(df_fit: pd.DataFrame, config: RunConfig) -> int:
    """
    Cuenta el numero de features que generaria esta config (lags + exog + tda).
    Util para penalizar la complejidad en la busqueda.
    """
    n = config.window                       # lags
    if config.exog_cols:
        present = [c for c in config.exog_cols if c in df_fit.columns]
        n += len(present)                   # exogenas
    if config.use_tda:
        n += len(TDA_FEATURE_NAMES)         # features topologicas (antes de sanear)
    return n


def evaluate_config_cv(
    df: pd.DataFrame,
    config: RunConfig,
    n_splits: int = 4,
    metric: str = "mae",
) -> dict[str, float]:
    """
    Evalua una config con TimeSeriesSplit (walk-forward sobre varios folds).

    A diferencia de evaluate_config (un solo tramo fit->eval), esto promedia
    el desempeno sobre n_splits cortes temporales, dando una estimacion mucho
    mas robusta de la capacidad de generalizacion. Reduce el sobreajuste de la
    busqueda de hiperparametros.

    Devuelve metricas promedio + su std entre folds.
    """
    from sklearn.model_selection import TimeSeriesSplit

    n = len(df)
    max_splits = max(2, min(n_splits, n // 18))   # cada fold con material suficiente
    tscv = TimeSeriesSplit(n_splits=max_splits)

    fold_metrics = []
    for tr_idx, te_idx in tscv.split(df):
        # tr_idx y te_idx son contiguos y ordenados por TimeSeriesSplit
        df_fit = df.iloc[tr_idx]
        df_eval = df.iloc[te_idx]
        # necesitamos que df_eval venga justo despues de df_fit (lo garantiza TSCV)
        try:
            m = evaluate_config(df_fit, df_eval, config)
            fold_metrics.append(m)
        except Exception:
            continue

    if not fold_metrics:
        raise ValueError("Ningun fold produjo predicciones validas.")

    keys = fold_metrics[0].keys()
    out = {}
    for k in keys:
        vals = np.array([fm[k] for fm in fold_metrics])
        out[k] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    out["n_folds"] = len(fold_metrics)
    return out
