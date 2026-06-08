"""
Modelos clasicos univariados de series temporales.

Todos operan directamente sobre la serie del target y predicen el periodo
de test con walk-forward (re-ajustando o usando forecast directo segun el
modelo). Buenos baselines para series mensuales estacionales con tendencia.

Modelos incluidos:
    ExponentialSmoothing (Holt-Winters)  - tendencia + estacionalidad
    SARIMA                                - ARIMA estacional
    Theta                                 - metodo Theta (robusto, ganador M3)
    Naive estacional                      - baseline: y[t] = y[t-12]
"""

from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    _SM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SM_AVAILABLE = False

try:
    from statsmodels.tsa.forecasting.theta import ThetaModel
    _THETA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _THETA_AVAILABLE = False

SEASONAL_PERIOD = 12


def sm_available() -> bool:
    return _SM_AVAILABLE


# Nombres de los modelos clasicos disponibles
def available_classical() -> list[str]:
    models = ["SeasonalNaive"]
    if _SM_AVAILABLE:
        models += ["ExponentialSmoothing", "SARIMA"]
    if _THETA_AVAILABLE:
        models += ["Theta"]
    return models


# ---------------------------------------------------------------------------
# Forecast directo: ajusta con train y predice todo el test de una vez
# ---------------------------------------------------------------------------

def forecast_classical(
    name: str,
    train: np.ndarray,
    horizon: int,
    seasonal_periods: int = SEASONAL_PERIOD,
) -> np.ndarray:
    """
    Predice `horizon` pasos con el modelo clasico indicado, ajustando una vez
    con `train`. Devuelve un array de longitud horizon (nivel).
    """
    train = np.asarray(train, dtype=float)

    if name == "SeasonalNaive":
        # Repite el ultimo ciclo estacional observado
        last_season = train[-seasonal_periods:]
        reps = int(np.ceil(horizon / seasonal_periods))
        return np.tile(last_season, reps)[:horizon]

    if name == "ExponentialSmoothing":
        if not _SM_AVAILABLE:
            raise ImportError("statsmodels no disponible.")
        model = ExponentialSmoothing(
            train, trend="add", seasonal="add",
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True)
        return np.asarray(fit.forecast(horizon), dtype=float)

    if name == "SARIMA":
        if not _SM_AVAILABLE:
            raise ImportError("statsmodels no disponible.")
        # Orden razonable por defecto para series mensuales estacionales
        model = SARIMAX(
            train, order=(1, 1, 1),
            seasonal_order=(1, 1, 1, seasonal_periods),
            enforce_stationarity=False, enforce_invertibility=False,
        )
        fit = model.fit(disp=False)
        return np.asarray(fit.forecast(horizon), dtype=float)

    if name == "Theta":
        if not _THETA_AVAILABLE:
            raise ImportError("ThetaModel no disponible.")
        model = ThetaModel(train, period=seasonal_periods)
        fit = model.fit()
        return np.asarray(fit.forecast(horizon), dtype=float)

    raise ValueError(f"Modelo clasico desconocido: {name}")


# ---------------------------------------------------------------------------
# Variantes walk-forward (re-ajuste en cada paso) - mas lentas, mas justas
# ---------------------------------------------------------------------------

def forecast_classical_walkforward(
    name: str,
    train: np.ndarray,
    test: np.ndarray,
    seasonal_periods: int = SEASONAL_PERIOD,
) -> np.ndarray:
    """
    Walk-forward t+1: en cada paso ajusta con el historico y predice 1 mes,
    luego incorpora el valor real. Mas honesto para evaluar t+1.
    """
    history = list(np.asarray(train, dtype=float))
    preds = []
    for t in range(len(test)):
        try:
            yhat = float(forecast_classical(name, np.array(history), 1, seasonal_periods)[0])
        except Exception:
            yhat = history[-1]
        preds.append(yhat)
        history.append(float(test[t]))
    return np.array(preds)
