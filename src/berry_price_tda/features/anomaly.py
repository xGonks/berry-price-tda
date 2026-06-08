"""
Estrategias de manejo del periodo anomalo (shock COVID).

El shock de 2020 distorsiona la serie y puede contaminar el entrenamiento.
Este modulo implementa 6 estrategias para tratarlo, aplicables a la serie
de entrenamiento (target + exogenas) antes de construir las ventanas:

    1. none        - Baseline sin correccion (referencia)
    2. dummy       - Variable dummy de intervencion (marca el periodo)
    3. winsorize   - Recorta/imputa valores extremos del periodo
    4. exclude     - Excluye el periodo del entrenamiento
    5. prophet     - (se maneja aparte) Prophet con changepoints
    6. isolation   - Isolation Forest para detectar y tratar anomalias

Las estrategias 1-4 y 6 devuelven (target_tratado, exog_tratado, dummy_opcional).
La estrategia 5 (Prophet) es un modelo completo y se evalua por separado.

Fechas COVID configurables (default segun el reto):
    COVID_START = 2020-03-01
    COVID_END   = 2020-10-01
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

COVID_START = "2020-03-01"
COVID_END = "2020-10-01"

ANOMALY_STRATEGIES = ["none", "dummy", "winsorize", "exclude", "isolation"]
# "prophet" se evalua aparte porque es un modelo, no un pretratamiento.


def covid_mask(index: pd.DatetimeIndex,
               start: str = COVID_START, end: str = COVID_END) -> np.ndarray:
    """Devuelve mascara booleana True en el periodo COVID."""
    idx = pd.DatetimeIndex(index)
    return (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))


# ---------------------------------------------------------------------------
# Estrategias de pretratamiento (operan sobre el DataFrame de entrenamiento)
# ---------------------------------------------------------------------------

def apply_strategy(
    df_train: pd.DataFrame,
    strategy: str,
    target_col: str,
    start: str = COVID_START,
    end: str = COVID_END,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """
    Aplica una estrategia de manejo de anomalia al DataFrame de ENTRENAMIENTO.

    Returns
    -------
    df_out : pd.DataFrame  (target + exogenas, posiblemente modificado)
    dummy  : np.ndarray | None
        Vector dummy de intervencion alineado con df_out, o None si la
        estrategia no usa dummy. Si no es None, debe anadirse como feature.
    """
    df = df_train.copy()
    mask = covid_mask(df.index, start, end)

    if strategy == "none":
        return df, None

    if strategy == "dummy":
        # Marca el periodo; el modelo aprende un offset para esas fechas.
        dummy = mask.astype(float)
        return df, dummy

    if strategy == "winsorize":
        # Recorta los valores del periodo COVID al percentil 5/95 de la
        # serie SIN COVID, luego interpola para suavizar.
        clean = df.loc[~mask, target_col]
        lo, hi = clean.quantile(0.05), clean.quantile(0.95)
        df.loc[mask, target_col] = df.loc[mask, target_col].clip(lo, hi)
        # Tambien tratar exogenas igual
        for c in df.columns:
            if c == target_col:
                continue
            clean_c = df.loc[~mask, c]
            lo_c, hi_c = clean_c.quantile(0.05), clean_c.quantile(0.95)
            df.loc[mask, c] = df.loc[mask, c].clip(lo_c, hi_c)
        return df, None

    if strategy == "exclude":
        # Elimina las filas del periodo COVID del entrenamiento.
        # OJO: rompe la continuidad temporal de las ventanas; el builder
        # debe construir ventanas solo dentro de tramos contiguos.
        df = df.loc[~mask].copy()
        return df, None

    if strategy == "isolation":
        # Deteccion no supervisada: marca como anomalia lo que IsolationForest
        # detecte (no solo COVID) e imputa esos puntos por interpolacion.
        from sklearn.ensemble import IsolationForest
        feats = df.values
        iso = IsolationForest(contamination=0.08, random_state=42)
        labels = iso.fit_predict(feats)        # -1 = anomalia
        anom = labels == -1
        df_imp = df.copy()
        df_imp[anom] = np.nan
        df_imp = df_imp.interpolate(method="linear").bfill().ffill()
        return df_imp, None

    raise ValueError(f"Estrategia desconocida: {strategy}")


# ---------------------------------------------------------------------------
# Prophet (estrategia 5) - modelo completo con changepoints
# ---------------------------------------------------------------------------

def prophet_available() -> bool:
    try:
        import prophet  # noqa: F401
        return True
    except ImportError:
        return False


def prophet_forecast(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str,
    use_covid_regressor: bool = True,
    start: str = COVID_START,
    end: str = COVID_END,
) -> np.ndarray:
    """
    Ajusta Prophet con deteccion automatica de changepoints y predice el test.

    Si use_covid_regressor=True, anade un regresor binario que marca el
    periodo COVID (equivalente a la dummy de intervencion en Prophet).

    Returns
    -------
    np.ndarray con las predicciones alineadas con df_test (nivel).
    """
    from prophet import Prophet

    train = pd.DataFrame({
        "ds": df_train.index,
        "y": df_train[target_col].values,
    })

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.5,   # mas flexible para captar quiebres
        seasonality_mode="additive",
    )

    if use_covid_regressor:
        train["covid"] = covid_mask(df_train.index, start, end).astype(float)
        m.add_regressor("covid")

    m.fit(train)

    future = pd.DataFrame({"ds": df_test.index})
    if use_covid_regressor:
        future["covid"] = covid_mask(df_test.index, start, end).astype(float)

    forecast = m.predict(future)
    return forecast["yhat"].values
