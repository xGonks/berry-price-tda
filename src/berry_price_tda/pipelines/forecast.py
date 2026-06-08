"""
Forecast multi-horizonte a futuro (proyeccion final).

Una vez identificada la mejor combinacion en el experimento factorial,
este modulo entrena con toda la serie y proyecta `horizon` meses hacia
adelante de forma recursiva.

Soporta los tres modos de objetivo (none / first / seasonal): el modelo
predice la diferencia y aqui se reconstruye el nivel paso a paso.

Las exogenas se extrapolan con carry-forward (ultimo valor conocido).
El usuario puede sustituir esto por sus propios escenarios.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.builder import SEASONAL_PERIOD, FeatureConfig, make_windows
from ..features.tda import extract_tda_features
from ..models.sklearn_models import get_sklearn_model


def recursive_forecast_sklearn(df, model_name, config: FeatureConfig, horizon=12):
    """
    Proyeccion recursiva a `horizon` meses usando un modelo de sklearn.

    Returns
    -------
    DataFrame con indice de fechas futuras y columna 'forecast' (en nivel).
    """
    from ..data.loader import EXOGENOUS_COLS, TARGET

    target = df[TARGET].to_numpy(dtype=float)
    exog_cols = [c for c in EXOGENOUS_COLS if c in df.columns]
    exog = df[exog_cols].to_numpy(dtype=float) if exog_cols else None

    # Entrenar con toda la serie
    X, y, _, _ = make_windows(target, exog, config)
    model = get_sklearn_model(model_name)
    model.fit(X, y)

    w = config.window_size
    n_exog = exog.shape[1] if exog is not None else 0

    # Historial que vamos extendiendo con las predicciones
    history = list(target)
    last_exog = exog[-1].copy() if (config.use_exog and exog is not None) else None

    preds = []
    for _ in range(horizon):
        window = np.array(history[-w:], dtype=float)

        blocks = [window.reshape(1, -1)]
        if config.use_exog and last_exog is not None:
            for lag in range(config.exog_lags + 1):
                blocks.append(last_exog.reshape(1, -1))   # carry-forward para todos los lags
        if config.use_tda:
            tda_feat = extract_tda_features(
                window, config.embedding_dim, config.time_delay
            )
            blocks.append(tda_feat.reshape(1, -1))
        x = np.hstack(blocks)

        y_hat = float(model.predict(x)[0])

        # Reconstruir nivel segun el modo de diferencia
        if config.difference == "none":
            level = y_hat
        elif config.difference == "first":
            level = history[-1] + y_hat
        elif config.difference == "seasonal":
            level = history[-SEASONAL_PERIOD] + y_hat
        else:
            level = y_hat

        preds.append(level)
        history.append(level)

    last_date = df.index[-1]
    future_idx = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=horizon, freq="MS",
    )
    return pd.DataFrame({"forecast": preds}, index=future_idx)
