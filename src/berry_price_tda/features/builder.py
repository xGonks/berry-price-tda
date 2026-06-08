"""
Construccion de matrices de diseno (X, y) para forecasting t+1.

A partir de la serie objetivo y las exogenas, genera ventanas deslizantes
y ensambla la matriz de features segun una configuracion factorial:

    use_lags : usa la ventana de valores pasados del target  (siempre True)
    use_exog : anade los valores actuales (y opcionalmente lags) de exogenas
    use_tda  : anade las caracteristicas topologicas de la ventana

Modo de objetivo (controlado por `difference`):
    difference="none"      -> y[i] = nivel del target en t+1  (predice el nivel)
    difference="first"     -> y[i] = target[t+1] - target[t]  (primera diferencia)
    difference="seasonal"  -> y[i] = target[t+1] - target[t+1-12] (dif. estacional)

Predecir la DIFERENCIA suele mejorar mucho los resultados cuando la serie
tiene tendencia (no estacionaria), porque vuelve el objetivo casi estacionario.
El nivel se reconstruye despues sumando la diferencia al valor base.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tda import N_TDA_FEATURES, TDA_FEATURE_NAMES, tda_features_matrix

SEASONAL_PERIOD = 12   # meses


@dataclass
class FeatureConfig:
    """Configuracion de que features incluir y como plantear el objetivo."""
    window_size: int = 12
    use_exog: bool = False
    use_tda: bool = False
    embedding_dim: int = 6
    time_delay: int = 1
    difference: str = "first"   # "none" | "first" | "seasonal"
    exog_lags: int = 0          # cuantos lags de exogenas anadir (ademas del actual)

    def label(self) -> str:
        """Etiqueta legible de la combinacion (para tablas comparativas)."""
        parts = [f"win{self.window_size}"]
        parts.append("exog" if self.use_exog else "noexog")
        parts.append("tda" if self.use_tda else "notda")
        if self.difference != "none":
            parts.append(f"diff-{self.difference}")
        return "+".join(parts)


def make_windows(
    target: np.ndarray,
    exog: np.ndarray | None,
    config: FeatureConfig,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """
    Genera (X, y, feature_names, base) para forecasting t+1.

    Returns
    -------
    X : np.ndarray  (n_samples, n_features)
    y : np.ndarray  (n_samples,)  -- objetivo (nivel o diferencia segun config)
    feature_names : list[str]
    base : np.ndarray (n_samples,) -- valor base para reconstruir el nivel:
        * difference="none"     -> ceros (y ya es el nivel)
        * difference="first"    -> target[t]       (nivel = base + y)
        * difference="seasonal" -> target[t+1-12]  (nivel = base + y)
    """
    target = np.asarray(target, dtype=float)
    n = len(target)
    w = config.window_size

    # El inicio efectivo depende de si usamos diferencia estacional
    start = max(w, SEASONAL_PERIOD) if config.difference == "seasonal" else w

    if n <= start:
        raise ValueError(
            f"Serie demasiado corta ({n}) para ventana {w} y modo {config.difference}"
        )

    n_samples = n - start
    windows = np.zeros((n_samples, w), dtype=float)
    y = np.zeros(n_samples, dtype=float)
    base = np.zeros(n_samples, dtype=float)

    for idx in range(n_samples):
        t = start + idx          # indice del punto "actual" (ultimo de la ventana)
        windows[idx] = target[t - w : t]
        level_next = target[t]   # valor a predecir (t+1 desde la perspectiva de la ventana)

        if config.difference == "none":
            y[idx] = level_next
            base[idx] = 0.0
        elif config.difference == "first":
            y[idx] = level_next - target[t - 1]
            base[idx] = target[t - 1]
        elif config.difference == "seasonal":
            y[idx] = level_next - target[t - SEASONAL_PERIOD]
            base[idx] = target[t - SEASONAL_PERIOD]
        else:
            raise ValueError(f"difference invalido: {config.difference}")

    blocks: list[np.ndarray] = []
    names: list[str] = []

    # 1) Lags del target (la ventana)
    blocks.append(windows)
    names.extend([f"lag_{w - j}" for j in range(w)])

    # 2) Exogenas: valor actual + lags opcionales
    if config.use_exog:
        if exog is None:
            raise ValueError("use_exog=True pero no se pasaron exogenas.")
        exog = np.asarray(exog, dtype=float)
        n_exog = exog.shape[1]
        for lag in range(config.exog_lags + 1):     # lag=0 es el valor actual
            block = np.zeros((n_samples, n_exog), dtype=float)
            for idx in range(n_samples):
                t = start + idx
                block[idx] = exog[t - lag]
            blocks.append(block)
            suffix = "" if lag == 0 else f"_lag{lag}"
            names.extend([f"exog_{k}{suffix}" for k in range(n_exog)])

    # 3) Caracteristicas topologicas de cada ventana
    if config.use_tda:
        tda_block = tda_features_matrix(
            windows, config.embedding_dim, config.time_delay
        )
        blocks.append(tda_block)
        names.extend([f"tda_{nm}" for nm in TDA_FEATURE_NAMES])

    X = np.hstack(blocks)
    return X, y, names, base


def reconstruct_level(y_pred_diff: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Reconstruye el nivel a partir de la diferencia predicha y la base."""
    return np.asarray(y_pred_diff) + np.asarray(base)


def all_configs(
    window_size: int = 12,
    embedding_dim: int = 6,
    time_delay: int = 1,
    difference: str = "first",
    exog_lags: int = 0,
) -> list[FeatureConfig]:
    """
    Genera las 4 combinaciones factoriales de exog * tda con un modo de
    diferencia fijo.
    """
    combos = []
    for use_exog in (False, True):
        for use_tda in (False, True):
            combos.append(
                FeatureConfig(
                    window_size=window_size,
                    use_exog=use_exog,
                    use_tda=use_tda,
                    embedding_dim=embedding_dim,
                    time_delay=time_delay,
                    difference=difference,
                    exog_lags=exog_lags,
                )
            )
    return combos
