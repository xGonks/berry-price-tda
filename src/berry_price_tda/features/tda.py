"""
Extracción de características topológicas (TDA) a partir de ventanas
de series temporales.

Pipeline TDA por ventana:
    serie 1D - embedding de Takens - nube de puntos en R^d
             - homología persistente (Vietoris-Rips)
             - diagrama de persistencia (H0, H1)
             - vector de características topológicas

Las características extraídas por ventana son:
    - max_pers_h1   : persistencia máxima de ciclos H1 (estacionalidad fuerte)
    - mean_pers_h1  : persistencia media de H1
    - std_pers_h1   : dispersión de la persistencia H1
    - n_cycles_h1   : número de ciclos H1
    - entropy_h1    : entropía de persistencia de H1
    - amplitude_h1  : amplitud (norma) del diagrama H1
    - max_pers_h0   : persistencia máxima de componentes H0
    - n_comp_h0     : número de componentes H0

Si giotto-tda no está disponible o la ventana es demasiado corta, se
devuelve un vector de ceros del tamaño correcto (degradación segura).
"""

from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")

# Nombres de las features en orden fijo (importante para reproducibilidad)
TDA_FEATURE_NAMES = [
    "max_pers_h1",
    "mean_pers_h1",
    "std_pers_h1",
    "n_cycles_h1",
    "entropy_h1",
    "amplitude_h1",
    "max_pers_h0",
    "n_comp_h0",
]
N_TDA_FEATURES = len(TDA_FEATURE_NAMES)

# Importación perezosa de giotto-tda
try:
    from gtda.time_series import SingleTakensEmbedding
    from gtda.homology import VietorisRipsPersistence
    _GTDA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _GTDA_AVAILABLE = False


def _persistence_entropy(persistences: np.ndarray) -> float:
    """Entropía de Shannon normalizada de un conjunto de persistencias."""
    persistences = persistences[persistences > 0]
    if len(persistences) == 0:
        return 0.0
    p = persistences / persistences.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))


def extract_tda_features(
    window: np.ndarray,
    embedding_dim: int = 6,
    time_delay: int = 1,
    min_points: int = 5,
) -> np.ndarray:
    """
    Extrae el vector de características topológicas de una ventana 1D.

    Parameters
    ----------
    window : np.ndarray
        Segmento 1D de la serie temporal.
    embedding_dim : int
        Dimensión del embedding de Takens.
    time_delay : int
        Retardo del embedding.
    min_points : int
        Mínimo de puntos en la nube embebida para calcular homología.

    Returns
    -------
    np.ndarray
        Vector de longitud N_TDA_FEATURES. Ceros si falla / datos insuficientes.
    """
    zeros = np.zeros(N_TDA_FEATURES, dtype=float)

    if not _GTDA_AVAILABLE:
        return zeros

    window = np.asarray(window, dtype=float).flatten()
    min_required = (embedding_dim - 1) * time_delay + 1
    if len(window) < min_required + min_points:
        return zeros

    try:
        ste = SingleTakensEmbedding(
            parameters_type="fixed",
            dimension=embedding_dim,
            time_delay=time_delay,
        )
        cloud = ste.fit_transform(window)
        if len(cloud) < min_points:
            return zeros

        vr = VietorisRipsPersistence(
            homology_dimensions=[0, 1],
            n_jobs=1,
            collapse_edges=True,
        )
        diagram = vr.fit_transform([cloud])[0]  # (n_pts, 3): birth, death, dim

        # --- H1 (ciclos) ---
        mask_h1 = diagram[:, 2] == 1
        feats = np.zeros(N_TDA_FEATURES, dtype=float)
        if np.any(mask_h1):
            h1 = diagram[mask_h1]
            pers_h1 = h1[:, 1] - h1[:, 0]
            pers_h1 = pers_h1[np.isfinite(pers_h1)]
            if len(pers_h1) > 0:
                feats[0] = np.max(pers_h1)            # max_pers_h1
                feats[1] = np.mean(pers_h1)           # mean_pers_h1
                feats[2] = np.std(pers_h1)            # std_pers_h1
                feats[3] = len(pers_h1)               # n_cycles_h1
                feats[4] = _persistence_entropy(pers_h1)  # entropy_h1
                feats[5] = np.sqrt(np.sum(pers_h1 ** 2))  # amplitude_h1 (L2)

        # --- H0 (componentes) ---
        mask_h0 = diagram[:, 2] == 0
        if np.any(mask_h0):
            h0 = diagram[mask_h0]
            pers_h0 = h0[:, 1] - h0[:, 0]
            pers_h0 = pers_h0[np.isfinite(pers_h0)]
            if len(pers_h0) > 0:
                feats[6] = np.max(pers_h0)   # max_pers_h0
                feats[7] = len(pers_h0)      # n_comp_h0

        return feats

    except Exception:
        return zeros


def tda_features_matrix(
    windows: np.ndarray,
    embedding_dim: int = 6,
    time_delay: int = 1,
) -> np.ndarray:
    """
    Aplica extract_tda_features a un conjunto de ventanas.

    Parameters
    ----------
    windows : np.ndarray
        Matriz (n_windows, window_size) de ventanas.

    Returns
    -------
    np.ndarray
        Matriz (n_windows, N_TDA_FEATURES).
    """
    out = np.zeros((len(windows), N_TDA_FEATURES), dtype=float)
    for i, w in enumerate(windows):
        out[i] = extract_tda_features(w, embedding_dim, time_delay)
    return out


def gtda_available() -> bool:
    """Indica si giotto-tda está instalado."""
    return _GTDA_AVAILABLE


def sanitize_tda_matrix(
    tda_train: np.ndarray,
    tda_test: np.ndarray,
    var_threshold: float = 1e-8,
    clip_sigma: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Sanea las matrices de features TDA para que no descalabren a los modelos
    lineales (era la causa de los R^2 de 10^19).

    Pasos:
      1. Elimina columnas con varianza casi nula (constantes) usando los
         estadisticos del TRAIN.
      2. Hace clip de outliers a +-clip_sigma desviaciones (estadisticos del
         train) para acotar valores extremos de persistencia.

    Returns
    -------
    tda_train_clean, tda_test_clean, kept_cols
        Matrices saneadas y los indices de columnas conservadas.
    """
    tda_train = np.asarray(tda_train, dtype=float)
    tda_test = np.asarray(tda_test, dtype=float)

    mu = tda_train.mean(axis=0)
    sigma = tda_train.std(axis=0)

    # 1) columnas con varianza util
    kept = [j for j in range(tda_train.shape[1]) if sigma[j] > var_threshold]
    if not kept:
        # nada util: devolver matrices vacias (el caller decide no usar TDA)
        return (np.zeros((len(tda_train), 0)),
                np.zeros((len(tda_test), 0)), [])

    tr = tda_train[:, kept].copy()
    te = tda_test[:, kept].copy()
    mu_k = mu[kept]
    sg_k = sigma[kept] + 1e-12

    # 2) clip de outliers segun estadisticos del train
    lo = mu_k - clip_sigma * sg_k
    hi = mu_k + clip_sigma * sg_k
    tr = np.clip(tr, lo, hi)
    te = np.clip(te, lo, hi)

    return tr, te, kept
