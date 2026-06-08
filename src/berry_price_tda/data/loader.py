"""Carga del dataset y utilidades de partición temporal."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "WPUSI01102B"
EXOGENOUS_COLS = ["cpi_fresh_fruits", "ppi_fertilizers", "diesel_price", "mxn_usd"]
DATE_COL = "observation_date"

# Ruta por defecto relativa a la raíz del proyecto
DEFAULT_DATA = Path("data/interim/berry_features.csv")


def load_dataset(path: Path | str = DEFAULT_DATA) -> pd.DataFrame:
    """
    Carga el CSV consolidado (target + exógenas) indexado por fecha.

    Devuelve un DataFrame con índice DatetimeIndex mensual y columnas
    [WPUSI01102B, cpi_fresh_fruits, ppi_fertilizers, diesel_price, mxn_usd].
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. Ejecuta primero:  make exog"
        )

    df = pd.read_csv(path, parse_dates=[DATE_COL], index_col=DATE_COL)
    df = df.sort_index()

    if TARGET not in df.columns:
        raise ValueError(f"La columna objetivo '{TARGET}' no está en {path}")

    # Asegurar que solo trabajamos con columnas conocidas y sin NaN
    cols = [TARGET] + [c for c in EXOGENOUS_COLS if c in df.columns]
    df = df[cols].astype(float)

    if df.isna().any().any():
        # Imputación conservadora por si quedaron huecos
        df = df.ffill().bfill()

    return df


def get_target(df: pd.DataFrame) -> np.ndarray:
    """Extrae el vector objetivo como np.ndarray 1D."""
    return df[TARGET].to_numpy(dtype=float)


def get_exogenous(df: pd.DataFrame) -> np.ndarray:
    """Extrae la matriz de exógenas (n_obs * n_exog) como np.ndarray 2D."""
    cols = [c for c in EXOGENOUS_COLS if c in df.columns]
    return df[cols].to_numpy(dtype=float)


def temporal_train_test_split(
    df: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Partición temporal simple (sin shuffle): primeros (1-test_size) para
    entrenamiento, últimos test_size para prueba final.
    """
    n = len(df)
    cut = int(n * (1 - test_size))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


# ---------------------------------------------------------------------------
# Split fijo por fecha (reto WPUSI01102B)
# ---------------------------------------------------------------------------
TRAIN_END = "2021-12-01"
TEST_START = "2022-01-01"


def date_train_test_split(
    df: pd.DataFrame,
    train_end: str = TRAIN_END,
    test_start: str = TEST_START,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Particion temporal fija por fecha:
        train: hasta train_end (inclusive)
        test : desde test_start (inclusive)
    """
    train = df.loc[:train_end].copy()
    test = df.loc[test_start:].copy()
    return train, test


# ---------------------------------------------------------------------------
# Split 3-vias para optimizacion con Optuna
# ---------------------------------------------------------------------------
# Optuna optimiza sobre VALIDACION; el TEST queda intacto para evaluacion
# final honesta (sin sesgo de optimizacion).
VAL_START = "2020-01-01"
VAL_END = "2021-12-01"


def date_train_val_test_split(
    df: pd.DataFrame,
    val_start: str = VAL_START,
    val_end: str = VAL_END,
    test_start: str = TEST_START,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Particion temporal en 3:
        train: hasta val_start (exclusivo)
        val  : [val_start, val_end]   (para Optuna)
        test : desde test_start       (evaluacion final, intacto)
    """
    train = df.loc[:val_start].iloc[:-1].copy()   # hasta justo antes de val
    val = df.loc[val_start:val_end].copy()
    test = df.loc[test_start:].copy()
    return train, val, test


def select_exogenous(df: pd.DataFrame, cols: list[str] | None) -> np.ndarray | None:
    """
    Extrae un subconjunto especifico de exogenas.
    cols=None o lista vacia -> None (sin exogenas).
    """
    if not cols:
        return None
    present = [c for c in cols if c in df.columns]
    if not present:
        return None
    return df[present].to_numpy(dtype=float)


def all_exog_subsets(exog_cols: list[str] | None = None) -> list[list[str]]:
    """
    Genera los 2^n subconjuntos de exogenas (incluye el vacio = sin exogenas).
    Para 4 variables -> 16 subconjuntos.
    """
    from itertools import combinations
    cols = exog_cols if exog_cols is not None else EXOGENOUS_COLS
    subsets = []
    for r in range(len(cols) + 1):
        for combo in combinations(cols, r):
            subsets.append(list(combo))
    return subsets
