"""
Barrido factorial COMPLETO para inferencia estadistica.

A diferencia de run_optuna (que busca el optimo rapido), este modulo evalua
MUCHAS combinaciones (idealmente todas) para responder, con pruebas de
hipotesis, preguntas como:
    - El uso de TDA mejora el desempeno de forma estadisticamente significativa?
    - El uso de exogenas mejora el desempeno?
    - Que factores (modelo, ventana, transformacion, ...) influyen mas?

Diseno:
    - Cada combinacion se evalua con TimeSeriesSplit sobre train+val.
    - Se guardan las metricas POR FOLD (no solo el promedio) -> material para
      pruebas pareadas (mismo setup con/sin un factor en el mismo fold).
    - El TEST (2022+) permanece INTACTO: la inferencia se hace sobre los folds
      de CV, no sobre el test.
    - Paralelizado por procesos (loky) + checkpoint incremental para reanudar.

El espacio factorial completo:
    modelos x ventanas{24,36,48} x transformaciones x 16 subconjuntos exog
    x TDA{si,no} x estrategias  =  ~19,200 combinaciones.
"""

from __future__ import annotations

import os
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.loader import (
    EXOGENOUS_COLS, all_exog_subsets, date_train_val_test_split, load_dataset,
)
from ..features.anomaly import ANOMALY_STRATEGIES
from ..features.tda import gtda_available
from ..models.sklearn_models import SKLEARN_MODELS
from .evaluate import RunConfig, evaluate_config

warnings.filterwarnings("ignore")

LARGE_WINDOWS = [12, 24, 36, 48]
ALL_TRANSFORMS_EXP = ["none", "standard", "diff", "log_diff", "seasonal_diff"]


def generate_grid(
    models=None, windows=None, transforms=None,
    exog_subsets=None, tda_options=None, strategies=None,
):
    """Genera todas las RunConfig del factorial completo (o el subconjunto dado)."""
    models = models or list(SKLEARN_MODELS.keys())
    windows = windows or LARGE_WINDOWS
    transforms = transforms or ALL_TRANSFORMS_EXP
    exog_subsets = exog_subsets if exog_subsets is not None else all_exog_subsets()
    if tda_options is None:
        tda_options = [False, True] if gtda_available() else [False]
    strategies = strategies or ANOMALY_STRATEGIES

    configs = []
    for model, w, tf, exog, tda, strat in product(
        models, windows, transforms, exog_subsets, tda_options, strategies
    ):
        configs.append(RunConfig(
            model_name=model, window=w, transform=tf,
            exog_cols=list(exog), use_tda=tda, strategy=strat,
        ))
    return configs


def _eval_one(args):
    """Evalua una config con CV y devuelve filas POR FOLD. Para paralelizar."""
    config, df_trainval, cv_splits, metric = args
    from sklearn.model_selection import TimeSeriesSplit
    rows = []
    try:
        n = len(df_trainval)
        max_splits = max(2, min(cv_splits, n // 18))
        tscv = TimeSeriesSplit(n_splits=max_splits)
        for fold_i, (tr_idx, te_idx) in enumerate(tscv.split(df_trainval)):
            df_fit = df_trainval.iloc[tr_idx]
            df_eval = df_trainval.iloc[te_idx]
            try:
                m = evaluate_config(df_fit, df_eval, config)
            except Exception:
                continue
            row = {**config.to_row(), "fold": fold_i, **m}
            rows.append(row)
    except Exception:
        pass
    return rows


def run_full_factorial(
    data_path=None,
    models=None, windows=None, transforms=None,
    exog_subsets=None, tda_options=None, strategies=None,
    cv_splits=4, metric="mae",
    n_jobs=-1,
    checkpoint=None,         # ruta CSV para guardado incremental / reanudacion
    verbose=True,
):
    """
    Ejecuta el barrido factorial completo y devuelve un DataFrame con una fila
    POR (combinacion, fold). Ese formato largo es el insumo del analisis
    estadistico (pruebas pareadas, ANOVA, etc.).

    checkpoint : si se da una ruta, guarda el progreso cada cierto numero de
                 configuraciones y, si el archivo ya existe, reanuda desde ahi.
    """
    n_cpu = os.cpu_count() or 1
    workers = max(1, n_cpu - 1) if n_jobs == -1 else max(1, min(n_jobs, n_cpu))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    df = load_dataset(data_path) if data_path else load_dataset()
    train, val, test = date_train_val_test_split(df)
    df_trainval = pd.concat([train, val])

    configs = generate_grid(models, windows, transforms,
                            exog_subsets, tda_options, strategies)

    if verbose:
        print("=" * 70)
        print("  Barrido factorial COMPLETO (inferencia estadistica)")
        print("=" * 70)
        print(f"  Combinaciones: {len(configs)}  x  ~{cv_splits} folds")
        print(f"  Train+Val: {len(df_trainval)}  |  Test intacto: {len(test)}")
        print(f"  TDA disponible: {gtda_available()}  |  Nucleos: {workers}/{n_cpu}\n")

    # Reanudacion desde checkpoint
    done_labels = set()
    existing_rows = []
    if checkpoint:
        Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
    if checkpoint and Path(checkpoint).exists():
        prev = pd.read_csv(checkpoint)
        existing_rows = prev.to_dict("records")
        done_labels = set(prev["__label__"].unique()) if "__label__" in prev else set()
        if verbose:
            print(f"  Reanudando: {len(done_labels)} combinaciones ya hechas.\n")

    pending = [c for c in configs if c.label() not in done_labels]

    tasks = [(c, df_trainval, cv_splits, metric) for c in pending]

    all_rows = list(existing_rows)
    save_every = max(50, len(tasks) // 50)

    def _run_serial():
        for i, t in enumerate(tasks):
            res = _eval_one(t)
            for r in res:
                r["__label__"] = t[0].label()
            all_rows.extend(res)
            if verbose and (i + 1) % save_every == 0:
                print(f"  {i+1}/{len(tasks)} combinaciones evaluadas")
            if checkpoint and (i + 1) % save_every == 0:
                pd.DataFrame(all_rows).to_csv(checkpoint, index=False)

    if workers == 1:
        _run_serial()
    else:
        from joblib import Parallel, delayed
        # Procesamos en lotes para poder hacer checkpoint
        batch = save_every
        for start in range(0, len(tasks), batch):
            chunk = tasks[start:start + batch]
            results = Parallel(n_jobs=workers, backend="loky")(
                delayed(_eval_one)(t) for t in chunk
            )
            for t, res in zip(chunk, results):
                for r in res:
                    r["__label__"] = t[0].label()
                all_rows.extend(res)
            if verbose:
                print(f"  {min(start+batch, len(tasks))}/{len(tasks)} combinaciones evaluadas")
            if checkpoint:
                pd.DataFrame(all_rows).to_csv(checkpoint, index=False)

    df_long = pd.DataFrame(all_rows)
    if checkpoint:
        df_long.to_csv(checkpoint, index=False)

    if verbose:
        n_combos = df_long["__label__"].nunique() if "__label__" in df_long else len(df_long)
        print(f"\n  Listo: {len(df_long)} filas (combinacion x fold), "
              f"{n_combos} combinaciones.")

    return df_long


# Compatibilidad: alias para el nombre antiguo
def run_full_experiment(*args, **kwargs):
    """Alias historico -> run_full_factorial."""
    return run_full_factorial(*args, **kwargs)
