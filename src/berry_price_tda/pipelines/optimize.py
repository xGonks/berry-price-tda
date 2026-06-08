"""
Optimizacion de hiperparametros con Optuna + factorial dirigido final.

Mejoras respecto a la version inicial (para combatir el sobreajuste de la
busqueda que se detecto con 24 puntos de validacion):

  1. ESPACIO ACOTADO: las exogenas que demostraron danar (R^2 negativos
     enormes) se excluyen por defecto; estrategias anti-anomalia reducidas
     a las utiles. Todo configurable.

  2. VALIDACION ROBUSTA: en vez de optimizar sobre un solo bloque de 24
     meses (COVID), se usa TimeSeriesSplit sobre train+val -> Optuna optimiza
     el promedio de varios folds temporales. Mucho menos sobreajuste.

  3. PENALIZACION DE COMPLEJIDAD: el objetivo es  metrica + lambda * n_features
     (normalizado), empujando hacia modelos simples que generalizan mejor.

El TEST (2022+) sigue intacto durante la busqueda; se evalua una sola vez al
final con la mejor config.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..data.loader import (
    EXOGENOUS_COLS, TEST_START, date_train_val_test_split, load_dataset,
)
from ..features.tda import gtda_available
from ..models.sklearn_models import SKLEARN_MODELS
from .evaluate import RunConfig, count_features, evaluate_config, evaluate_config_cv

warnings.filterwarnings("ignore")

# Ventanas GRANDES unicamente
LARGE_WINDOWS = [24, 36, 48]
SEARCH_TRANSFORMS = ["none", "standard", "diff", "log_diff", "seasonal_diff"]
# Estrategias reducidas a las que tienen sentido (se quito isolation por agresiva;
# se puede reactivar pasando search_strategies)
DEFAULT_STRATEGIES = ["none", "dummy", "winsorize", "exclude"]


def _suggest_config(trial, allow_tda, allow_exog, strategies):
    """Define el espacio de busqueda (acotado) y devuelve una RunConfig."""
    model_name = trial.suggest_categorical("model", list(SKLEARN_MODELS.keys()))
    window = trial.suggest_categorical("window", LARGE_WINDOWS)
    transform = trial.suggest_categorical("transform", SEARCH_TRANSFORMS)
    strategy = trial.suggest_categorical("strategy", strategies)

    if allow_exog:
        exog_cols = [c for c in EXOGENOUS_COLS
                     if trial.suggest_categorical(f"exog__{c}", [False, True])]
    else:
        exog_cols = []

    use_tda = trial.suggest_categorical("use_tda", [False, True]) if allow_tda else False

    return RunConfig(
        model_name=model_name, window=window, transform=transform,
        exog_cols=exog_cols, use_tda=use_tda, strategy=strategy,
    )


def run_optuna_search(
    data_path=None,
    n_trials: int = 300,
    metric: str = "mae",
    allow_tda: bool = True,
    allow_exog: bool = False,          # por defecto SIN exogenas (danaban)
    search_strategies=None,            # default: DEFAULT_STRATEGIES
    use_cv: bool = True,               # TimeSeriesSplit como motor de validacion
    cv_splits: int = 4,
    complexity_lambda: float = 0.05,   # peso de la penalizacion por n_features
    seed: int = 42,
    n_jobs: int = -1,
    verbose: bool = True,
):
    """
    Busca la mejor configuracion con Optuna.

    use_cv=True  -> optimiza el promedio de TimeSeriesSplit sobre train+val
                    (robusto, recomendado).
    use_cv=False -> optimiza sobre el bloque de validacion 2020-2021 (rapido
                    pero propenso a sobreajuste).

    El objetivo minimizado es:  metrica_norm * (1 + complexity_lambda * n_feat_norm)
    """
    import os

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if search_strategies is None:
        search_strategies = DEFAULT_STRATEGIES

    n_cpu = os.cpu_count() or 1
    workers = max(1, n_cpu - 1) if n_jobs == -1 else max(1, min(n_jobs, n_cpu))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    df = load_dataset(data_path) if data_path else load_dataset()
    train, val, test = date_train_val_test_split(df)
    df_trainval = pd.concat([train, val])     # material de busqueda (sin test)

    allow_tda = allow_tda and gtda_available()
    higher_better = metric == "r2"

    if verbose:
        print("=" * 70)
        print("  Optuna - busqueda robusta (test intacto)")
        print("=" * 70)
        print(f"  Train+Val: {len(df_trainval)}  Test: {len(test)}")
        print(f"  Trials: {n_trials}  |  metrica: {metric}  |  TDA: {allow_tda}  |  exog: {allow_exog}")
        print(f"  Validacion: {'TimeSeriesSplit (%d folds)' % cv_splits if use_cv else 'bloque 2020-2021'}")
        print(f"  Estrategias: {search_strategies}")
        print(f"  Penalizacion complejidad lambda={complexity_lambda}  |  Nucleos: {workers}/{n_cpu}\n")

    # max features posible (para normalizar la penalizacion)
    max_feat = max(LARGE_WINDOWS) + len(EXOGENOUS_COLS) + 8

    def objective(trial):
        config = _suggest_config(trial, allow_tda, allow_exog, search_strategies)
        try:
            if use_cv:
                m = evaluate_config_cv(df_trainval, config, n_splits=cv_splits, metric=metric)
            else:
                m = evaluate_config(train, val, config)
            base = m[metric]
            # normalizar a "menor es mejor"
            score = -base if higher_better else base
            # penalizacion por complejidad
            nfeat = count_features(df_trainval, config)
            penalty = 1.0 + complexity_lambda * (nfeat / max_feat)
            return score * penalty
        except Exception:
            return 1e9

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    if workers == 1:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)
    else:
        import joblib
        with joblib.parallel_backend("loky", n_jobs=workers):
            study.optimize(objective, n_trials=n_trials, n_jobs=workers,
                           show_progress_bar=verbose)

    bp = study.best_params
    best_config = RunConfig(
        model_name=bp["model"], window=bp["window"], transform=bp["transform"],
        exog_cols=[c for c in EXOGENOUS_COLS if bp.get(f"exog__{c}", False)],
        use_tda=bp.get("use_tda", False), strategy=bp["strategy"],
    )

    df_trials = study.trials_dataframe()

    if verbose:
        print(f"\n  Mejor objetivo (penalizado): {study.best_value:.4f}")
        print(f"  Mejor config: {best_config.label()}")

    return study, best_config, df_trials


def evaluate_on_test(data_path, config: RunConfig):
    """Reentrena con train+val y evalua UNA sola vez sobre el test (2022+)."""
    df = load_dataset(data_path) if data_path else load_dataset()
    train, val, test = date_train_val_test_split(df)
    df_fit = pd.concat([train, val])
    return evaluate_config(df_fit, test, config)


def directed_factorial_on_test(
    data_path,
    best_config: RunConfig,
    top_models=None,
    verbose=True,
):
    """
    Factorial DIRIGIDO sobre TEST: cruza los mejores modelos x {con/sin TDA}
    x {con/sin las exogenas de la mejor config}. Tabla comparativa limpia.
    """
    df = load_dataset(data_path) if data_path else load_dataset()
    train, val, test = date_train_val_test_split(df)
    df_fit = pd.concat([train, val])

    if top_models is None:
        top_models = ["Ridge", "ElasticNet", "Lasso", "LinearRegression",
                      "GradientBoosting", "KNN"]

    tda_opts = [False, True] if gtda_available() else [False]
    exog_opts = [[], best_config.exog_cols] if best_config.exog_cols else [[]]

    rows = []
    for model_name in top_models:
        for use_tda in tda_opts:
            for exog_cols in exog_opts:
                cfg = RunConfig(
                    model_name=model_name, window=best_config.window,
                    transform=best_config.transform, exog_cols=exog_cols,
                    use_tda=use_tda, strategy=best_config.strategy,
                )
                try:
                    m = evaluate_config(df_fit, test, cfg)
                    rows.append({**cfg.to_row(), **m})
                    if verbose:
                        print(f"  {cfg.label():55s} MAE={m['mae']:.3f} R2={m['r2']:.3f}")
                except Exception as e:
                    if verbose:
                        print(f"  {cfg.label()} ERROR: {e}")

    return pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)


def save_best_model(data_path, config: RunConfig, out_dir, metrics: dict = None):
    """
    Reentrena la mejor config con TODOS los datos disponibles (train+val+test)
    y serializa el modelo + su configuracion + metricas en `out_dir`.

    Guarda:
        best_model.joblib   - el estimador sklearn entrenado
        best_config.json    - la configuracion y metricas (legible)

    El modelo guardado esta listo para predecir el futuro. Nota: se entrena con
    toda la serie (incluido test) porque para PRODUCCION/forecast real quieres
    usar toda la informacion disponible; las metricas de test ya se reportaron
    aparte de forma honesta.
    """
    import json
    from pathlib import Path

    import joblib

    from ..data.loader import TARGET, load_dataset
    from ..features.transforms import Transformer
    from ..models.sklearn_models import get_sklearn_model

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path) if data_path else load_dataset()
    series = df[TARGET].to_numpy(dtype=float)
    n = len(series)

    exog = None
    if config.exog_cols:
        present = [c for c in config.exog_cols if c in df.columns]
        if present:
            exog = df[present].to_numpy(dtype=float)

    # Construir SOLO el bloque de ajuste con toda la serie (sin tramo de eval)
    from ..features.tda import (
        TDA_FEATURE_NAMES, extract_tda_features, sanitize_tda_matrix,
    )

    tr = Transformer(config.transform)
    z = tr.fit_transform(series)
    offset = len(series) - len(z)
    window = config.window

    Xl, y, idx_fit = [], [], []
    for i in range(len(z) - window):
        Xl.append(z[i:i + window]); y.append(z[i + window]); idx_fit.append(i + window)
    Xl = np.array(Xl); y = np.array(y)

    blocks = [Xl]
    if exog is not None:
        ex = np.array([exog[idx + offset] for idx in idx_fit])
        blocks.append(ex)

    if config.use_tda:
        tda = np.zeros((len(idx_fit), len(TDA_FEATURE_NAMES)))
        for r, idx in enumerate(idx_fit):
            lvl_end = idx + offset
            win_lvl = series[max(0, lvl_end - window):lvl_end]
            tda[r] = extract_tda_features(win_lvl, config.embedding_dim, config.time_delay)
        # sanear con los propios datos (no hay test aqui)
        tda_s, _, kept = sanitize_tda_matrix(tda, tda)
        if kept:
            blocks.append(tda_s)

    X_fit = np.hstack(blocks)
    y_fit = y

    model = get_sklearn_model(config.model_name)
    model.fit(X_fit, y_fit)

    # Serializar
    model_path = out_dir / "best_model.joblib"
    joblib.dump({"model": model, "config": config}, model_path)

    config_path = out_dir / "best_config.json"
    payload = {
        "label": config.label(),
        "model_name": config.model_name,
        "window": config.window,
        "transform": config.transform,
        "exog_cols": config.exog_cols,
        "use_tda": config.use_tda,
        "strategy": config.strategy,
        "embedding_dim": config.embedding_dim,
        "time_delay": config.time_delay,
        "test_metrics": metrics or {},
    }
    with open(config_path, "w") as f:
        json.dump(payload, f, indent=2)

    return model_path, config_path
