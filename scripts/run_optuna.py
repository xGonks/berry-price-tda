"""
Script: run_optuna.py
Optimizacion de forecasting de WPUSI01102B con Optuna + factorial dirigido.

Flujo:
    1. Optuna busca la mejor config optimizando sobre VALIDACION (2020-2021).
    2. La mejor config se evalua UNA vez sobre TEST (2022+).
    3. Factorial dirigido sobre los mejores modelos, evaluado en TEST.

Espacio: modelo x ventana_grande{24,36,48} x transformacion x
         16 subconjuntos de exogenas x TDA{si,no} x estrategia anti-COVID.

Uso:
    python scripts/run_optuna.py
    python scripts/run_optuna.py --trials 300 --metric mae
    python scripts/run_optuna.py --trials 150 --no-tda
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import warnings
warnings.filterwarnings("ignore")
import logging
for _l in ["tensorflow", "prophet", "cmdstanpy", "optuna"]:
    logging.getLogger(_l).setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from berry_price_tda.pipelines.optimize import (
    run_optuna_search, evaluate_on_test, directed_factorial_on_test,
    save_best_model,
)


def main():
    p = argparse.ArgumentParser(description="Optuna + factorial dirigido WPUSI01102B.")
    p.add_argument("--data", type=Path, default=Path("data/interim/berry_features.csv"))
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--metric", type=str, default="mae", choices=["mae", "rmse", "mape", "r2"])
    p.add_argument("--no-tda", action="store_true", help="No incluir TDA en la busqueda")
    p.add_argument("--with-exog", action="store_true", help="Incluir exogenas en la busqueda (por defecto NO)")
    p.add_argument("--no-cv", action="store_true", help="Usar bloque 2020-2021 en vez de TimeSeriesSplit")
    p.add_argument("--cv-splits", type=int, default=4, help="Folds de TimeSeriesSplit")
    p.add_argument("--complexity", type=float, default=0.05, help="Peso de penalizacion por complejidad")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--jobs", type=int, default=-1,
                   help="Trials en paralelo (-1 = todos los nucleos menos uno, 1 = serie)")
    p.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--model-dir", type=Path, default=Path("models"),
                   help="Carpeta donde guardar el mejor modelo entrenado")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Busqueda con Optuna sobre validacion
    study, best, df_trials = run_optuna_search(
        data_path=args.data, n_trials=args.trials, metric=args.metric,
        allow_tda=not args.no_tda, allow_exog=args.with_exog,
        use_cv=not args.no_cv, cv_splits=args.cv_splits,
        complexity_lambda=args.complexity,
        seed=args.seed, n_jobs=args.jobs, verbose=True,
    )
    df_trials.to_csv(args.out_dir / "optuna_trials.csv", index=False)

    # 2. Evaluacion final en test
    m = evaluate_on_test(args.data, best)
    print("\n" + "=" * 70)
    print("  EVALUACION FINAL EN TEST (config ganadora de Optuna)")
    print("=" * 70)
    print(f"  Config: {best.label()}")
    print(f"  MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  MAPE={m['mape']:.2f}%  R2={m['r2']:.3f}")

    # 3. Factorial dirigido sobre test
    print("\n" + "=" * 70)
    print("  FACTORIAL DIRIGIDO SOBRE TEST (mejores modelos)")
    print("=" * 70)
    fac = directed_factorial_on_test(args.data, best, verbose=True)
    fac.to_csv(args.out_dir / "factorial_dirigido_test.csv", index=False)

    print("\n  Top 5 del factorial dirigido (por MAE):")
    cols = ["modelo", "transform", "exog", "use_tda", "estrategia", "mae", "r2"]
    cols = [c for c in cols if c in fac.columns]
    print(fac[cols].head(5).to_string(index=False))

    print(f"\n[OK] Trials guardados en: {args.out_dir / 'optuna_trials.csv'}")
    print(f"[OK] Factorial guardado en: {args.out_dir / 'factorial_dirigido_test.csv'}")

    # 4. Guardar el mejor modelo entrenado en models/
    model_path, config_path = save_best_model(args.data, best, args.model_dir, metrics=m)
    print(f"[OK] Mejor modelo guardado en: {model_path}")
    print(f"[OK] Config del modelo en: {config_path}")


if __name__ == "__main__":
    main()
