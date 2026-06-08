"""
Script: run_experiment.py
Barrido factorial COMPLETO + analisis estadistico del efecto de TDA y exogenas.

A diferencia de run_optuna.py (busca el optimo rapido), este script evalua
MUCHAS combinaciones (por defecto TODAS) con TimeSeriesSplit, guardando las
metricas por fold, y luego hace pruebas de hipotesis para determinar si el
TDA y las exogenas mejoran el desempeno de forma estadisticamente significativa.

El analisis se hace sobre los folds de CV; el test 2022+ queda intacto.

ADVERTENCIA: el factorial completo (~19,200 combos x 4 folds) puede tardar
HORAS. Usa --checkpoint para poder reanudar, o reduce el espacio con --quick.

Uso:
    python scripts/run_experiment.py                       # factorial completo (lento)
    python scripts/run_experiment.py --quick               # subconjunto rapido
    python scripts/run_experiment.py --analyze-only data/processed/factorial_long.csv
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
for _l in ["tensorflow", "prophet", "cmdstanpy"]:
    logging.getLogger(_l).setLevel(logging.ERROR)

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from berry_price_tda.pipelines.experiment import run_full_factorial
from berry_price_tda.pipelines.analysis import full_analysis


def main():
    p = argparse.ArgumentParser(description="Factorial completo + analisis estadistico.")
    p.add_argument("--data", type=Path, default=Path("data/interim/berry_features.csv"))
    p.add_argument("--metric", type=str, default="mae", choices=["mae", "rmse", "mape", "r2"])
    p.add_argument("--cv-splits", type=int, default=4)
    p.add_argument("--jobs", type=int, default=-1)
    p.add_argument("--checkpoint", type=Path, default=Path("data/processed/factorial_long.csv"),
                   help="CSV de guardado incremental (permite reanudar)")
    p.add_argument("--quick", action="store_true",
                   help="Subconjunto reducido para prueba rapida")
    p.add_argument("--analyze-only", type=Path, default=None,
                   help="Saltar el barrido y solo analizar un CSV ya generado")
    p.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.analyze_only:
        df_long = pd.read_csv(args.analyze_only)
        print(f"Analizando {len(df_long)} filas de {args.analyze_only}\n")
    else:
        kw = {}
        if args.quick:
            kw = dict(
                models=["Ridge", "ElasticNet", "KNN", "GradientBoosting"],
                windows=[24, 36],
                transforms=["none", "standard", "diff"],
                exog_subsets=[[], ["ppi_fertilizers"], ["mxn_usd"],
                              ["ppi_fertilizers", "mxn_usd"]],
                tda_options=[False, True],
                strategies=["none", "winsorize"],
            )
            print("Modo --quick: subconjunto reducido.\n")

        df_long = run_full_factorial(
            data_path=args.data, cv_splits=args.cv_splits, metric=args.metric,
            n_jobs=args.jobs, checkpoint=str(args.checkpoint), verbose=True, **kw,
        )

    print("\n" + "=" * 70)
    print("  ANALISIS ESTADISTICO DEL EFECTO DE LOS FACTORES")
    print("=" * 70)
    lower_better = args.metric != "r2"
    res = full_analysis(df_long, metric=args.metric, lower_is_better=lower_better)

    print("\n--- Pruebas pareadas (Wilcoxon) ---")
    cols = ["factor", "n_pairs", "mean_diff", "p_value",
            "significativo_0.05", "cliffs_delta", "efecto", "ayuda"]
    cols = [c for c in cols if c in res["paired"].columns]
    print(res["paired"][cols].to_string(index=False))

    print("\n--- Regresion factorial (efecto controlando el resto) ---")
    print(res["regression"][["coef", "p_value", "significativo_0.05"]].to_string())

    print("\n--- Resumen interpretativo ---")
    print(res["summary"])

    res["paired"].to_csv(args.out_dir / "analisis_pruebas_pareadas.csv", index=False)
    res["regression"].to_csv(args.out_dir / "analisis_regresion.csv")
    print(f"\n[OK] Factorial largo: {args.checkpoint}")
    print(f"[OK] Pruebas pareadas: {args.out_dir / 'analisis_pruebas_pareadas.csv'}")
    print(f"[OK] Regresion: {args.out_dir / 'analisis_regresion.csv'}")


if __name__ == "__main__":
    main()
