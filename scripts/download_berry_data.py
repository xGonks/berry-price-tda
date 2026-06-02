"""
Script: download_berry_data.py
Descarga y actualiza el Producer Price Index: Berries (WPUSI01102B) desde FRED.

Comportamiento:
- Si el CSV ya existe: carga los datos locales, detecta NaNs y filas faltantes,
  y descarga los datos frescos para actualizarlos.
- Si no existe: descarga la serie completa desde FRED.
- Guarda el resultado ordenado y sin duplicados.

Uso:
    python scripts/download_berry_data.py
    python scripts/download_berry_data.py --output data/raw/WPUSI01102B.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

SERIES_ID   = "WPUSI01102B"
START_DATE  = "2008-06-01"
DATE_COL    = "observation_date"
DEFAULT_OUT = Path("data/raw/WPUSI01102B.csv")
FRED_URL    = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def fetch_from_fred() -> pd.DataFrame:
    """Descarga la serie completa desde FRED vía endpoint CSV público (sin API key)."""
    url = FRED_URL.format(series_id=SERIES_ID)
    print(f"  GET {url}")
    df = pd.read_csv(url, parse_dates=[DATE_COL], index_col=DATE_COL)
    df.index = pd.DatetimeIndex(df.index)
    df.columns = [SERIES_ID]
    df = df.loc[START_DATE:]
    return df


def load_local(path: Path) -> pd.DataFrame:
    """Carga el CSV local y devuelve un DataFrame indexado por observation_date."""
    df = pd.read_csv(path, parse_dates=[DATE_COL], index_col=DATE_COL)
    df.index = pd.DatetimeIndex(df.index)
    return df


def diagnose(df: pd.DataFrame, label: str = "local") -> None:
    """Imprime un resumen del estado del DataFrame."""
    total = len(df)
    nans  = df[SERIES_ID].isna().sum()
    min_d = df.index.min().date()
    max_d = df.index.max().date()
    print(f"  [{label}] Registros: {total} | Rango: {min_d} - {max_d} | NaNs: {nans}")


def main(output: Path) -> None:
    print(f"\n{'='*57}")
    print(f"  Berry Price TDA - actualizador FRED ({SERIES_ID})")
    print(f"{'='*57}\n")

    output.parent.mkdir(parents=True, exist_ok=True)

    # 1. Cargar CSV local si existe
    if output.exists():
        print(f"[1/4] CSV encontrado en: {output}")
        local_df = load_local(output)
        diagnose(local_df, "local")
    else:
        print(f"[1/4] No existe CSV local - se descargará la serie completa.")
        local_df = None

    # 2. Descargar datos frescos desde FRED
    print(f"\n[2/4] Conectando con FRED ...")
    try:
        fresh_df = fetch_from_fred()
    except Exception as e:
        print(f"\nError al conectar con FRED: {e}")
        print("    Verifica tu conexión a internet e intenta de nuevo.")
        sys.exit(1)

    diagnose(fresh_df, "FRED ")
    print(f"  Descarga exitosa.")

    # 3. Merge
    print(f"\n[3/4] Mergeando datos ...")

    if local_df is None:
        merged = fresh_df.copy()
    else:
        merged = local_df.combine_first(fresh_df)
        merged.update(fresh_df)

    merged = merged.sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]

    nans_after = merged[SERIES_ID].isna().sum()
    print(f"  Registros finales : {len(merged)}")
    print(f"  NaNs restantes    : {nans_after}")

    if nans_after > 0:
        fechas_nan = merged[merged[SERIES_ID].isna()].index.strftime("%Y-%m-%d").tolist()
        print(f"  Fechas con NaN    : {fechas_nan}")
        print("  (Puede que FRED aún no haya publicado esos valores)")

    # 4. Guardar
    print(f"\n[4/4] Guardando en {output} ...")
    merged.index.name = DATE_COL
    merged.to_csv(output, date_format="%Y-%m-%d")
    print(f"  Listo.\n")
    print(f"{'='*57}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Descarga y actualiza {SERIES_ID} desde FRED."
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Ruta del CSV de salida (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()
    main(args.output)
