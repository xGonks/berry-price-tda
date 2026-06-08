"""
Script: download_exogenous.py
Descarga variables exógenas relevantes para predecir el PPI de berries
(WPUSI01102B) desde FRED y Banxico, las alinea en frecuencia mensual y
las une con los datos crudos del PPI en data/interim/.

Variables descargadas:
    FRED     CUSR0000SAF113  CPI Frutas frescas (mensual)     - presión de demanda
    FRED     WPU0652         PPI Fertilizantes (mensual)      - costo de insumos
    FRED     APU000074714    Precio diésel, $/galón (mensual) - costo logístico
    BANXICO  SF43718         Tipo de cambio FIX MXN/USD (diario-mensual)

Uso:
    python scripts/download_exogenous.py
    python scripts/download_exogenous.py --raw data/raw/WPUSI01102B.csv \\
                                         --output data/interim/berry_features.csv
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
BERRY_SERIES   = "WPUSI01102B"
START_DATE     = "2008-06-01"
DATE_COL       = "observation_date"
DEFAULT_RAW    = Path("data/raw/WPUSI01102B.csv")
DEFAULT_OUT    = Path("data/interim/berry_features.csv")

FRED_URL       = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
FRED_RETRIES   = 3
FRED_BACKOFF   = 4   # segundos (se duplica en cada reintento)

BANXICO_URL    = (
    "https://www.banxico.org.mx/SieAPIRest/service/v1"
    "/series/{series_id}/datos/{start}/{end}"
)
BANXICO_SERIES = "SF43718"   # Tipo de cambio FIX MXN por USD (diario)

FRED_EXOGENOUS = {
    # col_name : (series_id, descripcion)
    "cpi_fresh_fruits": ("CUSR0000SAF113", "CPI Frutas Frescas"),
    "ppi_fertilizers":  ("WPU0652",        "PPI Fertilizantes"),
    "diesel_price":     ("APU000074714",   "Precio Diésel USD/galón"),
}


# ---------------------------------------------------------------------------
# Helpers - FRED
# ---------------------------------------------------------------------------

def fetch_fred(series_id: str) -> pd.Series:
    """Descarga una serie mensual de FRED con reintentos."""
    url = FRED_URL.format(series_id=series_id)
    print(f"    GET {url}")

    last_exc = None
    wait = FRED_BACKOFF
    for attempt in range(1, FRED_RETRIES + 1):
        try:
            df = pd.read_csv(url, parse_dates=[DATE_COL], index_col=DATE_COL)
            break
        except Exception as exc:
            last_exc = exc
            if attempt < FRED_RETRIES:
                print(f"    [intento {attempt}/{FRED_RETRIES}] {exc}  - reintentando en {wait}s ...")
                time.sleep(wait)
                wait *= 2
            else:
                raise last_exc

    df.index = pd.DatetimeIndex(df.index)
    series = df.iloc[:, 0].replace(".", pd.NA)
    series = pd.to_numeric(series, errors="coerce")
    series = series.loc[START_DATE:]
    series.index = series.index.to_period("M").to_timestamp()
    return series


# ---------------------------------------------------------------------------
# Helpers - Banxico
# ---------------------------------------------------------------------------

def _load_banxico_token() -> str | None:
    """
    Busca el token de Banxico en:
      1. Variable de entorno BANXICO_TOKEN
      2. Archivo .env en el directorio de trabajo
    Devuelve el token como string, o None si no se encuentra.
    """
    token = os.environ.get("BANXICO_TOKEN")
    if token:
        return token.strip()

    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("BANXICO_TOKEN"):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    return parts[1].strip().strip('"').strip("'")
    return None


def fetch_banxico_mxn_usd(token: str) -> pd.Series:
    """
    Descarga el tipo de cambio FIX diario (SF43718) desde la API SIE de Banxico,
    resamplea a promedio mensual y devuelve pd.Series indexada al primer día del mes.
    """
    url = BANXICO_URL.format(
        series_id=BANXICO_SERIES,
        start=START_DATE,
        end=pd.Timestamp.today().strftime("%Y-%m-%d"),
    )
    print(f"    GET {url}")

    try:
        import urllib.request
        req = urllib.request.Request(
            url,
            headers={
                "Bmx-Token": token,
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = json.loads(resp.read().decode())
    except Exception as exc:
        raise RuntimeError(f"Error al conectar con Banxico: {exc}") from exc

    # Parsear respuesta JSON
    try:
        datos = raw["bmx"]["series"][0]["datos"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Respuesta inesperada de Banxico: {raw}") from exc

    records = []
    for d in datos:
        try:
            fecha = pd.to_datetime(d["fecha"], dayfirst=True)
            valor = float(d["dato"].replace(",", "."))
            records.append((fecha, valor))
        except (ValueError, KeyError):
            continue   # saltar N/E u otros no-numéricos

    if not records:
        raise RuntimeError("Banxico no devolvió observaciones válidas.")

    series = pd.Series(
        dict(records),
        name="mxn_usd",
        dtype=float,
    )
    series.index = pd.DatetimeIndex(series.index)

    # Resamplear diario - mensual (promedio)
    series = series.resample("MS").mean()
    series = series.loc[START_DATE:]
    return series


# ---------------------------------------------------------------------------
# Helpers - general
# ---------------------------------------------------------------------------

def load_berry(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=[DATE_COL], index_col=DATE_COL)
    df.index = pd.DatetimeIndex(df.index)
    df.index = df.index.to_period("M").to_timestamp()
    series = df.iloc[:, 0]
    series.name = BERRY_SERIES
    return series


def diagnose(df: pd.DataFrame) -> None:
    print(f"\n  {'Columna':<22} {'Registros':>10}  {'NaN':>6}  {'Desde':>12}  {'Hasta':>12}")
    print(f"  {'-'*66}")
    for col in df.columns:
        n    = df[col].notna().sum()
        nans = df[col].isna().sum()
        if n > 0:
            first = df[col].first_valid_index().strftime("%Y-%m-%d")
            last  = df[col].last_valid_index().strftime("%Y-%m-%d")
        else:
            first = last = "N/A"
        print(f"  {col:<22} {n:>10}  {nans:>6}  {first:>12}  {last:>12}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(raw_path: Path, output: Path) -> None:
    print(f"\n{'='*60}")
    print(f"  Berry Price TDA - descarga de variables exógenas")
    print(f"{'='*60}\n")

    if not raw_path.exists():
        print(f"[!] No se encontró {raw_path}. Ejecuta primero:\n"
              f"    python scripts/download_berry_data.py  (o: make data)\n")
        sys.exit(1)

    # 1. Cargar berry como índice temporal base
    print(f"[1/4] Cargando PPI berries desde {raw_path} ...")
    berry = load_berry(raw_path)
    print(f"      {len(berry)} observaciones  ({berry.index.min().date()} - {berry.index.max().date()})")

    exog_series: dict[str, pd.Series] = {}
    failed: list[str] = []

    # 2a. Descargar series FRED
    n_fred = len(FRED_EXOGENOUS)
    print(f"\n[2/4] Descargando {n_fred} variables desde FRED ...")
    for col_name, (sid, desc) in FRED_EXOGENOUS.items():
        print(f"\n  • {desc} [{sid}]")
        try:
            s = fetch_fred(sid)
            s.name = col_name
            exog_series[col_name] = s
            print(f"    OK - {len(s)} observaciones mensuales")
        except Exception as e:
            print(f"    ERROR: {e}")
            failed.append(col_name)

    # 2b. Tipo de cambio desde Banxico
    print(f"\n  • Tipo de cambio FIX MXN/USD [{BANXICO_SERIES}] - Banxico SIE")
    token = _load_banxico_token()
    if not token:
        print(
            "    [OMITIDA] No se encontró BANXICO_TOKEN.\n"
            "    Obtén uno gratis en:\n"
            "      https://www.banxico.org.mx/SieAPIRest/service/v1/token\n"
            "    y agrégalo a .env:  BANXICO_TOKEN=tu_token_aqui"
        )
        failed.append("mxn_usd")
    else:
        try:
            s = fetch_banxico_mxn_usd(token)
            s.name = "mxn_usd"
            exog_series["mxn_usd"] = s
            print(f"    OK - {len(s)} observaciones mensuales")
        except Exception as e:
            print(f"    ERROR: {e}")
            failed.append("mxn_usd")

    if failed:
        print(f"\n[!] No se pudieron descargar: {failed}")

    # 3. Construir DataFrame alineado
    print(f"\n[3/4] Alineando y mergeando al índice de WPUSI01102B ...")
    idx = pd.date_range(berry.index.min(), berry.index.max(), freq="MS")

    df = pd.DataFrame(index=idx)
    df.index.name = DATE_COL
    df[BERRY_SERIES] = berry.reindex(idx)

    for col_name, s in exog_series.items():
        df[col_name] = s.reindex(idx)

    df = df.ffill(limit=3).bfill(limit=1)
    diagnose(df)

    remaining_nans = df.isna().sum().sum()
    if remaining_nans > 0:
        print(f"\n  [!] Quedan {remaining_nans} NaN totales.")
    else:
        print(f"\n  Cero NaN en el dataset final.")

    # 4. Guardar
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[4/4] Guardando en {output} ...")
    df.to_csv(output, date_format="%Y-%m-%d")
    print(f"      {len(df)} filas * {len(df.columns)} columnas\n")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Descarga exógenas FRED+Banxico y construye dataset interim."
    )
    parser.add_argument("--raw", "-r", type=Path, default=DEFAULT_RAW,
                        help=f"CSV crudo de WPUSI01102B (default: {DEFAULT_RAW})")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUT,
                        help=f"CSV de salida con features (default: {DEFAULT_OUT})")
    args = parser.parse_args()
    main(args.raw, args.output)
