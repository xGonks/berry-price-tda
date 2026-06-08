"""
Analisis estadistico del efecto de los factores (TDA, exogenas, etc.).

Toma el DataFrame en formato largo (combinacion x fold) producido por
run_full_factorial y responde, con pruebas de hipotesis, si cada factor
mejora el desempeno.

Metodos:
    1. Prueba pareada de Wilcoxon (signed-rank): compara el MISMO setup con
       vs sin el factor, emparejado por (resto de factores, fold). Es pareada
       porque las observaciones NO son independientes (comparten datos/modelo).
    2. Tamano de efecto: Cliff's delta (no parametrico) + diferencia de medias.
    3. Regresion OLS / ANOVA factorial: modela la metrica como funcion de todos
       los factores a la vez -> efecto marginal de cada uno controlando el resto.

Todo se hace sobre las metricas de CV (folds), NO sobre el test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from scipy import stats
    _SCIPY = True
except ImportError:
    _SCIPY = False

try:
    import statsmodels.formula.api as smf
    _SM = True
except ImportError:
    _SM = False


# ---------------------------------------------------------------------------
# Tamano de efecto
# ---------------------------------------------------------------------------

def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cliff's delta: tamano de efecto no parametrico en [-1, 1].
    delta > 0 -> valores de `a` tienden a ser mayores que los de `b`.
    Interpretacion (Romano et al.): |d|<0.147 insignificante, <0.33 pequeno,
    <0.474 mediano, >=0.474 grande.
    """
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return np.nan
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (na * nb)


def _cliff_magnitude(d: float) -> str:
    ad = abs(d)
    if ad < 0.147: return "insignificante"
    if ad < 0.330: return "pequeno"
    if ad < 0.474: return "mediano"
    return "grande"


# ---------------------------------------------------------------------------
# Prueba pareada de un factor binario (TDA si/no, exog si/no)
# ---------------------------------------------------------------------------

def paired_factor_test(
    df_long: pd.DataFrame,
    factor: str,
    metric: str = "mae",
    lower_is_better: bool = True,
) -> dict:
    """
    Prueba si activar `factor` (booleano) cambia `metric`, de forma pareada.

    Empareja filas identicas en todo MENOS el factor (y el mismo fold), toma
    la diferencia metric(con) - metric(sin), y aplica Wilcoxon signed-rank.

    factor: nombre de columna booleana, p.ej. "use_tda".
            Para exogenas se usa la columna derivada "has_exog" (ver prepare()).
    """
    if not _SCIPY:
        raise ImportError("scipy no disponible para las pruebas.")

    # columnas que identifican un "mismo setup" excepto el factor
    base_id = ["modelo", "window", "transform", "exog", "use_tda", "estrategia", "fold"]

    if factor == "has_exog":
        # Para exogenas: comparar sin-exog vs con-exog emparejando por el resto.
        # No podemos emparejar por "exog" (es justo lo que varia), asi que
        # promediamos las configs con exog para cada combinacion del resto.
        id_cols = [c for c in ["modelo", "window", "transform", "use_tda",
                               "estrategia", "fold"] if c in df_long.columns]
        on = (df_long[df_long["has_exog"] == True]      # noqa: E712
              .groupby(id_cols, as_index=False)[metric].mean())
        off = df_long[df_long["has_exog"] == False][id_cols + [metric]]  # noqa: E712
    else:
        id_cols = [c for c in base_id if c in df_long.columns and c != factor]
        on = df_long[df_long[factor] == True]    # noqa: E712
        off = df_long[df_long[factor] == False]  # noqa: E712

    merged = pd.merge(off, on, on=id_cols, suffixes=("_off", "_on"))
    if len(merged) == 0:
        return {"factor": factor, "n_pairs": 0, "error": "sin pares comparables"}

    diff = merged[f"{metric}_on"].to_numpy() - merged[f"{metric}_off"].to_numpy()
    diff = diff[np.isfinite(diff)]
    if len(diff) < 5:
        return {"factor": factor, "n_pairs": len(diff), "error": "muy pocos pares"}

    # Wilcoxon (la H0 es que la mediana de la diferencia es 0)
    try:
        stat, p = stats.wilcoxon(diff)
    except Exception:
        stat, p = np.nan, np.nan

    d = cliffs_delta(merged[f"{metric}_on"], merged[f"{metric}_off"])
    mean_diff = float(np.mean(diff))
    median_diff = float(np.median(diff))

    # interpretacion direccional
    if lower_is_better:
        # mejora si la metrica BAJA al activar el factor (diff < 0)
        helps = mean_diff < 0
    else:
        helps = mean_diff > 0

    return {
        "factor": factor,
        "n_pairs": int(len(diff)),
        f"mean_{metric}_on": float(merged[f"{metric}_on"].mean()),
        f"mean_{metric}_off": float(merged[f"{metric}_off"].mean()),
        "mean_diff": mean_diff,
        "median_diff": median_diff,
        "wilcoxon_stat": float(stat) if np.isfinite(stat) else np.nan,
        "p_value": float(p) if np.isfinite(p) else np.nan,
        "significativo_0.05": bool(p < 0.05) if np.isfinite(p) else False,
        "cliffs_delta": float(d),
        "efecto": _cliff_magnitude(d),
        "ayuda": bool(helps),
    }


# ---------------------------------------------------------------------------
# Regresion OLS / ANOVA factorial
# ---------------------------------------------------------------------------

def factor_regression(df_long: pd.DataFrame, metric: str = "mae") -> pd.DataFrame:
    """
    Regresion OLS de la metrica contra TODOS los factores a la vez.
    Da el efecto marginal de cada factor controlando por los demas, con su
    p-valor. Equivale a un ANOVA factorial.

    Devuelve un DataFrame con coeficientes, errores estandar y p-valores.
    """
    if not _SM:
        raise ImportError("statsmodels no disponible.")

    d = df_long.copy()
    d = d[np.isfinite(d[metric])]

    # variables: factores categoricos + binarios
    d["use_tda"] = d["use_tda"].astype(int)
    d["has_exog"] = (d["n_exog"] > 0).astype(int) if "n_exog" in d else 0

    formula = (f"{metric} ~ C(modelo) + C(window) + C(transform) "
               f"+ C(estrategia) + use_tda + has_exog")
    model = smf.ols(formula, data=d).fit()

    res = pd.DataFrame({
        "coef": model.params,
        "std_err": model.bse,
        "t": model.tvalues,
        "p_value": model.pvalues,
    })
    res["significativo_0.05"] = res["p_value"] < 0.05
    return res


def prepare(df_long: pd.DataFrame) -> pd.DataFrame:
    """Anade columnas derivadas utiles para el analisis (has_exog)."""
    d = df_long.copy()
    if "n_exog" in d.columns:
        d["has_exog"] = d["n_exog"] > 0
    elif "exog" in d.columns:
        d["has_exog"] = d["exog"].astype(str) != "none"
    return d


# ---------------------------------------------------------------------------
# Reporte completo
# ---------------------------------------------------------------------------

def full_analysis(df_long: pd.DataFrame, metric: str = "mae",
                  lower_is_better: bool = True) -> dict:
    """
    Ejecuta el analisis completo y devuelve un dict con:
        - 'paired': DataFrame con las pruebas pareadas (TDA, exogenas)
        - 'regression': DataFrame de la regresion factorial
        - 'summary': texto interpretativo
    """
    d = prepare(df_long)

    paired_rows = []
    # Efecto del TDA
    if "use_tda" in d.columns and d["use_tda"].nunique() > 1:
        paired_rows.append(paired_factor_test(d, "use_tda", metric, lower_is_better))
    # Efecto de las exogenas
    if "has_exog" in d.columns and d["has_exog"].nunique() > 1:
        paired_rows.append(paired_factor_test(d, "has_exog", metric, lower_is_better))

    paired_df = pd.DataFrame(paired_rows)

    try:
        reg_df = factor_regression(d, metric)
    except Exception as e:
        reg_df = pd.DataFrame({"error": [str(e)]})

    # Resumen textual
    lines = []
    for _, r in paired_df.iterrows():
        if "error" in r and isinstance(r.get("error"), str):
            lines.append(f"- {r['factor']}: {r['error']}")
            continue
        verdict = "MEJORA" if r["ayuda"] else "EMPEORA o neutral"
        sig = "significativo" if r["significativo_0.05"] else "NO significativo"
        lines.append(
            f"- {r['factor']}: {verdict} la metrica "
            f"(diff media={r['mean_diff']:+.3f}, p={r['p_value']:.4f} [{sig}], "
            f"Cliff's d={r['cliffs_delta']:+.3f} [{r['efecto']}], n={r['n_pairs']} pares)"
        )
    summary = "\n".join(lines)

    return {"paired": paired_df, "regression": reg_df, "summary": summary}
