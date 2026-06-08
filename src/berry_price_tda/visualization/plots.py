"""Visualización de los resultados del experimento factorial."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_metric_comparison(
    df: pd.DataFrame,
    metric: str = "mae",
    top_n: int = 15,
    ax=None,
):
    """Barras horizontales de las top_n combinaciones por una métrica."""
    ascending = metric != "r2"   # para R² mayor es mejor
    d = df.sort_values(metric, ascending=ascending).head(top_n).iloc[::-1]
    labels = d["modelo"] + " | " + d["config"]

    if ax is None:
        _, ax = plt.subplots(figsize=(10, max(4, top_n * 0.4)))

    colors = {
        "sklearn": "#378ADD",
        "keras":   "#D85A30",
        "clasico": "#1D9E75",
    }
    bar_colors = [colors.get(p, "#888780") for p in d["parte"]]

    ax.barh(labels, d[metric], color=bar_colors)
    ax.set_xlabel(metric.upper())
    ax.set_title(f"Top {top_n} combinaciones por {metric.upper()}")
    ax.grid(True, axis="x", alpha=0.3)
    return ax


def plot_factorial_heatmap(df: pd.DataFrame, metric: str = "mae", ax=None):
    """
    Heatmap modelo * (exog/tda) mostrando el efecto de cada factor.
    Promedia sobre la métrica para cada celda.
    """
    pivot = df.pivot_table(
        index="modelo", columns="config", values=metric, aggfunc="mean"
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(4, len(pivot) * 0.4)))

    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"{metric.upper()} por modelo y configuración")
    plt.colorbar(im, ax=ax, label=metric.upper())

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)
    return ax


def factor_effect_summary(df: pd.DataFrame, metric: str = "mae") -> pd.DataFrame:
    """
    Resume el efecto marginal de usar exógenas y TDA:
    promedio de la métrica con/sin cada factor.
    """
    rows = []
    for factor in ["use_exog", "use_tda"]:
        for val in [False, True]:
            sub = df[df[factor] == val]
            if len(sub) > 0:
                rows.append({
                    "factor": factor,
                    "activo": val,
                    f"{metric}_medio": sub[metric].mean(),
                    "n": len(sub),
                })
    return pd.DataFrame(rows)
