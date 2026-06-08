"""
Parte 1 - Modelos básicos de scikit-learn.

Cada entrada del diccionario MODELS es una factory que devuelve un
estimador nuevo (sin entrenar). Se envuelven en un Pipeline con
StandardScaler porque varios de ellos (lineales, SVR, KNN) lo requieren.
"""

from __future__ import annotations

from collections.abc import Callable

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def _scaled(estimator) -> Pipeline:
    """Envuelve un estimador con StandardScaler."""
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


# Factories: nombre -> callable que crea un estimador nuevo
SKLEARN_MODELS: dict[str, Callable[[], object]] = {
    "LinearRegression": lambda: _scaled(LinearRegression()),
    "Ridge":            lambda: _scaled(Ridge(alpha=1.0, random_state=42)),
    "Lasso":            lambda: _scaled(Lasso(alpha=0.1, random_state=42, max_iter=10000)),
    "ElasticNet":       lambda: _scaled(ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=10000)),
    "KNN":              lambda: _scaled(KNeighborsRegressor(n_neighbors=5)),
    "SVR_rbf":          lambda: _scaled(SVR(kernel="rbf", C=10.0, gamma="scale")),
    "DecisionTree":     lambda: DecisionTreeRegressor(max_depth=8, random_state=42),
    "GradientBoosting": lambda: GradientBoostingRegressor(
                            n_estimators=200, max_depth=3,
                            learning_rate=0.05, random_state=42),
}


def get_sklearn_model(name: str):
    """Devuelve un estimador nuevo por nombre."""
    if name not in SKLEARN_MODELS:
        raise KeyError(f"Modelo '{name}' no disponible. Opciones: {list(SKLEARN_MODELS)}")
    return SKLEARN_MODELS[name]()
