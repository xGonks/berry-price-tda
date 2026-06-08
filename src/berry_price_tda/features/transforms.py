"""
Transformaciones de la serie objetivo.

Cada transformacion implementa un par (forward, inverse) para poder
entrenar sobre la serie transformada y reconstruir el nivel original al
predecir. Todas operan sobre arrays 1D y guardan los parametros que
necesitan para invertir.

Transformaciones disponibles:
    none      : sin cambios
    standard  : z-score  (x - mu) / sigma
    minmax    : escala a [0, 1]
    log       : log natural (requiere valores > 0)
    diff       : primera diferencia  x[t] - x[t-1]
    log_diff  : diferencia del log (≈ retorno logaritmico)
    seasonal_diff : diferencia estacional  x[t] - x[t-12]

El objetivo de transformar es estabilizar la serie (varianza/tendencia)
para que los modelos lineales y las redes generalicen mejor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SEASONAL_PERIOD = 12


@dataclass
class Transformer:
    """
    Transformador con estado (guarda parametros del ajuste para invertir).

    Uso:
        tr = Transformer("standard")
        z = tr.fit_transform(serie)        # transforma
        x = tr.inverse_transform(z)        # reconstruye

    Para transformaciones que reducen la longitud (diff, seasonal_diff),
    inverse_transform requiere el contexto (los valores previos) que se
    pasan en `history`.
    """
    kind: str = "none"
    _params: dict = field(default_factory=dict)

    # -- forward -----------------------------------------------------------

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        k = self.kind

        if k == "none":
            return x.copy()

        if k == "standard":
            mu, sigma = x.mean(), x.std() + 1e-8
            self._params = {"mu": mu, "sigma": sigma}
            return (x - mu) / sigma

        if k == "minmax":
            lo, hi = x.min(), x.max()
            rng = (hi - lo) + 1e-8
            self._params = {"lo": lo, "rng": rng}
            return (x - lo) / rng

        if k == "log":
            if np.any(x <= 0):
                raise ValueError("log requiere valores estrictamente positivos.")
            return np.log(x)

        if k == "diff":
            self._params = {"x0": x[0]}
            return np.diff(x)                       # longitud n-1

        if k == "log_diff":
            if np.any(x <= 0):
                raise ValueError("log_diff requiere valores positivos.")
            lx = np.log(x)
            self._params = {"lx0": lx[0]}
            return np.diff(lx)                      # longitud n-1

        if k == "seasonal_diff":
            self._params = {"head": x[:SEASONAL_PERIOD].copy()}
            return x[SEASONAL_PERIOD:] - x[:-SEASONAL_PERIOD]   # longitud n-12

        raise ValueError(f"Transformacion desconocida: {k}")

    # -- inverse -----------------------------------------------------------

    def inverse_transform(self, z: np.ndarray, history: np.ndarray | None = None) -> np.ndarray:
        """
        Reconstruye el nivel original.

        Para diff / log_diff / seasonal_diff se necesita `history`: el tramo
        de la serie ORIGINAL inmediatamente anterior a las predicciones, para
        anclar la suma acumulada / el lag estacional.
        """
        z = np.asarray(z, dtype=float)
        k = self.kind

        if k == "none":
            return z.copy()
        if k == "standard":
            return z * self._params["sigma"] + self._params["mu"]
        if k == "minmax":
            return z * self._params["rng"] + self._params["lo"]
        if k == "log":
            return np.exp(z)

        if k in ("diff", "log_diff"):
            if history is None or len(history) < 1:
                raise ValueError(f"{k}.inverse_transform requiere history.")
            anchor = history[-1]
            if k == "diff":
                return anchor + np.cumsum(z)
            else:  # log_diff
                return np.exp(np.log(anchor) + np.cumsum(z))

        if k == "seasonal_diff":
            if history is None or len(history) < SEASONAL_PERIOD:
                raise ValueError("seasonal_diff.inverse_transform requiere >=12 de history.")
            out = np.zeros(len(z))
            buf = list(history)
            for i, dz in enumerate(z):
                out[i] = buf[-SEASONAL_PERIOD] + dz
                buf.append(out[i])
            return out

        raise ValueError(f"Transformacion desconocida: {k}")


# Catalogo de transformaciones a barrer en el experimento
ALL_TRANSFORMS = ["none", "standard", "minmax", "log", "diff", "log_diff", "seasonal_diff"]

# Subset recomendado (las que suelen ayudar en series con tendencia)
RECOMMENDED_TRANSFORMS = ["none", "standard", "diff", "log_diff", "seasonal_diff"]
