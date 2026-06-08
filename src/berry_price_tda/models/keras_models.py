"""
Parte 2 - Redes neuronales (TensorFlow / Keras).

Cinco arquitecturas para forecasting t+1:
    1. MLP          - perceptrón multicapa sobre el vector de features plano
    2. LSTM         - red recurrente Long Short-Term Memory
    3. GRU          - red recurrente Gated Recurrent Unit
    4. CNN1D        - red convolucional 1D sobre la secuencia
    5. CNN_LSTM     - híbrida: convolución seguida de LSTM

Las arquitecturas recurrentes/convolucionales esperan entrada con forma
(n_samples, timesteps, n_channels). El builder de features entrega una
matriz 2D plana; el wrapper KerasForecaster se encarga de reorganizarla.

giotto-tda y keras se importan de forma perezosa para no penalizar el
arranque cuando solo se usan modelos de sklearn.
"""

from __future__ import annotations

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    _TF_AVAILABLE = True
    tf.get_logger().setLevel("ERROR")
except ImportError:  # pragma: no cover
    _TF_AVAILABLE = False


KERAS_ARCHITECTURES = ["MLP", "LSTM", "GRU", "CNN1D", "CNN_LSTM"]


def tf_available() -> bool:
    return _TF_AVAILABLE


# ---------------------------------------------------------------------------
# Definición de arquitecturas
# ---------------------------------------------------------------------------

def _build_mlp(n_features: int):
    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def _build_lstm(timesteps: int, n_channels: int):
    model = keras.Sequential([
        layers.Input(shape=(timesteps, n_channels)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def _build_gru(timesteps: int, n_channels: int):
    model = keras.Sequential([
        layers.Input(shape=(timesteps, n_channels)),
        layers.GRU(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.GRU(32),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def _build_cnn1d(timesteps: int, n_channels: int):
    model = keras.Sequential([
        layers.Input(shape=(timesteps, n_channels)),
        layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        layers.Conv1D(32, kernel_size=3, activation="relu", padding="same"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def _build_cnn_lstm(timesteps: int, n_channels: int):
    model = keras.Sequential([
        layers.Input(shape=(timesteps, n_channels)),
        layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=2),
        layers.LSTM(32),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# ---------------------------------------------------------------------------
# Wrapper estilo sklearn
# ---------------------------------------------------------------------------

class KerasForecaster:
    """
    Envuelve una arquitectura Keras con API tipo sklearn (.fit / .predict).

    Internamente:
      - Escala X e y con estadísticos de entrenamiento (z-score).
      - Reorganiza X plano a 3D para arquitecturas secuenciales.
      - Usa EarlyStopping sobre una fracción de validación.

    Parameters
    ----------
    architecture : str
        Una de KERAS_ARCHITECTURES.
    window_size : int
        Tamaño de la ventana de lags (para reconstruir la dimensión temporal).
    n_exog : int
        Número de columnas exógenas presentes en X.
    n_tda : int
        Número de columnas TDA presentes en X.
    epochs, batch_size, patience, verbose : hiperparámetros de entrenamiento.
    """

    def __init__(
        self,
        architecture: str,
        window_size: int,
        n_exog: int = 0,
        n_tda: int = 0,
        epochs: int = 150,
        batch_size: int = 16,
        patience: int = 15,
        verbose: int = 0,
        seed: int = 42,
    ):
        if not _TF_AVAILABLE:
            raise ImportError("TensorFlow no está instalado.")
        if architecture not in KERAS_ARCHITECTURES:
            raise ValueError(f"Arquitectura '{architecture}' no válida.")

        self.architecture = architecture
        self.window_size = window_size
        self.n_exog = n_exog
        self.n_tda = n_tda
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.verbose = verbose
        self.seed = seed

        self.model_ = None
        self._x_mean = None
        self._x_std = None
        self._y_mean = None
        self._y_std = None

    # -- reorganización de features ----------------------------------------

    def _to_sequence(self, X: np.ndarray) -> np.ndarray:
        """
        Convierte X plano (n, window + n_exog + n_tda) en 3D
        (n, timesteps, channels) para LSTM/GRU/CNN.

        Estrategia: los lags forman la secuencia temporal (1 canal);
        las exógenas y TDA se repiten como canales constantes a lo largo
        del tiempo, de modo que cada timestep ve también el contexto.
        """
        n = X.shape[0]
        lags = X[:, : self.window_size]                      # (n, win)
        extra = X[:, self.window_size :]                     # (n, n_exog+n_tda)

        # Canal 0: la secuencia de lags
        seq = lags.reshape(n, self.window_size, 1)

        n_extra = extra.shape[1]
        if n_extra > 0:
            # Repetir cada feature extra en todos los timesteps
            extra_rep = np.repeat(
                extra[:, np.newaxis, :], self.window_size, axis=1
            )  # (n, win, n_extra)
            seq = np.concatenate([seq, extra_rep], axis=2)

        return seq  # (n, win, 1 + n_extra)

    def _build(self, X: np.ndarray):
        keras.utils.set_random_seed(self.seed)
        if self.architecture == "MLP":
            return _build_mlp(X.shape[1])

        seq = self._to_sequence(X)
        timesteps, n_channels = seq.shape[1], seq.shape[2]
        builder = {
            "LSTM": _build_lstm,
            "GRU": _build_gru,
            "CNN1D": _build_cnn1d,
            "CNN_LSTM": _build_cnn_lstm,
        }[self.architecture]
        return builder(timesteps, n_channels)

    # -- API sklearn -------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Escalado z-score con estadísticos de entrenamiento
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0) + 1e-8
        self._y_mean = y.mean()
        self._y_std = y.std() + 1e-8

        Xs = (X - self._x_mean) / self._x_std
        ys = (y - self._y_mean) / self._y_std

        self.model_ = self._build(Xs)

        X_in = Xs if self.architecture == "MLP" else self._to_sequence(Xs)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                restore_best_weights=True,
            )
        ]

        # Validación temporal: último 15% del train (sin shuffle)
        val_frac = 0.15
        self.model_.fit(
            X_in, ys,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=val_frac,
            shuffle=False,
            callbacks=callbacks,
            verbose=self.verbose,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Xs = (X - self._x_mean) / self._x_std
        X_in = Xs if self.architecture == "MLP" else self._to_sequence(Xs)
        preds_scaled = self.model_.predict(X_in, verbose=0).flatten()
        return preds_scaled * self._y_std + self._y_mean
