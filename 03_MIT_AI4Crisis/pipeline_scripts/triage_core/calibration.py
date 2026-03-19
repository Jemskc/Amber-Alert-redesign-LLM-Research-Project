from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression


class BinaryCalibrator:
    def __init__(self, method: str = "temperature"):
        self.method = method
        self.temperature = 1.0
        self.iso: IsotonicRegression | None = None

    def fit(self, probs: Iterable[float], labels: Iterable[int]) -> "BinaryCalibrator":
        probs = np.clip(np.asarray(list(probs), dtype=float), 1e-6, 1 - 1e-6)
        labels = np.asarray(list(labels), dtype=int)

        if self.method == "isotonic":
            self.iso = IsotonicRegression(out_of_bounds="clip")
            self.iso.fit(probs, labels)
            return self

        logits = np.log(probs / (1 - probs))

        def nll(temp: float) -> float:
            scaled = logits / temp
            p = 1 / (1 + np.exp(-scaled))
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))

        res = minimize(lambda x: nll(x[0]), x0=[1.0], bounds=[(0.1, 10.0)])
        self.temperature = float(res.x[0])
        return self

    def predict(self, probs: Iterable[float]) -> np.ndarray:
        probs = np.clip(np.asarray(list(probs), dtype=float), 1e-6, 1 - 1e-6)
        if self.method == "isotonic" and self.iso is not None:
            return self.iso.predict(probs)

        logits = np.log(probs / (1 - probs))
        scaled = logits / self.temperature
        return 1 / (1 + np.exp(-scaled))
