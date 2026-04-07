from typing import List
import numpy as np

class LagrangeSolver:
    def __init__(self):
        self.x: List[float] = []
        self.f: List[float] = []

    def start(self, x: List[float], f: List[float]):
        self.validate_input(x, f)
        self.x = list(x)
        self.f = list(f)

    def evaluate(self, t: float) -> float:
        return sum(self._L_i_scalar(i, t) * self.f[i] for i in range(len(self.x)))

    def _L_i_scalar(self, i: int, t: float) -> float:
        xi = self.x[i]
        result = 1.0
        for j, xj in enumerate(self.x):
            if j != i:
                result *= (t - xj) / (xi - xj)
        return result

    def evaluate_array(self, t_arr) -> np.ndarray:
        t_arr = np.asarray(t_arr, dtype=float)
        result = np.zeros_like(t_arr)
        for i in range(len(self.x)):
            Li = np.ones_like(t_arr)
            for j, xj in enumerate(self.x):
                if j != i:
                    Li *= (t_arr - xj) / (self.x[i] - xj)
            result += Li * self.f[i]
        return result

    def polynomial_coeffs(self) -> np.ndarray:
        n = len(self.x)
        total = np.zeros(n)
        for i in range(n):
            den = 1.0
            for j in range(n):
                if j != i:
                    den *= (self.x[i] - self.x[j])

            roots = [self.x[j] for j in range(n) if j != i]
            p = np.poly1d(roots, r=True)
            c_asc = p.coeffs[::-1]
            total[:len(c_asc)] += (self.f[i] / den) * c_asc
        return total

    def validate_input(self, x: List[float], f: List[float]):
        if x is None or f is None:
            raise ValueError("x und f müssen existieren")
        if len(x) != len(f):
            raise ValueError("x und f müssen gleich viele Punkte enthalten")
        if len(x) < 2:
            raise ValueError("Mindestens 2 Stützpunkte nötig")
        if len(set(x)) != len(x):
            raise ValueError("x-Werte müssen paarweise verschieden sein")