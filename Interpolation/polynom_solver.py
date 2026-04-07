from typing import List, Optional
import numpy as np


class VandermondeSolver:

    def __init__(self):
        self.x: List[float] = []
        self.f: List[float] = []
        self.coeffs: Optional[np.ndarray] = None   # [a_0, a_1, ..., a_n]
        self.V: Optional[np.ndarray] = None        # Vandermonde-Matrix

    def start(self, x: List[float], f: List[float]):
        if len(x) != len(f):
            raise ValueError("x und f müssen gleich lang sein")
        if len(x) < 2:
            raise ValueError("Mindestens 2 Stützpunkte nötig")
        if len(set(x)) != len(x):
            raise ValueError("x-Werte müssen paarweise verschieden sein")
        self.x = list(x)
        self.f = list(f)
        self.coeffs = None
        self.V = None

    def build_vandermonde(self) -> np.ndarray:
        # Baut (n+1)×(n+1) Vandermonde-Matrix
        n = len(self.x)
        V = np.zeros((n, n))
        for i, xi in enumerate(self.x):
            for j in range(n):
                V[i, j] = xi ** j
        self.V = V
        return V

    def solve(self) -> np.ndarray:
        # Löst V·a = f und speichert Koeffizienten
        if self.V is None:
            self.build_vandermonde()
        self.coeffs = np.linalg.solve(self.V, np.array(self.f))
        return self.coeffs

    def evaluate(self, t: float) -> float:
        if self.coeffs is None:
            raise RuntimeError("Noch nicht gelöst.")
        return float(sum(self.coeffs[j] * t ** j for j in range(len(self.coeffs))))

    def evaluate_array(self, t_arr) -> np.ndarray:
        if self.coeffs is None:
            raise RuntimeError("Noch nicht gelöst.")
        t_arr = np.asarray(t_arr, dtype=float)
        result = np.zeros_like(t_arr)
        for j, a in enumerate(self.coeffs):
            result += a * t_arr ** j
        return result

    def poly_string(self) -> str:
        # Gibt das Polynom als lesbaren String zurück
        if self.coeffs is None:
            return "—"
        parts = []
        for j, a in enumerate(self.coeffs):
            if abs(a) < 1e-12:
                continue
            a_str = f"{a:+.4f}"
            if j == 0:
                parts.append(f"{a:.4f}")
            elif j == 1:
                parts.append(f"{a_str}·x")
            else:
                parts.append(f"{a_str}·x^{j}")
        return "p(x) = " + " ".join(parts) if parts else "p(x) = 0"