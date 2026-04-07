from typing import List, Tuple
import numpy as np
from math import comb


class BezierSolver:
    """
    Bézierkurven - zwei Berechnungswege:
      1. de Casteljau   (rekursiv, numerisch stabil)
      2. Bernstein      (explizite Basispolynome)
    Beide liefern exakt dieselbe Kurve
    """

    def __init__(self):
        self.control_points: List[Tuple[float, float]] = []

    def start(self, points: List[Tuple[float, float]]):
        if len(points) < 2:
            raise ValueError("Mindestens 2 Kontrollpunkte nötig.")
        self.control_points = list(points)

    # Casteljau
    def de_casteljau_full(self, t: float) -> List[List[Tuple[float, float]]]:
        # Vollständiges Tableau aller Zwischenstufen
        tableau = [list(self.control_points)]
        current = list(self.control_points)
        while len(current) > 1:
            current = [
                ((1 - t) * current[i][0] + t * current[i + 1][0],
                 (1 - t) * current[i][1] + t * current[i + 1][1])
                for i in range(len(current) - 1)
            ]
            tableau.append(current)
        return tableau

    def evaluate_casteljau(self, t: float) -> Tuple[float, float]:
        return self.de_casteljau_full(t)[-1][0]

    # Bernstein
    def bernstein_basis(self, i: int, n: int, t: float) -> float:
        # b_{i,n}(t) = C(n,i) · t^i · (1-t)^(n-i)
        return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

    def bernstein_basis_array(self, i: int, n: int, t_arr: np.ndarray) -> np.ndarray:
        return comb(n, i) * (t_arr ** i) * ((1 - t_arr) ** (n - i))

    def evaluate_bernstein(self, t: float) -> Tuple[float, float]:
        pts = self.control_points
        n = len(pts) - 1
        x = sum(self.bernstein_basis(i, n, t) * pts[i][0] for i in range(n + 1))
        y = sum(self.bernstein_basis(i, n, t) * pts[i][1] for i in range(n + 1))
        return (x, y)

    def bernstein_steps(self, t: float) -> List[dict]:
        pts = self.control_points
        n = len(pts) - 1
        steps = []
        partial_x, partial_y = 0.0, 0.0
        for i in range(n + 1):
            b = self.bernstein_basis(i, n, t)
            cx = b * pts[i][0]
            cy = b * pts[i][1]
            partial_x += cx
            partial_y += cy
            steps.append({
                "i": i, "n": n,
                "b_val": b,
                "contrib": (cx, cy),
                "partial": (partial_x, partial_y),
            })
        return steps

    # für beide
    def curve_points(self, algorithm: str = "casteljau",
                     num: int = 300) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.control_points) < 2:
            return np.array([], dtype=float), np.array([], dtype=float)

        num = max(2, int(num))
        t_vals = np.linspace(0, 1, num)
        if algorithm == "bernstein":
            pts = self.control_points
            n = len(pts) - 1
            xs = np.zeros(num)
            ys = np.zeros(num)
            for i, p in enumerate(pts):
                b = self.bernstein_basis_array(i, n, t_vals)
                xs += b * p[0]
                ys += b * p[1]
        else:
            results = [self.evaluate_casteljau(float(t)) for t in t_vals]
            xs = np.array([r[0] for r in results])
            ys = np.array([r[1] for r in results])
        return xs, ys
    
    def evaluate(self, algorithm: str, t: float) -> Tuple[float, float]:
        if algorithm == "bernstein":
            return self.evaluate_bernstein(t)
        return self.evaluate_casteljau(t)