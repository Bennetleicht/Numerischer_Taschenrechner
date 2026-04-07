from typing import List, Callable, Tuple, Optional
import numpy as np

def _normalize_expr(expr_str: str) -> str:
    return expr_str.replace("^", "**")

def parse_function(expr_str: str):
    import math
    expr_str = _normalize_expr(expr_str)

    allowed = {
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
        "abs": abs, "pi": math.pi, "e": math.e,
    }

    def f(x: float) -> float:
        return eval(expr_str, {"__builtins__": {}}, {**allowed, "x": x})

    return f


class CubicSplineSolver:
    """
    Kubische Spline-Interpolation
    Randbedingungen: natural oder hermite
    """

    def __init__(self):
        self.x: List[float] = []
        self.f: List[float] = []
        self.h: List[float] = []
        self.M: List[float] = []
        self.a: List[float] = []
        self.b: List[float] = []
        self.c: List[float] = []
        self.d: List[float] = []
        self.boundary: str = "natural"
        self.df0: float = 0.0   # S'(x_0) für Hermite
        self.dfn: float = 0.0   # S'(x_n) für Hermite
        self.V: Optional[np.ndarray] = None
        self.rhs: Optional[np.ndarray] = None

    def start(self, x: List[float], f: List[float],
              boundary: str = "natural",
              df0: float = 0.0, dfn: float = 0.0):
        if len(x) != len(f):
            raise ValueError("x und f müssen gleich lang sein")
        if len(x) < 3:
            raise ValueError("Mindestens 3 Stützpunkte nötig")
        if len(set(x)) != len(x):
            raise ValueError("x-Werte müssen paarweise verschieden sein")
        xs = sorted(zip(x, f), key=lambda p: p[0])
        self.x = [p[0] for p in xs]
        self.f = [p[1] for p in xs]
        self.boundary = boundary
        self.df0 = df0
        self.dfn = dfn

    def compute(self) -> Tuple[np.ndarray, np.ndarray]:
        # Löst das Tridiagonalsystem für die Momente M
        x, f, h_list = self.x, self.f, []
        n = len(x)
        for i in range(n - 1):
            h_list.append(x[i + 1] - x[i])
        self.h = h_list

        if self.boundary == "natural":
            A, rhs = self._build_natural(n, h_list, f)
        else:
            A, rhs = self._build_hermite(n, h_list, f)

        self.V = A
        self.rhs = rhs
        M_inner = np.linalg.solve(A, rhs)

        if self.boundary == "natural":
            self.M = [0.0] + list(M_inner) + [0.0]
        else:
            self.M = list(M_inner)

        self._compute_coeffs()
        return A, rhs

    def _build_natural(self, n, h, f):
        size = n - 2
        A = np.zeros((size, size))
        rhs = np.zeros(size)
        for i in range(size):
            A[i, i] = 2 * (h[i] + h[i + 1])
            if i > 0:
                A[i, i - 1] = h[i]
            if i < size - 1:
                A[i, i + 1] = h[i + 1]
            rhs[i] = 6 * (
                (f[i + 2] - f[i + 1]) / h[i + 1] -
                (f[i + 1] - f[i]) / h[i]
            )
        return A, rhs

    def _build_hermite(self, n, h, f):
        # Vollständige Spline-Randbedingungen:
        size = n
        A = np.zeros((size, size))
        rhs = np.zeros(size)

        # Randbedingung links: h[0]/6·M_0 + h[0]/3·M_1 = (f[1]-f[0])/h[0] - df0
        A[0, 0] = h[0] / 3
        A[0, 1] = h[0] / 6
        rhs[0] = (f[1] - f[0]) / h[0] - self.df0

        # Innere Gleichungen
        for i in range(1, n - 1):
            A[i, i - 1] = h[i - 1]
            A[i, i]     = 2 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]
            rhs[i] = 6 * (
                (f[i + 1] - f[i]) / h[i] -
                (f[i] - f[i - 1]) / h[i - 1]
            )

        # Randbedingung rechts
        A[n - 1, n - 2] = h[-1] / 6
        A[n - 1, n - 1] = h[-1] / 3
        rhs[n - 1] = self.dfn - (f[-1] - f[-2]) / h[-1]

        return A, rhs

    def _compute_coeffs(self):
        x, f, h, M = self.x, self.f, self.h, self.M
        self.a, self.b, self.c, self.d = [], [], [], []
        for i in range(len(x) - 1):
            self.a.append(f[i])
            self.b.append(
                (f[i + 1] - f[i]) / h[i] - h[i] * (2 * M[i] + M[i + 1]) / 6
            )
            self.c.append(M[i] / 2)
            self.d.append((M[i + 1] - M[i]) / (6 * h[i]))

    def evaluate(self, t: float) -> float:
        x = self.x
        idx = len(x) - 2
        idx = max(0, min(len(x) - 2, int(np.searchsorted(x, t, side="right") - 1)))
        dx = t - x[idx]
        return (self.a[idx] + self.b[idx] * dx +
                self.c[idx] * dx ** 2 + self.d[idx] * dx ** 3)

    def evaluate_array(self, t_arr) -> np.ndarray:
        return np.array([self.evaluate(float(t)) for t in t_arr])