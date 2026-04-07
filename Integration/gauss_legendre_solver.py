from __future__ import annotations
import numpy as np

# Stützstellen und Gewichte
_GL_NODES = {
    1: [0.0],

    2: [-1/np.sqrt(3),
         1/np.sqrt(3)],

    3: [-np.sqrt(3/5),
         0.0,
         np.sqrt(3/5)],

    4: [-np.sqrt(3/7 + (2/7)*np.sqrt(6/5)),
        -np.sqrt(3/7 - (2/7)*np.sqrt(6/5)),
         np.sqrt(3/7 - (2/7)*np.sqrt(6/5)),
         np.sqrt(3/7 + (2/7)*np.sqrt(6/5))],

    5: [-1/3 * np.sqrt(5 + 2*np.sqrt(10/7)),
        -1/3 * np.sqrt(5 - 2*np.sqrt(10/7)),
         0.0,
         1/3 * np.sqrt(5 - 2*np.sqrt(10/7)),
         1/3 * np.sqrt(5 + 2*np.sqrt(10/7))],
}

_GL_WEIGHTS = {
    1: [2.0],

    2: [1.0,
        1.0],

    3: [5/9,
        8/9,
        5/9],

    4: [(18 - np.sqrt(30)) / 36,
        (18 + np.sqrt(30)) / 36,
        (18 + np.sqrt(30)) / 36,
        (18 - np.sqrt(30)) / 36],

    5: [(322 - 13*np.sqrt(70)) / 900,
        (322 + 13*np.sqrt(70)) / 900,
        128/225,
        (322 + 13*np.sqrt(70)) / 900,
        (322 - 13*np.sqrt(70)) / 900],
}


class Gauss_Legendre_Solver:

    def __init__(self):
        self.f       = None
        self.a       = None
        self.b       = None
        self.n       = None
        self.xi      = None   
        self.nodes   = None   
        self.weights = None
        self.fxi     = None
        self.I       = None

    def start(self, f, a, b, n):
        if not (a < b):
            raise ValueError("Es muss gelten: a < b")
        if n not in _GL_NODES:
            raise ValueError(f"n muss zwischen 1 und {max(_GL_NODES)} liegen, nicht {n}.")

        fa = float(f(a))
        fb = float(f(b))
        if not np.isfinite(fa) or not np.isfinite(fb):
            raise ValueError("f(a) oder f(b) ist NaN/Inf.")

        self.f  = f
        self.a  = float(a)
        self.b  = float(b)
        self.n  = int(n)

        self.xi      = np.array(_GL_NODES[self.n])
        self.weights = np.array(_GL_WEIGHTS[self.n])

        # Transformation: tᵢ = (b-a)/2 · xᵢ + (a+b)/2
        self.nodes = 0.5 * (b - a) * self.xi + 0.5 * (a + b)
        self.I     = None

    def step(self):
        if self.f is None:
            return "Nicht gestartet.", None, True

        a, b  = self.a, self.b
        f     = self.f
        nodes = self.nodes
        xi    = self.xi
        wi    = self.weights

        self.fxi = np.array([float(f(t)) for t in nodes])

        # Integral: (b-a)/2 · Σ wᵢ · f(tᵢ)
        self.I = 0.5 * (b - a) * float(np.dot(wi, self.fxi))

        row    = (a, b, self.n, xi, nodes, self.fxi, wi, self.I)
        status = f"Integral berechnet: {self.I:.6f}"
        done   = True

        return status, row, done