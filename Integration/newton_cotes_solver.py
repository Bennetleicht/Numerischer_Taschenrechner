from __future__ import annotations
import numpy as np


class Newton_Cotes_Solver:
    def __init__(self, verfahren, modus):
        self.f = None
        self.a = None
        self.b = None
        self.tol = 0.0
        self.m = 1
        self.verfahren = verfahren
        self.modus = modus

    def start(self, f, a, b, tol, m=1):
        if not (a < b):
            raise ValueError("Es muss gelten: a < b")
        if tol < 0:
            raise ValueError("Toleranz muss >= 0 sein.")
        if m < 1:
            raise ValueError("m muss >= 1 sein.")

        fa = float(f(a))
        fb = float(f(b))
        if not np.isfinite(fa) or not np.isfinite(fb):
            raise ValueError("f(a) oder f(b) ist NaN/Inf.")

        self.f = f
        self.a = float(a)
        self.b = float(b)
        self.tol = float(tol)
        self.m = int(m)

    def step(self):
        if self.f is None:
            return "Nicht gestartet.", None, True

        a, b, f, m = self.a, self.b, self.f, self.m

        if self.verfahren == "Trapezregel":
            xs = np.linspace(a, b, m + 1)
            ys = np.array([float(f(x)) for x in xs])
            h = (b - a) / m
            I = h * (0.5 * ys[0] + float(np.sum(ys[1:-1])) + 0.5 * ys[-1])
            xs_nodes = xs
            ys_nodes = ys

        elif self.verfahren == "Simpsonregel":
            if m % 2 != 0:
                m = m + 1
                self.m = m

            xs = np.linspace(a, b, m + 1)
            ys = np.array([float(f(x)) for x in xs])
            h = (b - a) / m
            I = (h / 3) * (
                ys[0] + ys[-1]
                + 4.0 * float(np.sum(ys[1:-1:2]))
                + 2.0 * float(np.sum(ys[2:-2:2]))
            )
            xs_nodes = xs
            ys_nodes = ys

        elif self.verfahren == "3/8-Regel":
            fa = float(f(a))
            fb = float(f(b))
            m1 = a + (b - a) / 3
            m2 = a + 2 * (b - a) / 3
            fm1 = float(f(m1))
            fm2 = float(f(m2))
            I = (b - a) * (fa + 3 * fm1 + 3 * fm2 + fb) / 8
            xs_nodes = np.array([a, m1, m2, b])
            ys_nodes = np.array([fa, fm1, fm2, fb])

        elif self.verfahren == "Milne-Regel":
            fa = float(f(a))
            fb = float(f(b))
            m1 = a + (b - a) / 4
            m2 = a + 2 * (b - a) / 4
            m3 = a + 3 * (b - a) / 4
            fm1 = float(f(m1))
            fm2 = float(f(m2))
            fm3 = float(f(m3))
            I = (b - a) * (7 * fa + 32 * fm1 + 12 * fm2 + 32 * fm3 + 7 * fb) / 90
            xs_nodes = np.array([a, m1, m2, m3, b])
            ys_nodes = np.array([fa, fm1, fm2, fm3, fb])

        else:
            raise ValueError("Unbekanntes Verfahren")

        row = (a, b, I, self.m, xs_nodes, ys_nodes)
        status = f"Integral berechnet: {I:.6f}"
        done = True
        return status, row, done