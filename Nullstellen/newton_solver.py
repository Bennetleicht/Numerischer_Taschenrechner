from __future__ import annotations
import numpy as np

#Rechenlogik für Newton-Verfahren
class NewtonSolver:
    def __init__(self):
        self.f = None
        self.df = None
        self.x = None
        self.k = 0
        self.tol = 0.0
        self.max_iter = 5000

    # Initialisierung mit Funktion, Ableitung, Startwert und Toleranz
    def start(self, f, df, x0, tol):
        if tol < 0:
            raise ValueError("Toleranz muss >= 0 sein.")

        y0 = float(f(x0))
        dy0 = float(df(x0))
        if not np.isfinite(y0) or not np.isfinite(dy0):
            raise ValueError("f(x0) oder f'(x0) ist NaN/Inf.")

        self.f = f
        self.df = df
        self.x = float(x0)
        self.tol = float(tol)
        self.k = 0

    def step(self):
        if self.f is None or self.df is None:
            return "Nicht gestartet.", None, True
        if self.k >= self.max_iter:
            return "Abbruch: Hard-Limit erreicht.", None, True

        xk = float(self.x)
        yk = float(self.f(xk))
        dyk = float(self.df(xk))

        # Sonderfall: ungültige Werte
        if not np.isfinite(yk) or not np.isfinite(dyk):
            self.k += 1
            return (
                "Abbruch: NaN/Inf in f oder f'.",
                (self.k, f"{xk:.12g}", f"{yk:.12g}", f"{dyk:.12g}", ""),
                True,
            )

        # Sonderfall: Ableitung zu klein
        if abs(dyk) < 1e-14:
            self.k += 1
            return (
                "Abbruch: f'(xk) ~ 0.",
                (self.k, f"{xk:.12g}", f"{yk:.12g}", f"{dyk:.12g}", ""),
                True,
            )

        # Newton-Formel
        xnext = xk - yk / dyk
        dx = abs(xnext - xk)
        self.k += 1
        self.x = float(xnext)

        row = (self.k, f"{xk:.12g}", f"{yk:.12g}", f"{dyk:.12g}", f"{xnext:.12g}")

        # Abbruchbedingung über die Differenz
        if dx <= self.tol:
            return f"Fertig: |Δx|<=Tol. x≈{self.x:.12g}", row, True

        return f"Iteration {self.k}: x = {self.x:.12g}", row, False