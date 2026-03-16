from __future__ import annotations
import numpy as np

#Rechenlogik für Sekantenverfahren
class SecantSolver:
    def __init__(self):
        self.f = None
        self.x_prev = None
        self.x_cur = None
        self.k = 0
        self.tol = 0.0
        self.max_iter = 5000

    def start(self, f, x0, x1, tol):
        if x0 == x1:
            raise ValueError("x0 und x1 duerfen nicht gleich sein.")
        if tol < 0:
            raise ValueError("Toleranz muss >= 0 sein.")

        f0 = float(f(x0))
        f1 = float(f(x1))
        if not np.isfinite(f0) or not np.isfinite(f1):
            raise ValueError("f(x0) oder f(x1) ist NaN/Inf.")

        self.f = f
        self.x_prev = float(x0)
        self.x_cur = float(x1)
        self.tol = float(tol)
        self.k = 0

    def step(self):
        if self.f is None:
            return "Nicht gestartet.", None, True
        if self.k >= self.max_iter:
            return "Abbruch: Hard-Limit erreicht.", None, True

        xp, xc = self.x_prev, self.x_cur
        fp = float(self.f(xp))
        fc = float(self.f(xc))

        # Nenner prüfen, damit keine Division durch 0 entsteht
        denom = (fc - fp)
        if abs(denom) < 1e-14:
            self.k += 1
            row = (self.k, f"{xp:.12g}", f"{xc:.12g}", f"{fp:.12g}", f"{fc:.12g}", "")
            return "Abbruch: f(xk)-f(xk-1)=0 (Division durch 0).", row, True

        # Sekantenformel
        xnext = xc - fc * (xc - xp) / denom
        dx = abs(xnext - xc)
        self.k += 1

        # Tabellenzeile mit alten Werten
        row = (self.k, f"{xp:.12g}", f"{xc:.12g}", f"{fp:.12g}", f"{fc:.12g}", f"{xnext:.12g}")

        # Werte aktualisieren
        self.x_prev, self.x_cur = xc, float(xnext)

        # Abbruchbedingung über die Differenz
        if dx <= self.tol:
            return f"Fertig: |Δx|<=Tol. x≈{self.x_cur:.12g}", row, True

        return f"Iteration {self.k}: x = {self.x_cur:.12g}", row, False