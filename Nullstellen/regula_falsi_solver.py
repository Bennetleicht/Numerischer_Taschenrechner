from __future__ import annotations
import numpy as np

#Rechenlogik für Regula Falsi 
class RegulaFalsiSolver:
    def __init__(self):
        self.f = None
        self.a = None
        self.b = None
        self.k = 0
        self.tol = 0.0
        self.max_iter = 5000
        self.last_x = None

    def start(self, f, a, b, tol):
        if not (a < b):
            raise ValueError("Es muss gelten: a < b")
        if tol < 0:
            raise ValueError("Toleranz muss >= 0 sein.")

        fa = float(f(a))
        fb = float(f(b))
        if not np.isfinite(fa) or not np.isfinite(fb):
            raise ValueError("f(a) oder f(b) ist NaN/Inf.")
        if fa * fb > 0 and fa != 0.0 and fb != 0.0:
            raise ValueError("Regula Falsi braucht Vorzeichenwechsel: f(a)*f(b) <= 0.")

        self.f = f
        self.a = float(a)
        self.b = float(b)
        self.tol = float(tol)
        self.k = 0
        self.last_x = None

    def step(self):
        if self.f is None:
            return "Nicht gestartet.", None, True
        if self.k >= self.max_iter:
            return "Abbruch: Hard-Limit erreicht.", None, True

        a, b = self.a, self.b
        fa = float(self.f(a))
        fb = float(self.f(b))

        # Sonderfall: Grenze = 0
        if fa == 0.0:
            self.k += 1
            row = (self.k, f"{a:.10g}", f"{b:.10g}", f"{fa:.10g}", f"{fb:.10g}", f"{a:.10g}", f"{fa:.10g}")
            return f"Fertig: a ist Nullstelle (a={a}).", row, True

        if fb == 0.0:
            self.k += 1
            row = (self.k, f"{a:.10g}", f"{b:.10g}", f"{fa:.10g}", f"{fb:.10g}", f"{b:.10g}", f"{fb:.10g}")
            return f"Fertig: b ist Nullstelle (b={b}).", row, True

        # Nenner prüfen, damit keine Division durch 0 entsteht
        denom = fb - fa
        if abs(denom) < 1e-14:
            self.k += 1
            row = (self.k, f"{a:.10g}", f"{b:.10g}", f"{fa:.10g}", f"{fb:.10g}", "", "")
            return "Abbruch: f(b)-f(a)=0 (Division durch 0).", row, True

        # Regula-Falsi-Formel
        x_new = (a * fb - b * fa) / denom
        fx = float(self.f(x_new))
        self.k += 1
        self.last_x = x_new

        # neue Grenzen setzen
        if fa * fx < 0:
            self.b = x_new
        else:
            self.a = x_new

        row = (self.k, f"{a:.10g}", f"{b:.10g}", f"{fa:.10g}", f"{fb:.10g}", f"{x_new:.10g}", f"{fx:.10g}")

        # Abbruchbedingung
        if abs(fx) <= self.tol:
            return f"Fertig: |f(x)|<=Tol. x≈{x_new:.12g}", row, True

        return f"Iteration {self.k}: neues Intervall [{self.a:.6g}, {self.b:.6g}]", row, False