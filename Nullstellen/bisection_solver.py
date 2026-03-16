from __future__ import annotations
import numpy as np

# Rechenlogik für das Bisektionsverfahren
class BisectionSolver:
    def __init__(self):  
        self.f = None
        self.a = None
        self.b = None
        self.k = 0
        self.tol = 0.0
        self.max_iter = 5000

    # Startet das Verfahren mit gegebener Funktion, Intervall und Toleranz
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
            raise ValueError("Bisektion braucht Vorzeichenwechsel: f(a)*f(b) <= 0.")

        
        self.f = f
        self.a = float(a)
        self.b = float(b)
        self.tol = float(tol)
        self.k = 0

    # Führt einen Schritt des Verfahrens durch, gibt Status, Zeile für Tabelle und ob fertig zurück
    def step(self):
        if self.f is None:
            return "Nicht gestartet.", None, True
        if self.k >= self.max_iter:
            return "Abbruch: Hard-Limit erreicht.", None, True

        # Kopie altes Intervall um die alten Werte noch anzeigen zu können
        a_old = self.a
        b_old = self.b
        fa = float(self.f(a_old))
        fb = float(self.f(b_old))

        # Sonderfall: Grenze = 0
        if fa == 0.0:
            self.k += 1
            return (
                f"Fertig: a ist die Nullstelle (a={a_old}).",
                (self.k, f"{a_old:.10g}", f"{b_old:.10g}", f"{a_old:.10g}", f"{fa:.10g}"),
                True,
            )

        if fb == 0.0:
            self.k += 1
            return (
                f"Fertig: b ist die Nullstelle (b={b_old}).",
                (self.k, f"{a_old:.10g}", f"{b_old:.10g}", f"{b_old:.10g}", f"{fb:.10g}"),
                True,
            )

        # Mittelpunkt berechnen
        m = 0.5 * (a_old + b_old)
        fm = float(self.f(m))
        self.k += 1

        if abs(b_old - a_old) <= self.tol:
            return (
                f"Fertig: Intervallbreite <= Tol. x≈{m:.12g}",
                (self.k, f"{a_old:.10g}", f"{b_old:.10g}", f"{m:.10g}", f"{fm:.10g}"),
                True,
            )

        # neue Grenzen setzen
        if fa * fm <= 0:
            self.b = m
        else:
            self.a = m

        status = f"Iteration {self.k}: neues Intervall [{self.a:.6g}, {self.b:.6g}]"
        row = (self.k, f"{a_old:.10g}", f"{b_old:.10g}", f"{m:.10g}", f"{fm:.10g}")
        return status, row, False