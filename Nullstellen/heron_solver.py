from __future__ import annotations

#Rechenlogik für Heron-Verfahren
class HeronSolver:
    def __init__(self):
        self.S = None
        self.x = None
        self.k = 0
        self.tol = 0.0
        self.max_iter = 5000

    def start(self, S, x0, tol):
        if S < 0:
            raise ValueError("S muss >= 0 sein.")
        if tol < 0:
            raise ValueError("Toleranz muss >= 0 sein.")
        if x0 == 0.0 and S != 0.0:
            raise ValueError("x0 darf nicht 0 sein, wenn S != 0 ist.")

        self.S = float(S)
        self.x = float(x0)
        self.tol = float(tol)
        self.k = 0

    def step(self):
        if self.S is None:
            return "Nicht gestartet.", None, True
        if self.k >= self.max_iter:
            return "Abbruch: Hard-Limit erreicht.", None, True

        S = self.S
        xk = float(self.x)

        # Sonderfall: Wurzel aus 0
        if S == 0.0:
            self.k += 1
            self.x = 0.0
            row = (self.k, f"{xk:.12g}", f"{0.0:.12g}", f"{abs(0.0 - xk):.3e}")
            return "Fertig: sqrt(S)=0. Ergebnis x=0", row, True

        # Heron-Formel
        xnext = 0.5 * (xk + S / xk)
        dx = abs(xnext - xk)
        self.k += 1
        self.x = float(xnext)

        row = (self.k, f"{xk:.12g}", f"{xnext:.12g}", f"{dx:.3e}")

        # Abbruchbedingung über die Differenz
        if dx <= self.tol:
            return f"Fertig: |Δx|<=Tol. x≈{self.x:.12g}", row, True

        return f"Iteration {self.k}: x = {self.x:.12g}", row, False