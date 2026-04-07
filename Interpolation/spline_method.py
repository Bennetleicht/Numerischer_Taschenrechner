from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Tuple, Optional
import numpy as np
from spline_solver import CubicSplineSolver, parse_function


class SplineStep:
    def __init__(self, kind: str, index: int, message: str):
        self.kind = kind
        self.index = index
        self.message = message


class SplineMethod:
    title = "Kubische Spline-Interpolation"
    BOUNDARIES = ["Natürlich", "Hermite"]

    def __init__(self):
        self.solver = CubicSplineSolver()
        self._steps: List[SplineStep] = []
        self._current = 0
        self.started: bool = False
        self._A: Optional[np.ndarray] = None
        self._rhs: Optional[np.ndarray] = None
        self.boundary: str = "natural"
        self.func_str: str = ""


    def on_start(self, func_str: str, a: float, b: float, n_intervals: int, 
             boundary: str = "natural", df0: float = 0.0, dfn: float = 0.0):

        self.func_str = func_str
        self.boundary = boundary

        # Stützpunkte berechnen
        n_points = n_intervals + 1
        f_scalar = parse_function(func_str)

        xs = np.linspace(a, b, n_points).tolist()
        
        try:
            fs = [f_scalar(xi) for xi in xs]
        except Exception as e:
            raise ValueError(f"Funktion konnte nicht ausgewertet werden: {e}")

        self.solver.start(xs, fs, boundary=boundary, df0=df0, dfn=dfn)
        self._A, self._rhs = self.solver.compute()
        self._steps = self._build_steps()
        self._current = 0
        self.started = True

        bnd_name = "natürlich" if boundary == "natural" else "Hermite"
        return (f"Spline ({bnd_name}, n={n_intervals}) aufgestellt.",
                None, False)

    def on_step(self) -> Tuple[str, Optional[SplineStep], bool]:
        if self._current >= len(self._steps):
            return "Fertig.", None, True
        step = self._steps[self._current]
        self._current += 1
        done = self._current >= len(self._steps)
        return step.message, step, done

    def is_done(self) -> bool:
        return self._current >= len(self._steps)

    def _build_steps(self) -> List[SplineStep]:
        steps = []
        solver = self.solver
        n = len(solver.x)

        # Schritt 1: h-Formel + n
        steps.append(SplineStep("setup", 0, "setup_h"))

        # Schritte 2: n+1: je ein Stützpunkt
        for i in range(n):
            steps.append(SplineStep("knot", i, f"knot_{i}"))

        # Randbedingung
        steps.append(SplineStep("boundary", 0, self.boundary))

        # Tridiagonalsystem (vollständig)
        steps.append(SplineStep("system", 0, "system"))

        # Reduziertes System nach Momenten (nur natural)
        if self.boundary == "natural":
            steps.append(SplineStep("system_reduced", 0, "system_reduced"))

        # Momente
        steps.append(SplineStep("moment", 0, "moment"))

        # Koeffizienten pro Segment
        for i in range(n - 1):
            steps.append(SplineStep("coeffs", i, f"coeffs_{i}"))

        # Alle Segmente zusammen
        steps.append(SplineStep("all_segments", 0, "all_segments"))

        # Done
        steps.append(SplineStep("done", -1, "done"))
        return steps


def main():
    from gui.spline_gui import SplineGUI
    app = SplineGUI()
    app.mainloop()


if __name__ == "__main__":
    main()