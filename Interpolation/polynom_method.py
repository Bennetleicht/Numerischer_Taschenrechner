from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Optional
from polynom_solver import VandermondeSolver


class PolynomStep:
    def __init__(self, kind: str, message: str, row_i: int = -1, data: dict | None = None):
        self.kind = kind
        self.message = message
        self.row_i = row_i
        self.data = data or {}


class PolynomMethod:
    title = "Polynominterpolation (Vandermonde)"

    def __init__(self):
        self.solver = VandermondeSolver()
        self._steps: List[PolynomStep] = []
        self._current = 0

    def on_start(self, values: dict):
        try:
            points = values["points"]
            x = [float(px) for px, _ in points]
            f = [float(py) for _, py in points]
        except Exception:
            raise ValueError("Ungültige Punktliste")
        
        self.solver.start(x, f)    
        self.solver.build_vandermonde()
        self.solver.solve()
        self._steps = self._build_steps()
        self._current = 0

    def on_step(self) -> tuple[PolynomStep, bool]:
        if self._current >= len(self._steps):
            return PolynomStep("done", "Fertig."), True

        step = self._steps[self._current]
        self._current += 1
        done = self._current >= len(self._steps)
        return step, done

    def is_done(self) -> bool:
        return self._current >= len(self._steps)
    
    def input_fields(self):
        return []

    def _build_steps(self) -> List[PolynomStep]:
        steps: List[PolynomStep] = []
        x = self.solver.x
        coeffs = self.solver.coeffs
        n = len(x)

        for i in range(n):
            msg = f"Zeile {i} der Vandermonde-Matrix aufgebaut."
            steps.append(PolynomStep("vandermonde", msg, row_i=i))

        steps.append(PolynomStep("coeffs", _poly_descending(coeffs)))

        final_msg = "✓ " + _poly_descending(coeffs)
        steps.append(PolynomStep("done", final_msg))

        return steps
    
def _poly_descending(coeffs) -> str:
    # coeffs = [a_0, a_1, ..., a_n] aufsteigend 
    if coeffs is None:
        return "—"
    n = len(coeffs) - 1
    parts = []
    for j in range(n, -1, -1):
        a = float(coeffs[j])
        if abs(a) < 1e-10:
            continue
        a_abs = f"{abs(a):.4f}".rstrip("0").rstrip(".")
        if not a_abs:
            a_abs = "0"
        if j == 0:
            term = a_abs
        elif j == 1:
            term = f"{a_abs}·x"
        else:
            term = f"{a_abs}·x^{j}"
        if not parts:
            parts.append(f"-{term}" if a < 0 else term)
        else:
            parts.append(f" - {term}" if a < 0 else f" + {term}")
    return "p(x) = " + "".join(parts) if parts else "p(x) = 0"

def main():
    from gui.polynom_gui import PolynomGUI
    app = PolynomGUI()
    app.mainloop()


if __name__ == "__main__":
    main()