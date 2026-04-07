from __future__ import annotations
from typing import List
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lagrange_solver import LagrangeSolver


class LagrangeStep:
    def __init__(self, kind: str, index: int, message: str):
        self.kind = kind
        self.index = index
        self.message = message


class LagrangeMethod:
    title = "Lagrange-Interpolation"

    def __init__(self):
        self.solver = LagrangeSolver()
        self._steps: List[LagrangeStep] = []
        self._current = 0

    def input_fields(self):
        return []

    def on_start(self, values: dict):
        try:
            points = values["points"]
            x = [float(px) for px, _ in points]
            f = [float(py) for _, py in points]
        except Exception:
            raise ValueError("Ungültige Punktliste")

        self.solver.start(x, f)
        self._steps = self._build_steps()
        self._current = 0

    def on_step(self) -> tuple[LagrangeStep, bool]:
        if self._current >= len(self._steps):
            return LagrangeStep("done", -1, "Fertig."), True

        step = self._steps[self._current]
        self._current += 1
        done = self._current >= len(self._steps)
        return step, done

    def is_done(self) -> bool:
        return self._current >= len(self._steps)

    def _build_steps(self) -> List[LagrangeStep]:
        steps: List[LagrangeStep] = []
        x = self.solver.x
        f = self.solver.f
        n = len(x)

        for i in range(n):
            den = 1.0
            for j in range(n):
                if j != i:
                    den *= (x[i] - x[j])

            numerator = " · ".join(
                f"(x - {x[j]})" for j in range(n) if j != i
            )

            msg = (
                f"L_{i}(x) = {numerator} / {den:.4f}\n"
                f"Beitrag zu p(x): {f[i]} · L_{i}(x)"
            )

            steps.append(LagrangeStep(
                kind="basis",
                index=i,
                message=msg
            ))

        expand_lines = []
        for i in range(n):
            den = 1.0
            for j in range(n):
                if j != i:
                    den *= (x[i] - x[j])

            coeff = f[i] / den
            factors_str = " · ".join(
                f"(x - {x[j]})" for j in range(n) if j != i
            )

            expand_lines.append(f"{coeff:+.4f} · {factors_str}")

        expand_msg = "p(x) = Σ f_i · L_i(x)\n\n" + "\n".join(expand_lines)
        steps.append(LagrangeStep("expand", -1, expand_msg))

        poly_coeffs = self.solver.polynomial_coeffs()
        poly_str = _format_poly_asc(poly_coeffs)
        steps.append(LagrangeStep("done", -1, poly_str))

        return steps


def _format_poly_asc(coeffs) -> str:
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
    from gui.lagrange_gui import LagrangeGUI
    app = LagrangeGUI()
    app.mainloop()


if __name__ == "__main__":
    main()