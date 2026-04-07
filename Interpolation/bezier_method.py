from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Tuple, Optional
from bezier_solver import BezierSolver


class BezierStep:
    def __init__(self, kind: str, level_or_i: int, t: float,
                 points_or_data, message: str, algorithm: str):
        self.kind = kind          
        self.level_or_i = level_or_i
        self.t = t
        self.data = points_or_data
        self.message = message
        self.algorithm = algorithm


class BezierMethod:
    title = "Bézierkurven"
    ALGORITHMS = ["De Casteljau", "Bernstein"]

    def __init__(self):
        self.solver = BezierSolver()
        self._steps: List[BezierStep] = []
        self._current = 0
        self.t_val: float = 0.5
        self.algorithm: str = "casteljau"   # standardmäßig casteljau ausgewählt
        self.started: bool = False

    def set_algorithm(self, algo: str):
        # casteljau oder bernstein
        self.algorithm = algo

    def on_start(self, values: dict) -> str:
        # eingabe prüfen
        try:
            pts = self._parse_points(values["points"])
            t = float(values["t"].replace(",", "."))
            if not (0.0 <= t <= 1.0):
                raise ValueError("t muss zwischen 0 und 1 liegen.")
        except ValueError as e:
            raise ValueError(f"Eingabefehler: {e}")

        # berechnung starten
        self.solver.start(pts)
        self.t_val = t
        self._steps = (self._build_casteljau_steps()
                       if self.algorithm == "casteljau"
                       else self._build_bernstein_steps())
        self._current = 0
        self.started = True
        algo_name = "De Casteljau" if self.algorithm == "casteljau" else "Bernstein"
        return f"{algo_name} für t={t} gestartet."

    # ein schritt
    def on_step(self) -> Tuple[str, Optional[BezierStep], bool]:
        if self._current >= len(self._steps):
            return "Fertig.", None, True

        step = self._steps[self._current]
        self._current += 1
        done = self._current >= len(self._steps)
        return step.message, step, done

    def is_done(self) -> bool:
        return self._current >= len(self._steps)

    def get_result(self) -> Optional[Tuple[float, float]]:
        if not self.started:
            return None
        return (self.solver.evaluate_casteljau(self.t_val)
                if self.algorithm == "casteljau"
                else self.solver.evaluate_bernstein(self.t_val))

    # casteljau schritte
    def _build_casteljau_steps(self) -> List[BezierStep]:
        steps = []
        t = self.t_val
        tableau = self.solver.de_casteljau_full(t)

        for level, pts in enumerate(tableau):
            if level == 0:
                msg = ("Stufe 0 – Kontrollpunkte:\n" +
                       "\n".join(f"  P_{i} = ({p[0]:.3f}, {p[1]:.3f})"
                                 for i, p in enumerate(pts)))
            else:
                prev = tableau[level - 1]
                lines = []
                for i, p in enumerate(pts):
                    lines.append(
                        f"  Q_{i}^({level}) = (1-{t})·({prev[i][0]:.3f}, {prev[i][1]:.3f})"
                        f" + {t}·({prev[i+1][0]:.3f}, {prev[i+1][1]:.3f})\n"
                        f"           = ({p[0]:.4f}, {p[1]:.4f})"
                    )
                msg = f"Stufe {level}:\n" + "\n".join(lines)

            steps.append(BezierStep("level", level, t, pts, msg, "casteljau"))

        final = tableau[-1][0]
        steps.append(BezierStep(
            "done", -1, t, [final],
            f"B(t={t}) = ({final[0]:.6f}, {final[1]:.6f})",
            "casteljau"
        ))
        return steps

    # Bernstein schritte
    def _build_bernstein_steps(self) -> List[BezierStep]:
        steps = []
        t = self.t_val
        raw = self.solver.bernstein_steps(t)
        pts = self.solver.control_points
        n = len(pts) - 1

        for s in raw:
            i = s["i"]
            b = s["b_val"]
            cx, cy = s["contrib"]
            px, py = pts[i]
            msg = (
                f"b_{{{i},{n}}}(t={t}) = C({n},{i})·{t}^{i}·(1-{t})^{n-i}\n"
                f"  = {n and __import__('math').comb(n,i)}·{t**i:.4f}·{(1-t)**(n-i):.4f}"
                f" = {b:.6f}\n\n"
                f"Beitrag: b·P_{i} = {b:.6f}·({px},{py})"
                f" = ({cx:.4f}, {cy:.4f})\n"
                f"Teilsumme: ({s['partial'][0]:.4f}, {s['partial'][1]:.4f})"
            )
            steps.append(BezierStep(
                "bernstein_basis", i, t, s, msg, "bernstein"
            ))

        final = raw[-1]["partial"]
        steps.append(BezierStep(
            "done", -1, t, [final],
            f"B(t={t}) = ({final[0]:.6f}, {final[1]:.6f})",
            "bernstein"
        ))
        return steps

    @staticmethod
    def _parse_points(raw: str) -> List[Tuple[float, float]]:
        pts = []
        for part in raw.split(";"):
            part = part.strip()
            if not part:
                continue
            coords = [v.strip() for v in part.split(",")]
            if len(coords) != 2:
                raise ValueError(f"Ungültiger Punkt: '{part}' - erwartet 'x,y'.")
            pts.append((float(coords[0]), float(coords[1])))
        return pts


def main():
    from gui.bezier_gui import BezierGUI
    app = BezierGUI()
    app.mainloop()


if __name__ == "__main__":
    main()