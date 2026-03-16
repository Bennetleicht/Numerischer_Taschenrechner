from __future__ import annotations

from typing import Tuple

from gauss_seidel_solver import (
    GaussSeidelSolver,
    Step,
)


class GaussSeidelMethod:
    """
    LOgik zwischen GUI und Solver.
    - Start eines neuen Gauß-Seidel-Solvers
    - Weiter-Schalten um genau eine Iteration
    - Bereitstellung eines einfachen Einstiegs für die GUI
    """

    title = "Gauß-Seidel-Verfahren"

    def __init__(self) -> None:
        # Reiner Solver ohne GUI-Abhängigkeiten
        self.solver = GaussSeidelSolver()

    def start(self, A, b, x0, tol: float = 1e-6, safety_limit: int = 100, eps: float = 1e-12,) -> Tuple[Step, dict]:
        #Initialisiert den Solver und liefert Startzustand für die GUI.
        self.solver.start(A, b, x0, tol=tol, safety_limit=safety_limit, eps=eps)

        # erster Step
        step = Step(kind="iter", iteration=0, x_old=x0[:], x_new=x0[:], max_diff=0.0,
            message="Solver initialisiert. Bereit für die erste Iteration.",row_details=[],)

        return step, self.solver.snapshot()

    def next_step(self) -> Tuple[Step, dict]:
        # eine Schritt
        step = self.solver.step()
        snapshot = self.solver.snapshot()
        return step, snapshot


class GaussSeidelStepper:
    def __init__(self, A, b, x0, tol: float = 1e-6, safety_limit: int = 100, eps: float = 1e-12,) -> None:
        self.solver = GaussSeidelSolver()
        self.solver.start(A, b, x0, tol=tol, safety_limit=safety_limit, eps=eps)

    @property
    def A(self):
        return self.solver.A

    @property
    def b(self):
        return self.solver.b

    @property
    def x(self):
        return self.solver.x

    @property
    def iteration(self):
        return self.solver.iteration

    def next_step(self) -> Step:
        # ein Schritt
        return self.solver.step()


def main():
    from gui.gauss_seidel_gui import GaussSeidelGUI
    app = GaussSeidelGUI()
    app.mainloop()


if __name__ == "__main__":
    main()