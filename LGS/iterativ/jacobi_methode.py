from __future__ import annotations

from typing import Tuple

from jacobi_solver import (
    JacobiSolver,
    JacobiRowDetail,
    Step,
)


class JacobiMethod:
    """
    Logik zwischen GUI und Solver.
    - Start eines neuen Jacobi-Solvers
    - Weiter-Schalten um genau eine Iteration
    - Bereitstellung eines einfachen Einstiegs für die GUI
    """

    title = "Jacobi-Verfahren"

    def __init__(self) -> None:
        # Reiner Solver ohne GUI-Abhaengigkeiten
        self.solver = JacobiSolver()

    def start(
        self,
        A,
        b,
        x0,
        tol: float = 1e-6,
        safety_limit: int = 100,
        eps: float = 1e-12,
    ) -> Tuple[Step, dict]:
        #Initialisiert den Solver
        self.solver.start(A, b, x0, tol=tol, safety_limit=safety_limit, eps=eps)

        # Step init
        step = Step(kind="iter", iteration=0, x_old=x0[:], x_new=x0[:], max_diff=0.0,
            message="Solver initialisiert. Bereit fuer die erste Iteration.",
            row_details=[],)

        return step, self.solver.snapshot()

    def next_step(self) -> Tuple[Step, dict]:
        # ein Schritt
        step = self.solver.step()
        snapshot = self.solver.snapshot()
        return step, snapshot


class JacobiStepper:
    def __init__( self, A, b, x0, tol: float = 1e-6, safety_limit: int = 100, eps: float = 1e-12,) -> None:
        self.solver = JacobiSolver()
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
    from gui.jacobi_gui import JacobiGUI
    app = JacobiGUI()
    app.mainloop()


if __name__ == "__main__":
    main()