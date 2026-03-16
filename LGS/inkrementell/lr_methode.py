from __future__ import annotations
from typing import Any, Dict, List, Optional
from lr_solver import LRSolver, Step


class LRMethod:
    """
    Dünner Wrapper zwischen GUI und LRSolver
    Die Klasse kapselt:
    - Start eines neuen Solvers
    - genau einen Weiter-Schritt
    - Zugriff auf Snapshot und Loesung
    """

    title = "LR-Zerlegung"
    plotter_kind = None

    def __init__(self) -> None:
        # leere Methodenklasse ohne aktiven Solver
        self.solver: Optional[LRSolver] = None
        self.last_step: Optional[Step] = None

    def start(self, A: List[List[float]], b: List[float], tol: float = 1e-12, change_tol: float = 1e-10,) -> Step:
        # Startet eine neue LR-Zerlegung
        self.solver = LRSolver(A=A, b=b, tol=tol, change_tol=change_tol)
        self.last_step = Step(
            kind="init",
            pivot=(0, 0),
            message="Solver initialisiert. Bereit fuer den ersten Schritt.",
            changed=[],
        )
        return self.last_step

    def next_step(self) -> Step:
        if self.solver is None:
            raise RuntimeError("Kein Solver aktiv. Bitte zuerst start() aufrufen.")
        self.last_step = self.solver.next_step()
        return self.last_step

    def snapshot(self) -> Dict[str, Any]:
        # aktuellen Solver-Snapshot
        if self.solver is None:
            raise RuntimeError("Kein Solver aktiv.")
        return self.solver.snapshot()

    def get_solution(self) -> List[float]:
        if self.solver is None:
            raise RuntimeError("Kein Solver aktiv.")
        return self.solver.get_solution()

    def is_running(self) -> bool:
        # läuft ein Solver?
        return self.solver is not None and not self.solver.done
    
def main():
    from gui.lr_gui import LRGUI
    app = LRGUI()
    app.mainloop()


if __name__ == "__main__":
    main()