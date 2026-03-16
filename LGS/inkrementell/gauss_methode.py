from __future__ import annotations

from typing import List, Optional, Tuple

from gauss_solver import GaussEliminationSolver, Step


class GaussEliminationGuiLogic:
    """
    Logik zwischen GUI und Gauß-Solver
    - Start eines neuen Solvers
    - Weiter-Schalten um genau einen Schritt
    - Bereitstellung von Matrix, Lösung und Statusinformationen
    """

    def __init__(self) -> None:
        # leere GUI-Logik ohne aktiven Solver
        self.solver: Optional[GaussEliminationSolver] = None
        self.last_step: Optional[Step] = None

    def start(self, A: List[List[float]], b: List[float], pivot_mode: str = "col",
              custom_pivot: Optional[Tuple[int, int]] = None, tol: float = 1e-12, 
              change_tol: float = 1e-10,) -> Step:
        """
        Startet eine neue inkrementelle Gauß-Elimination
        Direkt nach dem Start wird noch kein Rechenschritt ausgeführt
        Stattdessen wird ein neutraler Start-Step zurückgegeben
        """
        self.solver = GaussEliminationSolver(
            A=A,
            b=b,
            pivot_mode=pivot_mode,
            custom_pivot=custom_pivot,
            tol=tol,
            change_tol=change_tol,
        )

        self.last_step = Step(
            kind="init",
            pivot=(0, 0),
            message="Solver initialisiert. Bereit für den ersten Schritt",
            changed_cells=[],
        )
        return self.last_step

    def next_step(self) -> Step:
        # nächsten Schritt durchführen
        if self.solver is None:
            raise RuntimeError("Kein Solver aktiv. Bitte zuerst start() aufrufen")

        self.last_step = self.solver.next_step()
        return self.last_step

    def is_running(self) -> bool:
        # läuft ein Solver?
        return self.solver is not None and not self.solver.done

    def get_matrix(self) -> List[List[float]]:
        if self.solver is None:
            raise RuntimeError("Kein Solver aktiv")
        return self.solver.get_augmented_matrix()

    def get_solution_original_order(self) -> List[float]:
        # sortiert Lösungen wie Orignal
        if self.solver is None:
            raise RuntimeError("Kein Solver aktiv")
        return self.solver.get_solution_original_order()

    def get_permutation(self) -> List[int]:
        # akutelle Permutation
        if self.solver is None:
            raise RuntimeError("Kein Solver aktiv")
        return self.solver.perm[:]
    
def main():
    from gui.gauss_gui import GaussGUI
    app = GaussGUI()
    app.mainloop()


if __name__ == "__main__":
    main()