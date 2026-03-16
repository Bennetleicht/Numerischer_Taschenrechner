from typing import Any, Dict, List, Optional, Tuple

from qr_solver import QRGivensSolver, Step


class QRGivensMethod:
    title = "QR-Zerlegung mit Givens-Rotation"
    plotter_kind = None

    def __init__(self):
        # leeres Gui, ohne aktiven Solver
        self.solver: Optional[QRGivensSolver] = None
        self.last_step: Optional[Step] = None

    def start(self, A: List[List[float]], b: List[float], tol: float = 1e-12, change_tol: float = 1e-10,) -> Step:
        # Startet einen neuen Solver mit A und b
        self.solver = QRGivensSolver(A=A, b=b, tol=tol, change_tol=change_tol)

        # initialen Start-Schritt
        self.last_step = Step(
            kind="init",
            pivot=(0, 0),
            message="Solver initialisiert. Bereit fuer den ersten Schritt.",
            changed=[],
        )
        return self.last_step

    def next_step(self, target: Optional[Tuple[int, int]] = None) -> Step:
        # einen Schritt durhcführen
        if self.solver is None:
            raise RuntimeError("Kein Solver aktiv. Bitte zuerst start() aufrufen.")

        self.last_step = self.solver.next_step(target=target)
        return self.last_step

    def switch_to_backsub(self) -> Optional[str]:
        # Wechselt in die Rückwärtseinsetzungsphase
        if self.solver is None:
            raise RuntimeError("Kein Solver aktiv. Bitte zuerst start() aufrufen.")
        return self.solver.switch_to_backsub()

    def snapshot(self) -> Dict[str, Any]:
        # aktuellen Zustand des Solvers
        if self.solver is None:
            raise RuntimeError("Kein Solver aktiv.")
        return self.solver.snapshot()

    def get_solution(self) -> List[float]:
        # aktuell berechnete Lösung
        if self.solver is None:
            raise RuntimeError("Kein Solver aktiv.")
        return self.solver.get_solution()

    def is_running(self) -> bool:
        # ein aktiver Solver ?
        return self.solver is not None and not self.solver.done
    
def main():
    from gui.qr_gui import QRGivensGUI
    app = QRGivensGUI()
    app.mainloop()


if __name__ == "__main__":
    main()