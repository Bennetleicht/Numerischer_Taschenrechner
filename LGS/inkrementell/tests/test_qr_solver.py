from typing import Any, Dict, List, Optional, Tuple

from inkrementell.qr_solver import QRGivensSolver, Step


class QRGivensMethod:
    title = "QR-Zerlegung mit Givens-Rotation"
    plotter_kind = None

    def __init__(self):
        # Zu Beginn keinen aktiven Solver
        self.solver: Optional[QRGivensSolver] = None
        self.last_step: Optional[Step] = None

    def start(
        self,
        A: List[List[float]],
        b: List[float],
        eps: float = 1e-12,
        change_tol: float = 1e-10,
    ) -> Step:
        # Startet neuen Solver mit A und b
        self.solver = QRGivensSolver(A=A, b=b, tol=eps, change_tol=change_tol)

        # Erzeugt initialen Start-Schritt
        self.last_step = Step(
            kind="init",
            pivot=(0, 0),
            message="Solver initialisiert. Bereit fuer den ersten Schritt.",
            changed=[],
        )
        return self.last_step

    def next_step(self, target: Optional[Tuple[int, int]] = None) -> Step:
        # ein Schritt
        if self.solver is None:
            raise RuntimeError("Kein Solver aktiv. Bitte zuerst start() aufrufen.")

        self.last_step = self.solver.next_step(target=target)
        return self.last_step

    def switch_to_backsub(self) -> Optional[str]:
        # Wechselt in Rückwärtseinsetzungsphase
        if self.solver is None:
            raise RuntimeError("Kein Solver aktiv. Bitte zuerst start() aufrufen.")
        return self.solver.switch_to_backsub()

    def snapshot(self) -> Dict[str, Any]:
        # aktuellen Zustand Solvers
        if self.solver is None:
            raise RuntimeError("Kein Solver aktiv.")
        return self.solver.snapshot()

    def get_solution(self) -> List[float]:
        # Lösung
        if self.solver is None:
            raise RuntimeError("Kein Solver aktiv.")
        return self.solver.get_solution()

    def is_running(self) -> bool:
        # aktueller Solver aktiv
        return self.solver is not None and not self.solver.done