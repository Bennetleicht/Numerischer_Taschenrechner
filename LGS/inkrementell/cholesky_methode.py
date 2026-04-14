from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cholesky_solver import CholeskySolver


class CholeskyMethod:
    title = "Cholesky-Zerlegung"

    def __init__(self):
        self.solver = CholeskySolver()

    def start(self, A, b, tol=1e-12, change_tol=1e-10):
        # Solver initialisieren
        self.solver.start(A, b, tol=tol, change_tol=change_tol)

        # Snapshot für GUI direkt nach Start
        return self.solver.snapshot()

    def next_step(self):
        # Einen Rechenschritt ausführen
        step = self.solver.step()

        # Zusätzlich aktuellen Zustand zurückgeben
        snapshot = self.solver.snapshot()

        return step, snapshot
    
    
    
def main():
    from gui.cholesky_gui import CholeskyGUI
    app = CholeskyGUI()
    app.mainloop()


if __name__ == "__main__":
    main()