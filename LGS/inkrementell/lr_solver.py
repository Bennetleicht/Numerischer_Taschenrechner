from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Step:
    """
    kind:
        - "lr_update"   :ein Eliminations-/Zerlegungsschritt
        - "forward_sub" :Vorwärtseinsetzen in L*y = b
        - "back_sub"    :Rückwärtseinsetzen in R*x = y
        - "done"        :Verfahren beendet oder wegen Fehler abgebrochen
    """
    kind: str
    pivot: Tuple[int, int]
    target_row: Optional[int] = None
    message: str = ""
    changed: Optional[List[Tuple[str, int, int]]] = None


class LRSolver:
    """
    löst LR, Logik für den mathematischen Ablauf:
    - LR-Zerlegung ohne Pivoting
    - Vorwärtseinsetzen
    - Rückwürtseinsetzen
    """

    def __init__(self, A: List[List[float]], b: List[float], tol: float = 1e-12, 
                change_tol: float = 1e-10,) -> None:
        # Initialisiert den Solver mit A und b
        self._validate_input(A, b)

        self.tol = tol
        self.change_tol = change_tol
        self.n = len(A)

        self.R = [row[:] for row in A]
        self.L = [[0.0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            self.L[i][i] = 1.0

        self.b = b[:]
        self.y = [0.0] * self.n
        self.x = [0.0] * self.n

        self.phase = "decomp"   # "decomp", "forward", "back", "done"
        self.k = 0
        self.i = 1

        self.fw_i = 0
        self.bw_i = self.n - 1

        self.done = False

    @staticmethod
    def _validate_input(A: List[List[float]], b: List[float]) -> None:
        if not A:
            raise ValueError("A darf nicht leer sein.")

        n = len(A)

        if len(b) != n:
            raise ValueError("A und b müssen dieselbe Länge haben.")

        if any(len(row) != n for row in A):
            raise ValueError("A muss quadratisch sein.")

    def _fmt(self, x: float) -> str:
        # Formatiert Zahlen für Ausgabetexte
        if abs(x) < 1e-12:
            x = 0.0
        return "{:.6g}".format(x)

    def next_step(self) -> Step:
        # ein Schritt
        if self.done:
            return Step(
                kind="done",
                pivot=(max(self.n - 1, 0), max(self.n - 1, 0)),
                message="Fertig."
            )

        while True:
            if self.phase == "decomp":
                if self.k >= self.n - 1:
                    self.phase = "forward"
                    self.fw_i = 0
                    continue

                if self.i >= self.n:
                    self.k += 1
                    self.i = self.k + 1
                    continue

                pivot = self.R[self.k][self.k]
                if abs(pivot) < self.tol:
                    self.done = True
                    return Step(
                        kind="done",
                        pivot=(self.k, self.k),
                        message="Abbruch: Pivot in R ist 0 - ohne Pivoting nicht moeglich."
                    )

                i = self.i
                m = self.R[i][self.k] / pivot
                self.L[i][self.k] = m

                changed: List[Tuple[str, int, int]] = [("L", i, self.k)]
                for c in range(self.k, self.n):
                    before = self.R[i][c]
                    self.R[i][c] = self.R[i][c] - m * self.R[self.k][c]
                    if abs(self.R[i][c] - before) > self.change_tol:
                        changed.append(("R", i, c))

                msg = (
                    "LR-ZERLEGUNG\n"
                    f"Pivot: R[{self.k+1},{self.k+1}] = {self._fmt(pivot)}\n"
                    f"m = R[{i+1},{self.k+1}] / R[{self.k+1},{self.k+1}] = {self._fmt(m)}\n"
                    f"Setze L[{i+1},{self.k+1}] = m\n"
                    f"Update: R[{i+1},*] = R[{i+1},*] - m * R[{self.k+1},*]"
                )

                self.i += 1
                return Step(
                    kind="lr_update",
                    pivot=(self.k, self.k),
                    target_row=i,
                    message=msg,
                    changed=changed,
                )

            if self.phase == "forward":
                if self.fw_i >= self.n:
                    self.phase = "back"
                    self.bw_i = self.n - 1
                    continue

                i = self.fw_i
                s = 0.0
                for j in range(i):
                    s += self.L[i][j] * self.y[j]

                self.y[i] = self.b[i] - s

                msg = (
                    "VORWÄRTSEINSETZEN: L*y = b\n"
                    f"y[{i+1}] = b[{i+1}] - Summe(L[{i+1},j] * y[j])\n"
                    f"y[{i+1}] = {self._fmt(self.b[i])} - {self._fmt(s)} = {self._fmt(self.y[i])}"
                )

                self.fw_i += 1
                return Step(
                    kind="forward_sub",
                    pivot=(i, i),
                    message=msg,
                    changed=[("y", i, 0)],
                )

            if self.phase == "back":
                if self.bw_i < 0:
                    self.done = True
                    self.phase = "done"
                    return Step(
                        kind="done",
                        pivot=(0, 0),
                        message="Fertig: LR + Loesen abgeschlossen."
                    )

                i = self.bw_i
                diag = self.R[i][i]
                if abs(diag) < self.tol:
                    self.done = True
                    return Step(
                        kind="done",
                        pivot=(i, i),
                        message="Abbruch: Diagonale in R ist 0 - keine eindeutige Loesung."
                    )

                s = 0.0
                for j in range(i + 1, self.n):
                    s += self.R[i][j] * self.x[j]

                self.x[i] = (self.y[i] - s) / diag

                msg = (
                    "RÜCKWÄRTSEINSETZEN: R*x = y\n"
                    f"x[{i+1}] = (y[{i+1}] - Summe(R[{i+1},j] * x[j])) / R[{i+1},{i+1}]\n"
                    f"x[{i+1}] = ({self._fmt(self.y[i])} - {self._fmt(s)}) / {self._fmt(diag)} = {self._fmt(self.x[i])}"
                )

                self.bw_i -= 1
                return Step(
                    kind="back_sub",
                    pivot=(i, i),
                    message=msg,
                    changed=[("x", i, 0)],
                )

            self.done = True
            return Step(kind="done", pivot=(0, 0), message="Unbekannter Zustand.")

    def snapshot(self) -> Dict[str, Any]:
        # Kopie des Solvers
        return {
            "L": [row[:] for row in self.L],
            "R": [row[:] for row in self.R],
            "b": self.b[:],
            "y": self.y[:],
            "x": self.x[:],
            "phase": self.phase,
        }

    def get_solution(self) -> List[float]:
        # gibt aktuelle Lösung
        return self.x[:]