from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any


@dataclass
class Step:
    # Art des aktuellen Schritts:
    # "chol_update" -> ein Element in L wurde berechnet
    # "forward_sub" -> ein Element in y wurde berechnet
    # "back_sub"    -> ein Element in x wurde berechnet
    # "done"        -> fertig oder Abbruch
    kind: str

    # aktuelle Position im Algorithmus
    pivot: Tuple[int, int]

    # Erklärungstext für GUI / Ausgabe
    message: str = ""

    # Liste der geänderten Zellen
    changed: Optional[List[Tuple[str, int, int]]] = None


class CholeskySolver:
    def __init__(self):
        self.tol = 1e-12
        self.change_tol = 1e-10

        self.n = 0
        self.A = []
        self.L = []
        self.b = []
        self.y = []
        self.x = []

        self.phase = "done"
        self.i = 0
        self.j = 0
        self.fw_i = 0
        self.bw_i = -1
        self.done = True

    def start(self, A: List[List[float]], b: List[float], tol: float = 1e-12, change_tol: float = 1e-10):
        # Matrix darf nicht leer sein
        if not A or not b:
            raise ValueError("A und b dürfen nicht leer sein.")

        n = len(A)

        # A muss quadratisch sein
        if any(len(row) != n for row in A):
            raise ValueError("A muss quadratisch sein.")

        # Länge von b muss zu A passen
        if len(b) != n:
            raise ValueError("b muss die passende Länge haben.")

        self.tol = tol
        self.change_tol = change_tol
        self.n = n

        self.A = [row[:] for row in A]
        self.L = [[0.0] * n for _ in range(n)]
        self.b = b[:]
        self.y = [0.0] * n
        self.x = [0.0] * n

        self.phase = "decomp"
        self.i = 0
        self.j = 0
        self.fw_i = 0
        self.bw_i = n - 1
        self.done = False

    def _fmt(self, x: float) -> str:
        if abs(x) < 1e-12:
            x = 0.0
        return "{:.6g}".format(x)

    def _Lt(self) -> List[List[float]]:
        n = self.n
        Lt = [[0.0] * n for _ in range(n)]
        for r in range(n):
            for c in range(n):
                Lt[r][c] = self.L[c][r]
        return Lt

    def snapshot(self) -> Dict[str, Any]:
        return {
            "A": [row[:] for row in self.A],
            "L": [row[:] for row in self.L],
            "Lt": self._Lt(),
            "b": self.b[:],
            "y": self.y[:],
            "x": self.x[:],
            "phase": self.phase,
        }

    def step(self) -> Step:
        if self.done:
            return Step(kind="done", pivot=(max(self.n - 1, 0), max(self.n - 1, 0)), message="Fertig.")

        while True:
            if self.phase == "decomp":
                if self.i >= self.n:
                    self.phase = "forward"
                    self.fw_i = 0
                    continue

                i, j = self.i, self.j

                s = 0.0
                for k in range(j):
                    s += self.L[i][k] * self.L[j][k]

                changed: List[Tuple[str, int, int]] = []

                if i == j:
                    v = self.A[i][i] - s
                    if v <= self.tol:
                        self.done = True
                        return Step(
                            kind="done",
                            pivot=(i, j),
                            message=(
                                "Abbruch: Matrix ist nicht positiv definit (oder numerisch instabil).\n"
                                f"A[{i+1},{i+1}] - Summe = {self._fmt(v)} <= 0"
                            ),
                        )

                    before = self.L[i][j]
                    self.L[i][j] = math.sqrt(v)
                    if abs(self.L[i][j] - before) > self.change_tol:
                        changed.append(("L", i, j))
                        changed.append(("Lt", j, i))

                    msg = (
                        "Cholesky-Zerlegung: A = L * L^T\n"
                        f"Diagonal-Element L[{i+1},{i+1}]:\n"
                        f"L[{i+1},{i+1}] = sqrt(A[{i+1},{i+1}] - Summe(L[{i+1},k]^2))\n"
                        f"Summe = {self._fmt(s)}\n"
                        f"L[{i+1},{i+1}] = sqrt({self._fmt(self.A[i][i])} - {self._fmt(s)}) = {self._fmt(self.L[i][i])}"
                    )
                else:
                    diag = self.L[j][j]
                    if abs(diag) < self.tol:
                        self.done = True
                        return Step(
                            kind="done",
                            pivot=(i, j),
                            message=(
                                "Abbruch: L[j,j] ist 0 (numerisch instabil / nicht positiv definit).\n"
                                f"L[{j+1},{j+1}] = {self._fmt(diag)}"
                            ),
                        )

                    before = self.L[i][j]
                    self.L[i][j] = (self.A[i][j] - s) / diag
                    if abs(self.L[i][j] - before) > self.change_tol:
                        changed.append(("L", i, j))
                        changed.append(("Lt", j, i))

                    msg = (
                        "Cholesky-Zerlegung: A = L * L^T\n"
                        f"Neben-Diagonal-Element L[{i+1},{j+1}]:\n"
                        f"L[{i+1},{j+1}] = (A[{i+1},{j+1}] - Summe(L[{i+1},k]*L[{j+1},k])) / L[{j+1},{j+1}]\n"
                        f"Summe = {self._fmt(s)}\n"
                        f"L[{i+1},{j+1}] = ({self._fmt(self.A[i][j])} - {self._fmt(s)}) / {self._fmt(diag)} = {self._fmt(self.L[i][j])}"
                    )

                self.j += 1
                if self.j > self.i:
                    self.i += 1
                    self.j = 0

                return Step(kind="chol_update", pivot=(i, j), message=msg, changed=changed)

            if self.phase == "forward":
                if self.fw_i >= self.n:
                    self.phase = "back"
                    self.bw_i = self.n - 1
                    continue

                i = self.fw_i
                s = 0.0
                for j in range(i):
                    s += self.L[i][j] * self.y[j]

                diag = self.L[i][i]
                if abs(diag) < self.tol:
                    self.done = True
                    return Step(kind="done", pivot=(i, i), message="Abbruch: Diagonale in L ist 0.")

                before = self.y[i]
                self.y[i] = (self.b[i] - s) / diag

                changed = []
                if abs(self.y[i] - before) > self.change_tol:
                    changed.append(("y", i, 0))

                msg = (
                    "Vorwärtseinsetzen: L*y = b\n"
                    f"y[{i+1}] = (b[{i+1}] - Summe(L[{i+1},j]*y[j])) / L[{i+1},{i+1}]\n"
                    f"Summe = {self._fmt(s)}\n"
                    f"y[{i+1}] = ({self._fmt(self.b[i])} - {self._fmt(s)}) / {self._fmt(diag)} = {self._fmt(self.y[i])}"
                )

                self.fw_i += 1
                return Step(kind="forward_sub", pivot=(i, i), message=msg, changed=changed)

            if self.phase == "back":
                if self.bw_i < 0:
                    self.done = True
                    self.phase = "done"
                    return Step(kind="done", pivot=(0, 0), message="Fertig: Cholesky + Loesen abgeschlossen.")

                i = self.bw_i
                s = 0.0
                for j in range(i + 1, self.n):
                    s += self.L[j][i] * self.x[j]

                diag = self.L[i][i]
                if abs(diag) < self.tol:
                    self.done = True
                    return Step(kind="done", pivot=(i, i), message="Abbruch: Diagonale in L ist 0.")

                before = self.x[i]
                self.x[i] = (self.y[i] - s) / diag

                changed = []
                if abs(self.x[i] - before) > self.change_tol:
                    changed.append(("x", i, 0))

                msg = (
                    "Rückwärtseinsetzen: L^T*x = y\n"
                    f"x[{i+1}] = (y[{i+1}] - Summe(L[j,{i+1}]*x[j])) / L[{i+1},{i+1}]\n"
                    f"Summe = {self._fmt(s)}\n"
                    f"x[{i+1}] = ({self._fmt(self.y[i])} - {self._fmt(s)}) / {self._fmt(diag)} = {self._fmt(self.x[i])}"
                )

                self.bw_i -= 1
                return Step(kind="back_sub", pivot=(i, i), message=msg, changed=changed)

            self.done = True
            return Step(kind="done", pivot=(0, 0), message="Unbekannter Zustand.")