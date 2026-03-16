from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GaussSeidelRowDetail:
    # Index der aktuell berechneten Zeile
    row_index: int

    # Diagonalelement a_ii
    diag: float

    # Rechte Seite b_i
    rhs: float

    # Kompletter Vektor vor Beginn der Iteration
    x_old: List[float]

    # Zwischenstand von x direkt nach dieser Zeile
    x_new_partial: List[float]

    # Einzelterme der Summation:
    # (j, a_ij, verwendetes_x_j, produkt, uses_new)
    # uses_new = True -> bereits aktualisierter Wert aus derselben Iteration
    # uses_new = False -> alter Wert aus vorheriger Iteration
    terms: List[Tuple[int, float, float, float, bool]]

    # Neu berechneter Wert x_i
    new_value: float



# Ein Schritt
@dataclass
class Step:
    kind: str

    # Nummer der aktuellen Iteration
    iteration: int

    # Vektor vor der Iteration
    x_old: List[float]

    # Vektor nach der Iteration
    x_new: List[float]

    # Maximale Änderung
    max_diff: float

    # Meldung für Ausgabe
    message: str

    # Detaildaten für jede bearbeitete Zeile
    row_details: List[GaussSeidelRowDetail]


class GaussSeidelSolver:
    """
    Logik:
    - Matrix A und rechte Seite b
    - aktuellen Iterationsvektor x
    - Abbruch ueber Toleranz oder Sicherheitslimit
    - Detailinformationen je Zeile für die Anzeige in der GUI
    """

    def __init__(self) -> None:
        # Standardparameter
        self.tol = 1e-6
        self.safety_limit = 100
        self.eps = 1e-12

        # Problemgroesse und Daten
        self.n = 0
        self.A: List[List[float]] = []
        self.b: List[float] = []
        self.x: List[float] = []

        # Solver-Zustand
        self.iteration = 0
        self.done = True
        self.started = False

    def start(
        self,
        A: List[List[float]],
        b: List[float],
        x0: List[float],
        tol: float = 1e-6,
        safety_limit: int = 100,
        eps: float = 1e-12,
    ) -> None:
        #Initialisiert Solver mit LGS und Startvektor
        self._validate_input(A, b, x0, tol, safety_limit, eps)

        self.A = [row[:] for row in A]
        self.b = b[:]
        self.x = x0[:]
        self.tol = tol
        self.safety_limit = safety_limit
        self.eps = eps
        self.n = len(A)

        self.iteration = 0
        self.done = False
        self.started = True

    @staticmethod
    def _validate_input(A: List[List[float]], b: List[float], x0: List[float], tol: float,
        safety_limit: int, eps: float,) -> None:
        if not A or not b or not x0:
            raise ValueError("A, b und x0 dürfen nicht leer sein.")

        n = len(A)

        if any(len(row) != n for row in A):
            raise ValueError("A muss quadratisch sein.")
        
        if len(b) != n:
            raise ValueError("b muss die passende Länge haben.")

        if len(x0) != n:
            raise ValueError("x0 muss die passende Länge haben.")

        if tol < 0:
            raise ValueError("tol darf nicht negativ sein.")

        if safety_limit < 0:
            raise ValueError("safety_limit darf nicht negativ sein.")

        if eps <= 0:
            raise ValueError("eps muss größer als 0 sein.")

        for i in range(n):
            if abs(A[i][i]) < eps:
                raise ValueError(
                    f"Gauss-Seidel nicht möglich: Diagonalelement a[{i+1},{i+1}] ist 0."
                )

    def _fmt(self, x: float) -> str:
        #Formatiert Zahlen kompakt für Meldungen
        if abs(x) < 1e-12:
            x = 0.0
        return "{:.6g}".format(x)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "A": [row[:] for row in self.A],
            "b": self.b[:],
            "x": self.x[:],
            "iteration": self.iteration,
            "tol": self.tol,
            "safety_limit": self.safety_limit,
            "done": self.done,
            "started": self.started,
        }

    def step(self) -> Step:
        # ein Schritt
        if not self.started or self.done:
            return Step(
                kind="done",
                iteration=self.iteration,
                x_old=self.x[:],
                x_new=self.x[:],
                max_diff=0.0,
                message="Fertig.",
                row_details=[],
            )

        # Sicherheitslimit vor der nächsten Iteration prüfen
        if self.iteration >= self.safety_limit:
            self.done = True
            return Step(
                kind="done",
                iteration=self.iteration,
                x_old=self.x[:],
                x_new=self.x[:],
                max_diff=0.0,
                message=f"Abbruch: Sicherheitslimit von {self.safety_limit} Iterationen erreicht.",
                row_details=[],
            )

        x_old = self.x[:]
        x_new = self.x[:]
        row_details: List[GaussSeidelRowDetail] = []

        # Jede Zeile nacheinander mit bereits aktualisierten Werten berechnen
        for i in range(self.n):
            diag = self.A[i][i]
            rhs = self.b[i]
            s = 0.0
            terms: List[Tuple[int, float, float, float, bool]] = []

            for j in range(self.n):
                if j == i:
                    continue

                if j < i:
                    # Links der Diagonale wurden die Werte schon aktualisiert
                    used_x = x_new[j]
                    uses_new = True
                else:
                    # Rechts der Diagonale stehen noch die alten Werte
                    used_x = x_old[j]
                    uses_new = False

                prod = self.A[i][j] * used_x
                s += prod
                terms.append((j, self.A[i][j], used_x, prod, uses_new))

            x_new[i] = (rhs - s) / diag

            row_details.append(
                GaussSeidelRowDetail(
                    row_index=i,
                    diag=diag,
                    rhs=rhs,
                    x_old=x_old[:],
                    x_new_partial=x_new[:],
                    terms=terms,
                    new_value=x_new[i],
                )
            )

        max_diff = max(abs(x_new[i] - x_old[i]) for i in range(self.n))
        self.iteration += 1
        self.x = x_new[:]

        msg = f"Iteration {self.iteration}"

        # Abbruch über Toleranz
        if max_diff < self.tol:
            self.done = True
            return Step(
                kind="done",
                iteration=self.iteration,
                x_old=x_old,
                x_new=x_new,
                max_diff=max_diff,
                message=msg + f"\nAbbruch: Toleranz {self._fmt(self.tol)} erreicht.",
                row_details=row_details,
            )

        return Step(
            kind="iter",
            iteration=self.iteration,
            x_old=x_old,
            x_new=x_new,
            max_diff=max_diff,
            message=msg,
            row_details=row_details,
        )
