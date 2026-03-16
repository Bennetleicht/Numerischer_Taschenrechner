import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any


@dataclass
class Step:
    # Beschreibt einen einzelnen Rechenschritt
    kind: str  # "givens", "back_sub", "done"
    pivot: Tuple[int, int]
    message: str = ""
    changed: Optional[List[Tuple[str, int, int]]] = None  # ("R",r,c) ("G",r,c) ("y",i,0) ("x",i,0)


class QRGivensSolver:
    # Reine Solver-Klasse für QR-Zerlegung mit Givens-Rotation + Rückwärtseinsetzen

    def __init__(self, A: List[List[float]], b: List[float], tol: float = 1e-12, change_tol: float = 1e-10):
        # Speichert numerische Toleranzen
        self.tol = tol
        self.change_tol = change_tol

        self.m = len(A)
        self.n = len(A[0]) if self.m else 0

        if self.m == 0 or self.n == 0:
            raise ValueError("Matrix A darf nicht leer sein.")

        if len(b) != self.m:
            raise ValueError("Vektor b muss so viele Eintraege haben wie A Zeilen (m).")

        for r in range(self.m):
            if len(A[r]) != self.n:
                raise ValueError("Alle Zeilen in A muessen die gleiche Spaltenanzahl haben (n).")

        # Kopiert A als Arbeitsmatrix R
        self.R = [row[:] for row in A]

        # Kopiert b als Arbeitsvektor y = Q^T b
        self.y = b[:]

        # Initialisiert den Lösungsvektor x
        self.x = [0.0] * self.n

        # Speichert aktuelle Givens-Rotationsmatrix
        self.G = [[0.0] * self.m for _ in range(self.m)]
        self._set_G_identity()

        # Startet in der Rotationsphase
        self.phase = "rot"   # "rot", "back", "done"
        self.done = False

        # Index für das Rückwärtseinsetzen
        self.bw_i = self.n - 1

    def _set_G_identity(self):
        # Setzt G auf die Einheitsmatrix zurück
        for r in range(self.m):
            for c in range(self.m):
                self.G[r][c] = 1.0 if r == c else 0.0

    def _fmt(self, x: float) -> str:
        # Formatiert Zahlen kompakt für Ausgabetexte
        if abs(x) < 1e-12:
            x = 0.0
        return "{:.6g}".format(x)

    def snapshot(self) -> Dict[str, Any]:
        # Gibt den aktuellen Zustand als Kopie zurück
        return {
            "R": [row[:] for row in self.R],
            "G": [row[:] for row in self.G],
            "y": self.y[:],
            "x": self.x[:],
            "phase": self.phase,
            "m": self.m,
            "n": self.n,
        }

    def _build_givens(self, a: float, b: float) -> Tuple[float, float, float]:
        # Berechnet c, s und r für die Givens-Rotation
        r = math.hypot(a, b)

        # Falls a und b praktisch 0 sind, wird keine echte Rotation benötigt
        if r < self.tol:
            return 1.0, 0.0, 0.0

        c = a / r
        s = b / r
        return c, s, r

    def _apply_givens(self, i: int, k: int) -> Step:
        # Führt eine Givens-Rotation aus, um R[i][k] zu eliminieren
        a = self.R[k][k]
        b = self.R[i][k]

        # Wenn das Zielelement bereits 0 ist, passiert rechnerisch nichts
        if abs(b) < self.tol:
            self._set_G_identity()
            msg = (
                "QR-Zerlegung mit Givens-Rotation\n"
                f"Ziel: R[{i+1},{k+1}] -> 0\n\n"
                "R ist dort bereits ~0 -> kein Update."
            )
            return Step(kind="givens", pivot=(i, k), message=msg, changed=[])

        # Berechnet die Rotationsparameter
        c, s, rnorm = self._build_givens(a, b)

        # Baut die aktuelle Givens-Matrix auf
        self._set_G_identity()
        self.G[k][k] = c
        self.G[k][i] = s
        self.G[i][k] = -s
        self.G[i][i] = c

        changed: List[Tuple[str, int, int]] = [
            ("G", k, k), ("G", k, i),
            ("G", i, k), ("G", i, i),
        ]

        # Wendet die Rotation auf R an
        for col in range(k, self.n):
            before_k = self.R[k][col]
            before_i = self.R[i][col]

            t_k = c * before_k + s * before_i
            t_i = -s * before_k + c * before_i

            self.R[k][col] = t_k
            self.R[i][col] = t_i

            if abs(t_k - before_k) > self.change_tol:
                changed.append(("R", k, col))
            if abs(t_i - before_i) > self.change_tol:
                changed.append(("R", i, col))

        # Wendet die Rotation auch auf y = Q^T b an
        by_k = self.y[k]
        by_i = self.y[i]

        ny_k = c * by_k + s * by_i
        ny_i = -s * by_k + c * by_i

        self.y[k] = ny_k
        self.y[i] = ny_i

        if abs(ny_k - by_k) > self.change_tol:
            changed.append(("y", k, 0))
        if abs(ny_i - by_i) > self.change_tol:
            changed.append(("y", i, 0))

        msg = (
            "QR-Zerlegung mit Givens-Rotation\n"
            f"Ziel: R[{i+1},{k+1}] -> 0\n\n"
            f"a = R[{k+1},{k+1}] = {self._fmt(a)}\n"
            f"b = R[{i+1},{k+1}] = {self._fmt(b)}\n\n"
            f"r = sqrt(a^2 + b^2) = {self._fmt(rnorm)}\n"
            f"c = a/r = {self._fmt(c)}\n"
            f"s = b/r = {self._fmt(s)}"
        )

        return Step(kind="givens", pivot=(i, k), message=msg, changed=changed)

    def next_step(self, target: Optional[Tuple[int, int]] = None) -> Step:
        # ein Schritt
        if self.done:
            return Step(kind="done", pivot=(0, 0), message="Fertig.")

        # Rotationsphase
        if self.phase == "rot":
            # Ohne Zielauswahl kann keine Elimination durchgeführt werden
            if target is None:
                return Step(kind="done", pivot=(0, 0), message="Abbruch: Kein Element ausgewaehlt.")

            i, k = target

            if not (0 <= i < self.m and 0 <= k < self.n):
                return Step(kind="done", pivot=(0, 0), message="Abbruch: Auswahl ausserhalb des Bereichs.")

            # Es dürfen nur Elemente unter der Diagonale eliminiert werden
            if i <= k:
                return Step(kind="done", pivot=(i, k), message="Ungueltig: Es muss i > k gelten (unter der Diagonale).")

            return self._apply_givens(i, k)

        # Rückwärtseinsetzen
        if self.phase == "back":
            # Wenn alle x_i berechnet sind, ist das Verfahren fertig
            if self.bw_i < 0:
                self.done = True
                self.phase = "done"
                self._set_G_identity()
                return Step(kind="done", pivot=(0, 0), message="Fertig: Loesen abgeschlossen.")

            i = self.bw_i
            diag = self.R[i][i]

            # Prüft auf Null auf der Diagonale
            if abs(diag) < self.tol:
                self.done = True
                return Step(kind="done", pivot=(i, i), message="Abbruch: Diagonale in R ist 0 (keine eindeutige Loesung).")

            ssum = 0.0
            for j in range(i + 1, self.n):
                ssum += self.R[i][j] * self.x[j]

            before = self.x[i]
            self.x[i] = (self.y[i] - ssum) / diag

            changed = []
            if abs(self.x[i] - before) > self.change_tol:
                changed.append(("x", i, 0))

            msg = (
                "Rueckwaertseinsetzen (R*x = y)\n"
                f"x[{i+1}] = (y[{i+1}] - Summe(R[{i+1},j]*x[j])) / R[{i+1},{i+1}]\n"
                f"Summe = {self._fmt(ssum)}\n"
                f"x[{i+1}] = ({self._fmt(self.y[i])} - {self._fmt(ssum)}) / {self._fmt(diag)} = {self._fmt(self.x[i])}"
            )

            self.bw_i -= 1
            self._set_G_identity()
            return Step(kind="back_sub", pivot=(i, i), message=msg, changed=changed)

        # Fallback bei undefiniertem Zustand
        self.done = True
        return Step(kind="done", pivot=(0, 0), message="Unbekannter Zustand.")

    def switch_to_backsub(self) -> Optional[str]:
        # Wechselt zur Rückwärtseinsetzungsphase, wenn R oben-trapezförmig ist

        # Unterbestimmter Fall: keine eindeutige Lösung über normales Rückwärtseinsetzen
        if self.m < self.n:
            return "m < n (unterbestimmt). Rueckwaertseinsetzen liefert keine eindeutige Loesung."

        # Prüft ob unter der Diagonale noch relevante Einträge vorhanden sind
        for r in range(1, self.m):
            for c in range(0, min(r, self.n)):
                if abs(self.R[r][c]) > 1e-9:
                    return "R ist noch nicht obere (Trapez-)Matrix. Eliminiere erst weiter unter der Diagonale."

        self.phase = "back"
        self.bw_i = self.n - 1
        self._set_G_identity()
        return None

    def get_solution(self) -> List[float]:
        # Gibt die aktuell berechnete Lösung zurück
        return self.x[:]