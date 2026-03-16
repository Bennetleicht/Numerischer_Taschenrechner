from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Step:
    """
    kind:
        - "swap"    :Pivotierung durch Zeilen-/Spaltentausch
        - "elim"    :Eliminationsschritt in der Vorwärtsphase
        - "backsub" :Rückwärtseinsetzen
        - "done"    :Verfahren beendet oder Abbruch
    """
    kind: str
    pivot: Tuple[int, int]
    target_row: Optional[int] = None
    factor: Optional[float] = None
    message: str = ""
    changed_cells: Optional[List[Tuple[int, int]]] = None


class GaussEliminationSolver:
    """
    Logik:
    - die augmentierte Matrix [A|b]
    - den aktuellen Rechenschritt
    - Pivotierung
    - Rückwärtseinsetzen
    - die Permutation bei Spaltentausch
    """

    def __init__(self, A: List[List[float]], b: List[float], pivot_mode: str = "col",   # "col", "row", "total", "custom"
        custom_pivot: Optional[Tuple[int, int]] = None, tol: float = 1e-12, change_tol: float = 1e-10,) -> None:
        """
        Initialisiert den Solver mit linearem Gleichungssystem.
        Args:
            A: Koeffizientenmatrix
            b: rechte Seite
            pivot_mode: Art der Pivotierung
            custom_pivot: nur bei pivot_mode="custom", gewünschter Pivot (r, c)
            tol: Schwelle für numerisch '0'
            change_tol: Schwelle, ab wann eine Zellenänderung als geändert gilt
        """
        self._validate_input(A, b, pivot_mode, custom_pivot)

        self.tol = tol
        self.change_tol = change_tol
        self.n = len(A)

        # Augmentierte Matrix [A|b]
        self.M: List[List[float]] = [row[:] + [b_i] for row, b_i in zip(A, b)]

        # Steuerung der Vorwärtsphase
        self.k = 0
        self.i = 1
        self.phase = "forward"
        self.done = False

        # Ergebnisvektor in aktueller Spaltenreihenfolge
        self.x = [0.0] * self.n
        self.back_i = self.n - 1

        # perm[cur_col] = ursprünglicher Variablenindex
        self.perm = list(range(self.n))

        self.pivot_mode = pivot_mode
        self.custom_pivot = custom_pivot
        self._pivot_applied_for_k = False
        self._custom_applied = False

    @staticmethod
    def _validate_input(
        A: List[List[float]],
        b: List[float],
        pivot_mode: str,
        custom_pivot: Optional[Tuple[int, int]],
    ) -> None:
        # Prüft Eingaben auf Fehler
        if not A:
            raise ValueError("A darf nicht leer sein.")

        n = len(A)

        if len(b) != n:
            raise ValueError("A und b müssen dieselbe Länge haben.")

        if any(len(row) != n for row in A):
            raise ValueError("A muss eine quadratische Matrix sein.")

        if pivot_mode not in {"col", "row", "total", "custom"}:
            raise ValueError("pivot_mode muss 'col', 'row', 'total' oder 'custom' sein.")

        if pivot_mode == "custom" and custom_pivot is not None:
            r, c = custom_pivot
            if not (0 <= r < n and 0 <= c < n):
                raise ValueError("custom_pivot liegt außerhalb der Matrix.")

    def _fmt(self, x: float) -> str:
        # Formatiert Zahlen für Ausgabetexte
        if abs(x) < 1e-12:
            x = 0.0
        return "{:.6g}".format(x)

    def _swap_rows(self, r1: int, r2: int) -> None:
        # Tauscht zwei Zeilen der augmentierten Matrix
        self.M[r1], self.M[r2] = self.M[r2], self.M[r1]

    def _swap_cols(self, c1: int, c2: int) -> None:
        """
        Tauscht zwei Spalen der Koeffizientenmatrix innerhalb [A|b]
        rechte Seite b unberührt
        Variablen-Permutation aktualisiert
        """
        for r in range(self.n):
            self.M[r][c1], self.M[r][c2] = self.M[r][c2], self.M[r][c1]
        self.perm[c1], self.perm[c2] = self.perm[c2], self.perm[c1]

    def _pivot_row_for_col(self, k: int) -> int:
        # Findet bei Spaltenpivotisierung die beste Zeile in Spalte k
        best = k
        best_val = abs(self.M[k][k])
        for r in range(k + 1, self.n):
            v = abs(self.M[r][k])
            if v > best_val:
                best_val = v
                best = r
        return best

    def _pivot_col_for_row(self, k: int) -> int:
        # Findet bei Zeilenpivotisierung die beste Spalte in Zeile k
        best = k
        best_val = abs(self.M[k][k])
        for c in range(k + 1, self.n):
            v = abs(self.M[k][c])
            if v > best_val:
                best_val = v
                best = c
        return best

    def _pivot_total(self, k: int) -> Tuple[int, int]:
        # Findet bei Totalpivotisierung das größte Element im Restblock
        best_r = k
        best_c = k
        best_val = abs(self.M[k][k])

        for r in range(k, self.n):
            for c in range(k, self.n):
                v = abs(self.M[r][c])
                if v > best_val:
                    best_val = v
                    best_r, best_c = r, c

        return best_r, best_c

    def _apply_pivot_for_current_k(self) -> Optional[Step]:
        # Führt die Pivotierung für die aktuelle Stufe k aus.
        k = self.k
        n = self.n
        changed: List[Tuple[int, int]] = []
        did_change = False
        msg_lines: List[str] = []

        if self.pivot_mode == "custom" and (not self._custom_applied) and k == 0:
            if self.custom_pivot is None:
                self.done = True
                return Step(
                    kind="done",
                    pivot=(k, k),
                    message="Abbruch: kein Pivot (Eigene Wahl) ausgewaehlt."
                )

            r0, c0 = self.custom_pivot

            if r0 != 0:
                self._swap_rows(0, r0)
                did_change = True
                msg_lines.append(f"Eigene Wahl: Tausche Zeilen 1 <-> {r0+1}")
                for c in range(n + 1):
                    changed.append((0, c))
                    changed.append((r0, c))

            if c0 != 0:
                self._swap_cols(0, c0)
                did_change = True
                msg_lines.append(f"Eigene Wahl: Tausche Spalten 1 <-> {c0+1}")
                for r in range(n):
                    changed.append((r, 0))
                    changed.append((r, c0))

            self._custom_applied = True

        elif self.pivot_mode == "col":
            pr = self._pivot_row_for_col(k)
            if pr != k:
                self._swap_rows(k, pr)
                did_change = True
                msg_lines.append(f"Spaltenpivot: Tausche Zeilen {k+1} <-> {pr+1}")
                for c in range(n + 1):
                    changed.append((k, c))
                    changed.append((pr, c))

        elif self.pivot_mode == "row":
            pc = self._pivot_col_for_row(k)
            if pc != k:
                self._swap_cols(k, pc)
                did_change = True
                msg_lines.append(f"Zeilenpivot: Tausche Spalten {k+1} <-> {pc+1}")
                for r in range(n):
                    changed.append((r, k))
                    changed.append((r, pc))

        elif self.pivot_mode == "total":
            pr, pc = self._pivot_total(k)

            if pr != k:
                self._swap_rows(k, pr)
                did_change = True
                msg_lines.append(f"Totalpivot: Tausche Zeilen {k+1} <-> {pr+1}")
                for c in range(n + 1):
                    changed.append((k, c))
                    changed.append((pr, c))

            if pc != k:
                self._swap_cols(k, pc)
                did_change = True
                msg_lines.append(f"Totalpivot: Tausche Spalten {k+1} <-> {pc+1}")
                for r in range(n):
                    changed.append((r, k))
                    changed.append((r, pc))

        if abs(self.M[k][k]) < self.tol:
            self.done = True
            return Step(
                kind="done",
                pivot=(k, k),
                message="Abbruch: Pivot ist 0 (singulaer/instabil)."
            )

        if did_change:
            return Step(
                kind="swap",
                pivot=(k, k),
                message="TAUSCH (Pivotierung)\n" + "\n".join(msg_lines),
                changed_cells=changed,
            )

        return None

    def next_step(self) -> Step:
        """
        ein Schritt
        Ablauf:
        - Vorwärtselimination inkl. Pivotierung
        - Rückwärtseinsetzen
        - danach `done`
        Returns:
            Ein `Step`-Objekt mit Beschreibung des gerade ausgeführten Schritts.
        """
        if self.done:
            return Step(
                kind="done",
                pivot=(max(self.n - 1, 0), max(self.n - 1, 0)),
                message="Fertig."
            )

        if self.phase == "forward":
            if self.k >= self.n:
                self.phase = "back"
                self.back_i = self.n - 1
                return self.next_step()

            if not self._pivot_applied_for_k:
                self._pivot_applied_for_k = True
                s = self._apply_pivot_for_current_k()
                if s is not None:
                    return s

            if self.i >= self.n:
                self.k += 1
                self.i = self.k + 1
                self._pivot_applied_for_k = False
                return self.next_step()

            pivot_val = self.M[self.k][self.k]
            a_ik = self.M[self.i][self.k]
            m = a_ik / pivot_val

            changed: List[Tuple[int, int]] = []
            for c in range(self.k, self.n + 1):
                before = self.M[self.i][c]
                self.M[self.i][c] = self.M[self.i][c] - m * self.M[self.k][c]
                if abs(self.M[self.i][c] - before) > self.change_tol:
                    changed.append((self.i, c))

            msg = (
                "ELIMINATION\n"
                f"Pivot a[{self.k+1},{self.k+1}] = {self._fmt(pivot_val)}\n"
                f"m = a[{self.i+1},{self.k+1}] / a[{self.k+1},{self.k+1}] = "
                f"{self._fmt(a_ik)} / {self._fmt(pivot_val)} = {self._fmt(m)}\n"
                f"Z{self.i+1} = Z{self.i+1} - m * Z{self.k+1}"
            )

            step = Step(
                kind="elim",
                pivot=(self.k, self.k),
                target_row=self.i,
                factor=m,
                message=msg,
                changed_cells=changed,
            )

            self.i += 1
            return step

        if self.phase == "back":
            if self.back_i < 0:
                self.done = True
                return Step(
                    kind="done",
                    pivot=(0, 0),
                    message="Fertig: Rückwärtseinsetzen abgeschlossen."
                )

            i = self.back_i
            s = 0.0
            for j in range(i + 1, self.n):
                s += self.M[i][j] * self.x[j]

            diag = self.M[i][i]
            if abs(diag) < self.tol:
                self.done = True
                return Step(
                    kind="done",
                    pivot=(i, i),
                    message=f"Abbruch: Diagonale a[{i+1},{i+1}] ist 0."
                )

            rhs = self.M[i][self.n]
            xi = (rhs - s) / diag
            self.x[i] = xi

            msg = (
                "RÜCKWÄRTSEINSETZEN\n"
                f"x{i+1} = ({self._fmt(rhs)} - {self._fmt(s)}) / {self._fmt(diag)} = {self._fmt(xi)}"
            )

            self.back_i -= 1
            return Step(kind="backsub", pivot=(i, i), message=msg)

        self.done = True
        return Step(kind="done", pivot=(0, 0), message="Unbekannter Zustand.")

    def get_augmented_matrix(self) -> List[List[float]]:
        # Kopie augmentierten Matrix
        return [row[:] for row in self.M]

    def get_solution_current_order(self) -> List[float]:
        # Lösung in aktueller Spaltenreihenfolge
        return self.x[:]

    def get_solution_original_order(self) -> List[float]:
        # Lösung in ursprünglicher Variablenreihenfolge
        x_out = [0.0] * self.n
        for cur_col, orig_idx in enumerate(self.perm):
            x_out[orig_idx] = self.x[cur_col]
        return x_out