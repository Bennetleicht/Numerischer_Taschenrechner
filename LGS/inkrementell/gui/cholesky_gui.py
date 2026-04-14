from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Optional, Dict, Tuple, Any

from cholesky_solver import CholeskySolver, Step
from gui.base_lgs_inkrementell_gui import BaseLGSInkrementellGUI


class CholeskyStepper(CholeskySolver):
    pass


class CholeskyGUI(BaseLGSInkrementellGUI):

    title_text = "Cholesky-Zerlegung"

    def _create_stepper(self, A: List[List[float]], b: List[float]):
        if not self._is_symmetric(A):
            messagebox.showerror(
                "Fehler",
                "Cholesky nur für symmetrische Matrizen: A muss symmetrisch sein.")
            return None
        stepper = CholeskyStepper()
        stepper.start(A, b)
        return stepper

    def _is_symmetric(self, A: List[List[float]], tol: float = 1e-9) -> bool:
        n = len(A)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i][j] - A[j][i]) > tol:
                    return False
        return True

    def _is_pivot_cell(self, name, r, c, pivot_r, pivot_c, step) -> bool:
        return (step.kind == "chol_update"
                and name == "L"
                and r == pivot_r and c == pivot_c)

    def _do_one_step(self):
        if not self.stepper:
            return
        step = self.stepper.step()
        snap = self.stepper.snapshot()
        self._append_step_card(snap, step)
        if step.kind == "done":
            self._append_final_solution_card()
            self.started = False
            self.stepper = None

    def _append_step_card(self, snap: Dict[str, Any], step: Step,
                           extra_title: Optional[str] = None):
        self.step_count += 1
        kind_map = {
            "chol_update": "ZERLEGUNG",
            "forward_sub": "VORWÄRTS",
            "back_sub":    "RÜCKWÄRTS",
            "done":        "FERTIG",
        }
        kind_txt = kind_map.get(step.kind, step.kind.upper())
        title = extra_title or f"Schritt {self.step_count}: {kind_txt}"

        changed = set(step.changed or [])
        pivot_r, pivot_c = step.pivot

        card = ttk.Frame(self.history.inner, padding=10, style="Card.TFrame")
        card.pack(fill="x", pady=8, padx=2)
        ttk.Label(card, text=title, style="Header.TLabel",
                  padding=(8, 6)).pack(fill="x")

        body = ttk.Frame(card, style="Card.TFrame")
        body.pack(fill="x", pady=(10, 0))
        body.columnconfigure(0, weight=1); body.columnconfigure(1, weight=1)

        left  = ttk.Frame(body, style="Card.TFrame")
        right = ttk.Frame(body, style="Card.TFrame")
        left.grid(row=0, column=0, sticky="nw", padx=(0, 12))
        right.grid(row=0, column=1, sticky="new")

        top_row = ttk.Frame(left, style="Card.TFrame")
        top_row.pack(anchor="w")
        for fname, key in [("A", "A"), ("L", "L"), ("L^T", "Lt")]:
            f = ttk.Frame(top_row, style="Card.TFrame")
            f.pack(side="left", anchor="n", padx=(0, 18))
            ttk.Label(f, text=fname, style="Sub.TLabel").pack(anchor="w")
            self._draw_matrix(f, key, snap[key], changed,
                              pivot_r, pivot_c, step)

        vec_row = ttk.Frame(left, style="Card.TFrame")
        vec_row.pack(anchor="w", pady=(10, 0))
        for vname, key in [("b", "b"), ("y", "y"), ("x", "x")]:
            f = ttk.Frame(vec_row, style="Card.TFrame")
            f.pack(side="left", anchor="n", padx=(0, 18))
            ttk.Label(f, text=vname, style="Sub.TLabel").pack(anchor="w")
            self._draw_vector(f, key, snap[key], changed)

        msg = (step.message or "").strip()
        txt = tk.Text(right,
                      height=max(6, min(12, msg.count("\n") + 4)),
                      wrap="word", relief="solid", borderwidth=1,
                      font=("Segoe UI", 10))
        txt.pack(fill="x", expand=False)
        txt.insert("1.0", msg)
        txt.configure(state="disabled")

        self.history.canvas.update_idletasks()
        self.history.canvas.yview_moveto(1.0)

    def _append_final_solution_card(self):
        if not self.stepper:
            return
        x_final = self.stepper.x[:]
        parts = [f"x{i+1} = {self._fmt(v)}" for i, v in enumerate(x_final)]
        sol = "   |   ".join(parts)
        dummy = Step(kind="done", pivot=(0, 0), message="LÖSUNG\n" + sol)
        snap = self.stepper.snapshot()
        self._append_step_card(snap, dummy, extra_title="LÖSUNG (Endergebnis)")
        messagebox.showinfo("Lösung", sol)


def main():
    app = CholeskyGUI()
    app.mainloop()


if __name__ == "__main__":
    main()