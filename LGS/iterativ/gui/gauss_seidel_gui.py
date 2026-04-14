from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tkinter as tk
from tkinter import ttk
from typing import List

from gauss_seidel_methode import GaussSeidelStepper
from gauss_seidel_solver import Step as GaussStep
from gui.base_lgs_iterativ_gui import BaseIterativGUI


class GaussSeidelGUI(BaseIterativGUI):

    title_text      = "Gauß-Seidel-Verfahren"
    formula_label   = "Gauß-Seidel-Formel"
    _solution_title = "Gauß-Seidel-Ergebnis"
    formula_latex   = (
        r"$x_i^{(k+1)}"
        r" = \left( b_i"
        r" - \sum_{j=1}^{i-1} a_{ij}x_j^{(k+1)}"
        r" - \sum_{j=i+1}^{n} a_{ij}x_j^{(k)} \right) / a_{ii}$"
    )

    def _create_stepper(self, A, b, x0, tol):
        return GaussSeidelStepper(A, b, x0, tol=tol, safety_limit=100)

    # Gauss-Seidel: Terme mit uses_new=True nutzen k_new
    def _term_x_symbol(self, term, j, k_old, k_new):
        uses_new = term[4] if len(term) > 4 else False
        return self._x_symbol(j + 1, k_new if uses_new else k_old)

    def _append_step_card(self, A: List[List[float]], b: List[float],
                           step: GaussStep):
        self.step_count += 1
        n = len(A)

        title = (f"Iteration {step.iteration}" if step.kind == "iter"
                 else f"Abschluss nach Iteration {step.iteration}")

        card = ttk.Frame(self.history.inner, padding=12, style="Card.TFrame")
        card.pack(fill="x", pady=8, padx=2)
        ttk.Label(card, text=title, style="Header.TLabel",
                  padding=(10, 8)).pack(fill="x")

        body = ttk.Frame(card, style="Card.TFrame")
        body.pack(fill="x", pady=(12, 0))
        body.columnconfigure(0, weight=0); body.columnconfigure(1, weight=1)

        left  = ttk.Frame(body, style="Card.TFrame")
        right = ttk.Frame(body, style="Card.TFrame")
        left.grid(row=0, column=0, sticky="nw", padx=(0, 24))
        right.grid(row=0, column=1, sticky="nsew")

        # Matrix A | b
        grid = ttk.Frame(left, style="Card.TFrame")
        grid.pack(anchor="w")

        for c in range(n):
            tk.Label(grid, text="", bg=self.card_bg, width=10,
                     font=("Consolas", 12, "bold")).grid(
                row=0, column=c, padx=3, pady=(0, 4))
        tk.Label(grid, text="", bg=self.card_bg,
                 font=("Consolas", 12, "bold")).grid(
            row=0, column=n, padx=(8, 8), pady=(0, 4))
        tk.Label(grid, text="Matrix A", bg=self.card_bg, width=10,
                 font=("Segoe UI", 12, "bold")).grid(
            row=0, column=n-2, padx=3, pady=(0, 4))
        tk.Label(grid, text="b", bg=self.card_bg, width=10,
                 font=("Segoe UI", 12, "bold")).grid(
            row=0, column=n+1, padx=3, pady=(0, 4))

        for r in range(n):
            gr = r + 1
            for c in range(n):
                bg = self.bg_diag if r == c else self.bg_offdiag
                tk.Label(grid, text=self._fmt(A[r][c]), width=10, bg=bg,
                         relief="solid", borderwidth=1,
                         font=("Consolas", 12)).grid(
                    row=gr, column=c, padx=3, pady=3)
            tk.Label(grid, text="|", bg=self.card_bg,
                     font=("Consolas", 12)).grid(
                row=gr, column=n, padx=(8, 8), pady=3)
            tk.Label(grid, text=self._fmt(b[r]), width=10, bg=self.bg_rhs,
                     relief="solid", borderwidth=1,
                     font=("Consolas", 12)).grid(
                row=gr, column=n+1, padx=3, pady=3)

        # Legende
        legend = ttk.Frame(left, style="Card.TFrame")
        legend.pack(anchor="w", pady=(10, 0))
        self._legend_item(legend, "aᵢᵢ",   self.bg_diag,    0)
        self._legend_item(legend, "aᵢⱼ",   self.bg_offdiag, 1)
        self._legend_item(legend, "bᵢ",    self.bg_rhs,     2)
        self._legend_item(legend, "x⁽k⁾",  self.bg_oldx,    3)
        self._legend_item(legend, "x⁽k+1⁾",self.bg_newx,    4)

        # Berechnungsbereich
        calc_outer = tk.Frame(right, bg=self.calc_bg, relief="solid",
                               borderwidth=1, highlightthickness=1,
                               highlightbackground=self.border)
        calc_outer.pack(fill="x", expand=True)
        k_old = step.iteration - 1; k_new = step.iteration
        calc_grid = tk.Frame(calc_outer, bg=self.calc_bg)
        calc_grid.pack(fill="x", padx=18, pady=12)

        tk.Label(calc_grid, text=f"Rechnung {step.iteration}",
                 bg=self.calc_bg, fg=self.text_dark,
                 font=("Segoe UI", 12, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 8))
        tk.Label(calc_grid,
                 text=f"{self._x_symbol(None, k_old)} = ["
                      + ", ".join(self._fmt(v) for v in step.x_old) + "]",
                 bg=self.bg_oldx, fg=self.text_dark,
                 font=("Segoe UI", 12), anchor="w",
                 padx=10, pady=6).grid(
            row=1, column=0, sticky="ew", pady=(0, 10))

        for idx, detail in enumerate(step.row_details):
            self._append_colored_math_row(calc_grid, detail, k_old, k_new, 2 + idx)

        tk.Label(calc_grid,
                 text=f"{self._x_symbol(None, k_new)} = ["
                      + ", ".join(self._fmt(v) for v in step.x_new) + "]",
                 bg=self.bg_newx, fg=self.text_dark,
                 font=("Segoe UI", 12, "bold"), anchor="w",
                 padx=10, pady=6).grid(
            row=2 + len(step.row_details), column=0,
            sticky="ew", pady=(10, 0))

        self.history.canvas.update_idletasks()
        self.history.canvas.yview_moveto(1.0)


def main():
    app = GaussSeidelGUI()
    app.mainloop()


if __name__ == "__main__":
    main()