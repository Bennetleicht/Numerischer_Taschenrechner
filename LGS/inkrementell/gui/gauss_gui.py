from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Optional, Tuple

from gauss_solver import GaussEliminationSolver, Step
from base_gui.gui_utils import PivotTile
from gui.base_lgs_inkrementell_gui import BaseLGSInkrementellGUI


class GaussEliminationStepper(GaussEliminationSolver):
    pass


class GaussGUI(BaseLGSInkrementellGUI):

    title_text = "Gauss-Elimination"

    def __init__(self):
        # Gauss-spezifischer Zustand vor super().__init__
        self.pivot_mode: Optional[str] = None
        self.custom_pivot: Optional[Tuple[int, int]] = None
        self.custom_select_active = False
        self.custom_prev: Optional[Tuple[int, int]] = None
        super().__init__()

    # ── Extra Farben ──────────────────────────────────────────────────────
    def _init_colors(self):
        super()._init_colors()
        self.custom_selected_bg  = self.bg_pivot
        self.mode_bg             = "#ffffff"
        self.mode_bg_selected    = "#dbeafe"
        self.mode_border         = "#d1d5db"
        self.mode_border_selected = "#3b82f6"

    # ── Pivot-spezifische Eingabesperre ───────────────────────────────────
    def _set_input_locked(self, locked: bool):
        super()._set_input_locked(locked)
        self._refresh_mode_visuals()

    # ── Matrix-Eingabe mit Pivot-Tiles ────────────────────────────────────
    def _build_input_matrix(self):
        for w in self.input_matrix_frame.winfo_children():
            w.destroy()
        self.entries_A.clear()
        self.entries_b.clear()
        n = self._get_n()

        box = ttk.Frame(self.input_matrix_frame, padding=8, style="Card.TFrame")
        box.pack(fill="x")
        top = ttk.Frame(box, style="Card.TFrame")
        top.pack(fill="x", pady=(0, 4))
        ttk.Label(top, text="Eingabe: Matrix A | Vektor b",
                  style="Small.TLabel").pack(side="left")

        content = ttk.Frame(box, style="Card.TFrame")
        content.pack(fill="x")
        content.columnconfigure(0, weight=1); content.columnconfigure(1, weight=0)

        grid = ttk.Frame(content, style="Card.TFrame")
        grid.grid(row=0, column=0, sticky="w")

        for r in range(n):
            row_entries = []
            for c in range(n):
                e = tk.Entry(grid, width=10, justify="center",
                             bg=self.bg_default, relief="solid",
                             borderwidth=1, font=("Consolas", 12))
                e.grid(row=r, column=c, padx=3, pady=3)
                e.insert(0, "0")
                e.bind("<Button-1>",
                       lambda _ev, rr=r, cc=c: self._on_A_cell_click(rr, cc))
                row_entries.append(e)
            self.entries_A.append(row_entries)

            ttk.Label(grid, text="|",
                      style="Sub.TLabel").grid(row=r, column=n, padx=10)

            eb = tk.Entry(grid, width=10, justify="center",
                          bg=self.bg_default, relief="solid",
                          borderwidth=1, font=("Consolas", 12))
            eb.grid(row=r, column=n + 1, padx=3, pady=3)
            eb.insert(0, "0")
            self.entries_b.append(eb)

        # Pivot-Modus-Tiles rechts
        piv = ttk.Frame(content, padding=(10, 0), style="Card.TFrame")
        piv.grid(row=0, column=1, sticky="n")
        ttk.Label(piv, text="Pivot-Modus",
                  style="Sub.TLabel").pack(anchor="w", pady=(0, 6))

        tiles = tk.Frame(piv, bg=self.card_bg)
        tiles.pack()

        self.tile_1 = PivotTile(tiles, "1  Spaltenpivot",
                                 lambda: self._set_pivot_mode("col"))
        self.tile_2 = PivotTile(tiles, "2  Totalpivot",
                                 lambda: self._set_pivot_mode("total"))
        self.tile_3 = PivotTile(tiles, "3  Zeilenpivot",
                                 lambda: self._set_pivot_mode("row"))
        self.tile_4 = PivotTile(tiles, "4  Eigene Wahl",
                                 lambda: self._set_pivot_mode("custom"))

        self.tile_1.grid(row=0, column=0, padx=6, pady=6, sticky="ew")
        self.tile_2.grid(row=0, column=1, padx=6, pady=6, sticky="ew")
        self.tile_3.grid(row=1, column=0, padx=6, pady=6, sticky="ew")
        self.tile_4.grid(row=1, column=1, padx=6, pady=6, sticky="ew")
        tiles.columnconfigure(0, weight=1); tiles.columnconfigure(1, weight=1)

        self.lbl_custom = ttk.Label(piv, text="", style="Small.TLabel")
        self.lbl_custom.pack(anchor="w", pady=(8, 0))
        self.lbl_custom_hint = ttk.Label(piv, text="", style="Small.TLabel")
        self.lbl_custom_hint.pack(anchor="w", pady=(2, 0))

        self._refresh_mode_visuals()

    # ── Pivot-Logik ───────────────────────────────────────────────────────
    def _refresh_mode_visuals(self):
        if not hasattr(self, "tile_1"):
            return
        tiles_state = "disabled" if self.started else "normal"
        for tile in (self.tile_1, self.tile_2, self.tile_3, self.tile_4):
            tile.configure(cursor="arrow" if self.started else "hand2")
            tile.label.configure(state=tiles_state)

        self.tile_1.set_selected(self.pivot_mode == "col",
            self.mode_bg, self.mode_bg_selected,
            self.mode_border_selected, self.mode_border)
        self.tile_2.set_selected(self.pivot_mode == "total",
            self.mode_bg, self.mode_bg_selected,
            self.mode_border_selected, self.mode_border)
        self.tile_3.set_selected(self.pivot_mode == "row",
            self.mode_bg, self.mode_bg_selected,
            self.mode_border_selected, self.mode_border)
        self.tile_4.set_selected(self.pivot_mode == "custom",
            self.mode_bg, self.mode_bg_selected,
            self.mode_border_selected, self.mode_border)

        if self.pivot_mode == "custom":
            if self.custom_pivot is None:
                self.lbl_custom.configure(text="Eigenes Pivot: nicht gesetzt")
                self.lbl_custom_hint.configure(
                    text="Bitte Pivot-Element in Matrix A anklicken.")
            else:
                r, c = self.custom_pivot
                self.lbl_custom.configure(text=f"Eigenes Pivot: A[{r+1},{c+1}]")
                self.lbl_custom_hint.configure(text="")
        else:
            self.lbl_custom.configure(text="")
            self.lbl_custom_hint.configure(text="")

    def _set_pivot_mode(self, mode: str):
        if self.started:
            return
        self.pivot_mode = mode
        if mode != "custom":
            self.custom_select_active = False
            self._clear_custom_highlight()
            self.custom_pivot = None
        else:
            self.custom_select_active = True
        self._refresh_mode_visuals()

    def _clear_custom_highlight(self):
        if self.custom_prev is None:
            return
        r, c = self.custom_prev
        if 0 <= r < len(self.entries_A) and 0 <= c < len(self.entries_A[r]):
            self.entries_A[r][c].configure(bg=self.bg_default)
        self.custom_prev = None

    def _on_A_cell_click(self, r: int, c: int):
        if self.started or self.pivot_mode != "custom":
            return
        if not self.custom_select_active:
            return
        self._clear_custom_highlight()
        self.entries_A[r][c].configure(bg=self.custom_selected_bg)
        self.custom_prev = (r, c)
        self.custom_pivot = (r, c)
        self.custom_select_active = False
        self._refresh_mode_visuals()

    # ── Hooks für Reset/Rebuild ───────────────────────────────────────────
    def _on_reset_extra(self):
        self.custom_pivot = None
        self.custom_select_active = False
        self._clear_custom_highlight()
        self.pivot_mode = None
        self._refresh_mode_visuals()

    def _rebuild_extra(self):
        self.custom_pivot = None
        self.custom_select_active = False
        self._clear_custom_highlight()
        self.pivot_mode = None

    # ── Stepper erstellen ─────────────────────────────────────────────────
    def _create_stepper(self, A, b):
        if self.pivot_mode is None:
            self.pivot_mode = "col"
            self._refresh_mode_visuals()
        if self.pivot_mode == "custom" and self.custom_pivot is None:
            messagebox.showerror(
                "Pivot-Modus",
                "Bitte Pivot-Element in Matrix A anklicken (Eigene Wahl).")
            return None
        return GaussEliminationStepper(
            A, b,
            pivot_mode=self.pivot_mode,
            custom_pivot=self.custom_pivot,
            change_tol=1e-10)

    # ── on_start_or_next: gauss braucht _refresh_mode_visuals danach ──────
    def on_start_or_next(self):
        super().on_start_or_next()
        self._refresh_mode_visuals()

    # ── Einen Schritt ausführen ───────────────────────────────────────────
    def _do_one_step(self):
        if not self.stepper:
            return
        step = self.stepper.next_step()
        self._append_step_card(self.stepper.M, step)
        if step.kind == "done":
            self._append_final_solution_card()
            self.started = False
            self._refresh_mode_visuals()

    # ── Schrittkarte ──────────────────────────────────────────────────────
    def _append_step_card(self, M: List[List[float]], step: Step,
                           extra_title: Optional[str] = None):
        self.step_count += 1
        n = len(M)
        kind_map = {"swap": "TAUSCH", "elim": "ELIMINATION",
                    "backsub": "RÜCKWÄRTS", "done": "FERTIG"}
        kind_txt = kind_map.get(step.kind, step.kind.upper())
        title = extra_title or f"Schritt {self.step_count}: {kind_txt}"

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

        grid = ttk.Frame(left, style="Card.TFrame")
        grid.pack(anchor="w")

        pivot_r, pivot_c = step.pivot
        changed = set(step.changed_cells or [])

        for r in range(n):
            for c in range(n + 1):
                val = self._fmt(M[r][c])
                if step.kind == "backsub" and r == pivot_r:
                    bg = self.bg_changed
                elif r == pivot_r and c == pivot_c:
                    bg = self.bg_pivot
                elif (r, c) in changed:
                    bg = self.bg_changed
                else:
                    bg = self.bg_default

                if c == n:
                    tk.Label(grid, text="|", bg=self.card_bg,
                             font=("Consolas", 12)).grid(
                        row=r, column=c, padx=(10, 10))
                    col_index = c + 1
                else:
                    col_index = c

                tk.Label(grid, text=val, width=12, bg=bg,
                         relief="solid", borderwidth=1,
                         font=("Consolas", 12)).grid(
                    row=r, column=col_index, padx=2, pady=2)

        msg = (step.message or "").strip()
        txt = tk.Text(right,
                      height=max(6, min(12, msg.count("\n") + 4)),
                      wrap="word", relief="solid", borderwidth=1,
                      font=("Segoe UI", 10))
        txt.pack(fill="x", expand=False)
        txt.insert("1.0", msg)
        txt.configure(state="disabled")

        if self.stepper and step.kind in ("backsub", "done"):
            x_out = self.stepper.get_solution_original_order()
            x_preview = ", ".join(self._fmt(v) for v in x_out)
            ttk.Label(right, text=f"x = [{x_preview}]",
                      style="Small.TLabel").pack(anchor="w", pady=(8, 0))

        self.history.canvas.update_idletasks()
        self.history.canvas.yview_moveto(1.0)

    def _append_final_solution_card(self):
        if not self.stepper:
            return
        x_out = self.stepper.get_solution_original_order()
        parts = [f"x{i+1} = {self._fmt(v)}" for i, v in enumerate(x_out)]
        sol = "   |   ".join(parts)
        dummy = Step(kind="done", pivot=(0, 0), message="LÖSUNG\n" + sol)
        self._append_step_card(self.stepper.M, dummy,
                               extra_title="LÖSUNG (Endergebnis)")
        messagebox.showinfo("Lösung", sol)


def main():
    app = GaussGUI()
    app.mainloop()


if __name__ == "__main__":
    main()