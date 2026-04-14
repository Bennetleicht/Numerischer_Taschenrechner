from __future__ import annotations

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Tuple

from base_gui.gui_utils import ScrollableFrame

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from GUI import maximize_and_lock


def _maximize_window(w: tk.Tk):
    """Fenster maximieren und Größe sperren."""
    maximize_and_lock(w)


class BaseLGSInkrementellGUI(tk.Tk):
    """
    Gemeinsame Basis für Gauss-, Cholesky- und LR-GUI.
    """

    # Unterklassen setzen diesen Klassennamen
    title_text: str = "LGS"

    def __init__(self):
        super().__init__()
        self.title(self.title_text)
        self.geometry("1280x720")
        _maximize_window(self)

        self.n_var = tk.StringVar(value="4")
        self.entries_A: List[List[tk.Entry]] = []
        self.entries_b: List[tk.Entry] = []

        self.stepper = None
        self.step_count = 0
        self.started = False

        self._init_colors()
        self._init_style()
        self._build_ui()
        self._build_input_matrix()

    # ── Farben ────────────────────────────────────────────────────────────
    def _init_colors(self):
        """Gemeinsame Farben. Unterklasse kann erweitern."""
        self.bg_default = "#ffffff"
        self.bg_pivot   = "#ffe08a"
        self.bg_changed = "#ffb3c6"
        self.card_bg    = "#ffffff"
        self.header_bg  = "#f3f4f6"

    # ── Style (100% identisch in gauss/cholesky/lr) ───────────────────────
    def _init_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("App.TFrame",   background="#f6f7fb")
        style.configure("Card.TFrame",  background=self.card_bg,
                        relief="solid", borderwidth=1)
        style.configure("Title.TLabel", background="#f6f7fb",
                        font=("Segoe UI", 14, "bold"))
        style.configure("Hint.TLabel",  background="#f6f7fb",
                        font=("Segoe UI", 9), foreground="#6b7280")
        style.configure("Small.TLabel", background=self.card_bg,
                        font=("Segoe UI", 9))
        style.configure("TButton",      font=("Segoe UI", 10))
        style.configure("TSpinbox",     font=("Segoe UI", 10))

    # ── UI-Grundstruktur (99% identisch in gauss/cholesky/lr) ────────────
    def _build_ui(self):
        main = ttk.Frame(self, padding=10, style="App.TFrame")
        main.pack(fill="both", expand=True)

        head = ttk.Frame(main, style="App.TFrame")
        head.pack(fill="x", pady=(0, 8))
        ttk.Label(head, text=self.title_text,
                  style="Title.TLabel").pack(side="left")

        controls = ttk.Frame(main, style="App.TFrame")
        controls.pack(fill="x", pady=(0, 8))

        ttk.Label(controls, text="Größe n:",
                  style="Hint.TLabel").pack(side="left")
        self.n_spin = ttk.Spinbox(controls, from_=2, to=10,
                                   textvariable=self.n_var, width=5)
        self.n_spin.pack(side="left", padx=8)

        vcmd = (self.register(self._validate_n_input), "%P")
        self.n_spin.configure(validate="key", validatecommand=vcmd)
        self.n_var.trace_add("write", self._on_n_changed)
        self.n_spin.bind("<Return>",   lambda _e: self._rebuild_input())
        self.n_spin.bind("<FocusOut>", lambda _e: self._rebuild_input())

        # Hook: Unterklasse kann extra Controls hinzufügen
        self._extra_controls(controls)

        self.btn_start = ttk.Button(controls, text="Start",
                                     command=self.on_start_or_next)
        self.btn_reset = ttk.Button(controls, text="Reset",
                                     command=self.on_reset)
        self.btn_start.pack(side="left", padx=6)
        self.btn_reset.pack(side="left", padx=6)

        self.input_matrix_frame = ttk.Frame(main, style="App.TFrame")
        self.input_matrix_frame.pack(fill="x")

        ttk.Separator(main).pack(fill="x", pady=8)

        self.history = ScrollableFrame(main)
        self.history.pack(fill="both", expand=True)

    def _extra_controls(self, controls: ttk.Frame):
        """Hook: Unterklasse fügt hier z.B. Pivot-Modus-Buttons ein."""
        pass

    # ── n-Eingabe (100% identisch) ────────────────────────────────────────
    def _validate_n_input(self, proposed: str) -> bool:
        return proposed == "" or (proposed.isdigit() and 2 <= int(proposed) <= 10)

    def _on_n_changed(self, *_args):
        try:
            n = int(self.n_var.get())
            if 2 <= n <= 10:
                self._rebuild_input()
        except ValueError:
            pass

    def _get_n(self) -> int:
        try:
            n = int(self.n_var.get())
            return n if 2 <= n <= 10 else 4
        except ValueError:
            return 4

    # ── Eingabe sperren (identisch ohne gauss-spezifisches) ───────────────
    def _set_input_locked(self, locked: bool):
        state = "disabled" if locked else "normal"
        self.n_spin.configure(state=state)
        for row in self.entries_A:
            for e in row:
                e.configure(state=state)
        for e in self.entries_b:
            e.configure(state=state)

    # ── Matrix aufbauen (100% identisch in cholesky/lr) ───────────────────
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

        grid = ttk.Frame(box, style="Card.TFrame")
        grid.pack(anchor="w")

        for r in range(n):
            row_entries = []
            for c in range(n):
                e = tk.Entry(grid, width=10, justify="center",
                             bg=self.bg_default, relief="solid",
                             borderwidth=1, font=("Consolas", 12))
                e.grid(row=r, column=c, padx=3, pady=3)
                e.insert(0, "0")
                row_entries.append(e)
            self.entries_A.append(row_entries)

            ttk.Label(grid, text="|",
                      style="Small.TLabel").grid(row=r, column=n, padx=10)

            eb = tk.Entry(grid, width=10, justify="center",
                          bg=self.bg_default, relief="solid",
                          borderwidth=1, font=("Consolas", 12))
            eb.grid(row=r, column=n + 1, padx=3, pady=3)
            eb.insert(0, "0")
            self.entries_b.append(eb)

    # ── Eingaben lesen (100% identisch) ───────────────────────────────────
    def _read_inputs(self) -> Tuple[List[List[float]], List[float]]:
        n = self._get_n()
        A, b = [], []
        for r in range(n):
            row = []
            for c in range(n):
                try:
                    row.append(float(self.entries_A[r][c].get().replace(",", ".")))
                except ValueError:
                    raise ValueError(f"Ungültiger Wert in A[{r+1}][{c+1}]")
            A.append(row)
            try:
                b.append(float(self.entries_b[r].get().replace(",", ".")))
            except ValueError:
                raise ValueError(f"Ungültiger Wert in b[{r+1}]")
        return A, b

    # ── Formatierung (100% identisch) ─────────────────────────────────────
    def _fmt(self, x: float) -> str:
        if x == int(x):
            return str(int(x))
        return f"{x:.4f}".rstrip("0").rstrip(".")

    # ── Reset (Kern identisch in cholesky/lr) ─────────────────────────────
    def on_reset(self):
        self.stepper = None
        self.started = False
        self.btn_start.configure(text="Start")
        self._set_input_locked(False)
        self.history.clear()
        self.step_count = 0

        for row in self.entries_A:
            for e in row:
                e.delete(0, tk.END)
                e.insert(0, "0")
                e.configure(bg=self.bg_default)
        for e in self.entries_b:
            e.delete(0, tk.END)
            e.insert(0, "0")

        self.history.canvas.update_idletasks()
        self.history.canvas.yview_moveto(0.0)
        # Hook für verfahrensspezifischen Reset
        self._on_reset_extra()

    def _on_reset_extra(self):
        """Hook: gauss überschreibt für Pivot-Reset."""
        pass

    # ── Start/Weiter (Grundstruktur identisch) ────────────────────────────
    def on_start_or_next(self):
        if self.started:
            self._do_one_step()
            return

        if self.step_count > 0:
            for w in self.history.inner.winfo_children():
                w.destroy()
            self.step_count = 0
        self.stepper = None

        try:
            A, b = self._read_inputs()
        except ValueError as e:
            messagebox.showerror("Eingabefehler", str(e))
            return

        # Unterklasse erstellt Stepper (kann None zurückgeben bei Fehler)
        stepper = self._create_stepper(A, b)
        if stepper is None:
            return

        self.stepper = stepper
        self.started = True
        self.btn_start.configure(text="Weiter")
        self._set_input_locked(True)
        self.after(0, self._do_one_step)

    # ── Rebuild (Kern identisch) ──────────────────────────────────────────
    def _rebuild_input(self):
        self.stepper = None
        self.started = False
        self.btn_start.configure(text="Start")
        self._rebuild_extra()
        self._build_input_matrix()

    def _rebuild_extra(self):
        """Hook: gauss überschreibt für Pivot/Custom-Reset."""
        pass

    # ── Matrix/Vektor zeichnen (91-100% identisch) ────────────────────────
    def _draw_matrix(self, parent: ttk.Frame, name: str,
                     M: List[List[float]], changed: set,
                     pivot_r: int, pivot_c: int, step):
        grid = ttk.Frame(parent, style="Card.TFrame")
        grid.pack(anchor="w", pady=(4, 0))
        for r in range(len(M)):
            for c in range(len(M[0])):
                val = self._fmt(M[r][c])
                bg = self.bg_default
                if (name, r, c) in changed:
                    bg = self.bg_changed
                if self._is_pivot_cell(name, r, c, pivot_r, pivot_c, step):
                    bg = self.bg_pivot
                tk.Label(grid, text=val, width=12, bg=bg,
                         relief="solid", borderwidth=1,
                         font=("Consolas", 12)).grid(
                    row=r, column=c, padx=2, pady=2)

    def _is_pivot_cell(self, name: str, r: int, c: int,
                       pivot_r: int, pivot_c: int, step) -> bool:
        """Hook: Unterklasse definiert welche Zelle die Pivot-Zelle ist."""
        return False

    def _draw_vector(self, parent: ttk.Frame, name: str,
                     v: List[float], changed: set):
        grid = ttk.Frame(parent, style="Card.TFrame")
        grid.pack(anchor="w", pady=(4, 0))
        for r in range(len(v)):
            val = self._fmt(v[r])
            bg = self.bg_changed if (name, r, 0) in changed else self.bg_default
            tk.Label(grid, text=val, width=12, bg=bg,
                     relief="solid", borderwidth=1,
                     font=("Consolas", 12)).grid(
                row=r, column=0, padx=2, pady=2)

    # ── Abstrakte Methoden ────────────────────────────────────────────────
    def _create_stepper(self, A: List[List[float]], b: List[float]):
        """Erstellt und gibt den verfahrensspezifischen Stepper zurück."""
        raise NotImplementedError

    def _append_step_card(self, *args, **kwargs):
        raise NotImplementedError

    def _append_final_solution_card(self):
        raise NotImplementedError

    def _do_one_step(self):
        raise NotImplementedError