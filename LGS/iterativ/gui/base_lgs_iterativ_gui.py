"""
Gemeinsame Basis-GUI für iterative LGS-Verfahren.
Jacobi und Gauss-Seidel sind zu ~95% identisch.

Geteilt (100%):
  _init_colors, _init_style, _build_ui, _fmt, _sup_digit, _sub_digit,
  _x_symbol, _set_input_locked, _validate_n_input, _on_n_changed, _get_n,
  _rebuild_input, _build_input_matrix, _read_inputs, _legend_item, on_reset

Geteilt mit kleinen Unterschieden → in Base:
  _render_formula   → Unterklasse liefert die Formel-Strings
  _append_colored_math_row → 90% gleich, Unterschied: uses_new Flag
  _append_final_solution_card → nur messagebox-Titel verschieden
  on_start_or_next  → nur Stepper-Typ verschieden

Unterklasse muss implementieren:
  title_text: str
  formula_latex: str           – LaTeX-String für die Iterations-Formel
  formula_label: str           – Label über der Formel
  _create_stepper(A, b, x0, tol)
  _append_step_card(...)
  _solution_title: str         – Titel für Ergebnis-Messagebox
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Tuple, Optional

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from base_gui.gui_utils import ScrollableFrame, maximize_window


class BaseIterativGUI(tk.Tk):

    title_text:    str = "Iteratives LGS"
    formula_latex: str = ""
    formula_label: str = "Formel"
    _solution_title: str = "Ergebnis"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.title(self.title_text)
        self.minsize(480, 320)
        maximize_window(self)

        self.n_var   = tk.StringVar(value="4")
        self.tol_var = tk.StringVar(value="1e-6")

        self.entries_A:  List[List[tk.Entry]] = []
        self.entries_b:  List[tk.Entry] = []
        self.entries_x0: List[tk.Entry] = []

        self.stepper = None
        self.started = False
        self.step_count = 0
        self.formula_canvas = None

        self._init_colors()
        self._init_style()
        self.configure(bg=self.app_bg)
        self._build_ui()
        self._build_input_matrix()

    # ── Farben (100% identisch) ───────────────────────────────────────────
    def _init_colors(self):
        self.bg_default  = "#ffffff"
        self.bg_diag     = "#f0d484"
        self.bg_offdiag  = "#e9bcc9"
        self.bg_rhs      = "#b8e0b8"
        self.bg_oldx     = "#f4d2a7"
        self.bg_newx     = "#b8d5f0"
        self.card_bg     = "#ffffff"
        self.header_bg   = "#eef1f5"
        self.app_bg      = "#f6f7fb"
        self.formula_bg  = "#f3f3f5"
        self.calc_bg     = "#fbfbfd"
        self.border      = "#cfd4dc"
        self.text_dark   = "#1f2937"

    # ── Style (100% identisch) ────────────────────────────────────────────
    def _init_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("App.TFrame",    background=self.app_bg)
        style.configure("Card.TFrame",   background=self.card_bg,
                        relief="solid", borderwidth=1)
        style.configure("Header.TLabel", background=self.header_bg,
                        font=("Segoe UI", 12, "bold"))
        style.configure("Sub.TLabel",    background=self.card_bg,
                        font=("Segoe UI", 10))
        style.configure("Small.TLabel",  background=self.card_bg,
                        font=("Segoe UI", 9))
        style.configure("Title.TLabel",  background=self.app_bg,
                        font=("Segoe UI", 14, "bold"))
        style.configure("Hint.TLabel",   background=self.app_bg,
                        foreground="#4b5563", font=("Segoe UI", 9))
        style.configure("TButton",       font=("Segoe UI", 10))
        style.configure("TSpinbox",      font=("Segoe UI", 10))

    # ── UI (99% identisch) ────────────────────────────────────────────────
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
        self.n_spin = ttk.Spinbox(controls, from_=2, to=7,
                                   textvariable=self.n_var, width=5)
        self.n_spin.pack(side="left", padx=(8, 16))

        ttk.Label(controls, text="Toleranz:",
                  style="Hint.TLabel").pack(side="left")
        self.tol_entry = ttk.Entry(controls, textvariable=self.tol_var,
                                    width=10)
        self.tol_entry.pack(side="left", padx=(8, 16))

        self.btn_start = ttk.Button(controls, text="Start",
                                     command=self.on_start_or_next)
        self.btn_reset = ttk.Button(controls, text="Reset",
                                     command=self.on_reset)
        self.btn_start.pack(side="left", padx=6)
        self.btn_reset.pack(side="left", padx=6)

        vcmd = (self.register(self._validate_n_input), "%P")
        self.n_spin.configure(validate="key", validatecommand=vcmd)
        self.n_var.trace_add("write", self._on_n_changed)
        self.n_spin.bind("<Return>",   lambda _e: self._rebuild_input())
        self.n_spin.bind("<FocusOut>", lambda _e: self._rebuild_input())

        self.input_matrix_frame = ttk.Frame(main, style="App.TFrame")
        self.input_matrix_frame.pack(fill="x")

        ttk.Separator(main).pack(fill="x", pady=8)

        self.history = ScrollableFrame(main)
        self.history.pack(fill="both", expand=True)

    # ── Hilfsmethoden (100% identisch) ───────────────────────────────────
    def _fmt(self, x: float) -> str:
        if abs(x) < 1e-12:
            x = 0.0
        return "{:.6g}".format(x)

    def _sup_digit(self, s) -> str:
        return str(s).translate(str.maketrans("0123456789-()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁽⁾"))

    def _sub_digit(self, s) -> str:
        return str(s).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))

    def _x_symbol(self, idx: Optional[int] = None,
                  iteration: Optional[int] = None) -> str:
        base = "x" if idx is None else f"x{self._sub_digit(idx)}"
        if iteration is None:
            return base
        return f"{base}{self._sup_digit(f'({iteration})')}"

    def _validate_n_input(self, proposed: str) -> bool:
        return proposed == "" or (proposed.isdigit() and 2 <= int(proposed) <= 10)

    def _on_n_changed(self, *_args):
        try:
            if 2 <= int(self.n_var.get()) <= 10:
                self._rebuild_input()
        except Exception:
            pass

    def _get_n(self) -> int:
        try:
            return int(self.n_var.get())
        except Exception:
            return 4

    def _rebuild_input(self):
        self.stepper = None
        self.started = False
        self.btn_start.configure(text="Start")
        self._build_input_matrix()

    def _set_input_locked(self, locked: bool):
        state = "disabled" if locked else "normal"
        self.n_spin.configure(state=state)
        self.tol_entry.configure(state=state)
        for row in self.entries_A:
            for e in row:
                e.configure(state=state)
        for e in self.entries_b:
            e.configure(state=state)
        for e in self.entries_x0:
            e.configure(state=state)

    # ── Matrix-Eingabe (100% identisch) ───────────────────────────────────
    def _build_input_matrix(self):
        for w in self.input_matrix_frame.winfo_children():
            w.destroy()
        self.entries_A.clear()
        self.entries_b.clear()
        self.entries_x0.clear()

        n = self._get_n()
        box = ttk.Frame(self.input_matrix_frame, padding=8, style="Card.TFrame")
        box.pack(fill="x")

        top = ttk.Frame(box, style="Card.TFrame")
        top.pack(fill="x", pady=(0, 6))
        ttk.Label(top, text="Eingabe: Matrix A | Vektor b | Startvektor x⁽⁰⁾",
                  style="Sub.TLabel").pack(side="left")

        content = ttk.Frame(box, style="Card.TFrame")
        content.pack(fill="x")
        content.columnconfigure(0, weight=0)
        content.columnconfigure(1, weight=1)

        left = ttk.Frame(content, style="Card.TFrame")
        left.grid(row=0, column=0, sticky="nw", padx=(0, 24))
        grid = ttk.Frame(left, style="Card.TFrame")
        grid.pack(anchor="w")

        for r in range(n):
            row_entries = []
            for c in range(n):
                e = tk.Entry(grid, width=8, justify="center",
                             bg=self.bg_default, relief="solid",
                             borderwidth=1, font=("Consolas", 12))
                e.grid(row=r, column=c, padx=3, pady=3)
                e.insert(0, "0")
                row_entries.append(e)
            self.entries_A.append(row_entries)

            ttk.Label(grid, text="|",
                      style="Sub.TLabel").grid(row=r, column=n, padx=8)

            eb = tk.Entry(grid, width=8, justify="center",
                          bg=self.bg_default, relief="solid",
                          borderwidth=1, font=("Consolas", 12))
            eb.grid(row=r, column=n+1, padx=3, pady=3)
            eb.insert(0, "0")
            self.entries_b.append(eb)

            ttk.Label(grid, text="|",
                      style="Sub.TLabel").grid(row=r, column=n+2, padx=8)

            ex = tk.Entry(grid, width=8, justify="center",
                          bg=self.bg_default, relief="solid",
                          borderwidth=1, font=("Consolas", 12))
            ex.grid(row=r, column=n+3, padx=3, pady=3)
            ex.insert(0, "0")
            self.entries_x0.append(ex)

        right = ttk.Frame(content, padding=(10, 0), style="Card.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        ttk.Label(right, text=self.formula_label,
                  style="Sub.TLabel").pack(anchor="w", pady=(0, 8))

        formula_holder = tk.Frame(right, bg=self.formula_bg,
                                   relief="solid", borderwidth=1,
                                   width=1020, height=150)
        formula_holder.pack_propagate(False)
        formula_holder.pack(anchor="w")
        self._render_formula(formula_holder)

    # ── Formel-Canvas (Unterklasse setzt formula_latex) ───────────────────
    def _render_formula(self, parent):
        fig = Figure(figsize=(10, 1.8), dpi=120)
        ax  = fig.add_subplot(111)
        fig.patch.set_facecolor(self.formula_bg)
        ax.set_facecolor(self.formula_bg)
        ax.axis("off")
        ax.text(0.02, 0.5, self.formula_latex,
                fontsize=20, ha="left", va="center")
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.configure(width=1000, height=140)
        widget.pack(padx=8, pady=8)
        self.formula_canvas = canvas

    # ── Eingaben lesen (100% identisch) ───────────────────────────────────
    def _read_inputs(self) -> Tuple[List[List[float]], List[float],
                                     List[float], float]:
        n = self._get_n()
        A: List[List[float]] = []
        b: List[float] = []
        x0: List[float] = []
        try:
            for r in range(n):
                row = [float(self.entries_A[r][c].get().strip())
                       for c in range(n)]
                A.append(row)
                b.append(float(self.entries_b[r].get().strip()))
                x0.append(float(self.entries_x0[r].get().strip()))
            tol = float(self.tol_var.get().strip())
        except ValueError:
            raise ValueError("Ungültige Eingabe: Bitte nur Zahlen eintragen.")
        if tol <= 0:
            raise ValueError("Toleranz muss > 0 sein.")
        return A, b, x0, tol

    # ── Legende (100% identisch) ──────────────────────────────────────────
    def _legend_item(self, parent, text, color, col):
        frame = ttk.Frame(parent, style="Card.TFrame")
        frame.grid(row=0, column=col, padx=(0, 10), sticky="w")
        tk.Label(frame, bg=color, width=2,
                 relief="solid", borderwidth=1).pack(side="left", padx=(0, 4))
        ttk.Label(frame, text=text, style="Small.TLabel").pack(side="left")

    # ── Canvas-Zeile (90% identisch — Hook für uses_new) ──────────────────
    def _append_colored_math_row(self, parent, detail, k_old: int,
                                  k_new: int, row: int):
        """
        Zeichnet eine farbige Berechnungszeile.
        detail.terms ist eine Liste von Tupeln — Unterklasse überschreibt
        _get_term_x_symbol() um den richtigen x-Symbol zu liefern.
        """
        ROW_H = 28; PADY_BOX = 3
        font_main = ("Cambria Math", 13)
        font_bold = ("Cambria Math", 13, "bold")

        cv = tk.Canvas(parent, bg=self.calc_bg, height=ROW_H,
                       highlightthickness=0, bd=0)
        cv.grid(row=row, column=0, sticky="w", pady=1)
        x = 4

        def measure(text, bold=False):
            f = font_bold if bold else font_main
            return cv.tk.call("font", "measure", f, text)

        def draw_txt(text, bold=False):
            nonlocal x
            f = font_bold if bold else font_main
            cv.create_text(x, ROW_H // 2, text=text, anchor="w",
                           font=f, fill=self.text_dark)
            x += measure(text, bold) + 2

        def draw_box(text, bg, bold=False):
            nonlocal x
            f = font_bold if bold else font_main
            tw = measure(text, bold); pad = 4; box_w = tw + pad * 2
            cv.create_rectangle(x, PADY_BOX, x + box_w, ROW_H - PADY_BOX,
                                 fill=bg, outline="#888888", width=1)
            cv.create_text(x + pad, ROW_H // 2, text=text, anchor="w",
                           font=f, fill=self.text_dark)
            x += box_w + 3

        i = detail.row_index + 1
        draw_txt(f"{self._x_symbol(i, k_new)} = (")
        draw_box(self._fmt(detail.rhs), self.bg_rhs)
        draw_txt("  - (")

        for idx, term in enumerate(detail.terms):
            if idx > 0:
                draw_txt("  +  ")
            j, aij = term[0], term[1]
            draw_box(self._fmt(aij), self.bg_offdiag)
            draw_txt(" · ")
            draw_txt(self._term_x_symbol(term, j, k_old, k_new))

        draw_txt("))  /  ")
        draw_box(self._fmt(detail.diag), self.bg_diag)
        draw_txt("  =  ")
        draw_box(self._fmt(detail.new_value), self.bg_newx, bold=True)
        cv.configure(width=x + 4)

    def _term_x_symbol(self, term, j: int, k_old: int, k_new: int) -> str:
        """
        Hook: Jacobi verwendet immer k_old; Gauss-Seidel verwendet k_new
        wenn uses_new=True (5. Element im Tupel).
        """
        return self._x_symbol(j + 1, k_old)

    # ── Endergebnis (fast identisch) ──────────────────────────────────────
    def _append_final_solution_card(self):
        if not self.stepper:
            return
        x = self.stepper.x[:]
        k = self.stepper.iteration

        card = ttk.Frame(self.history.inner, padding=10, style="Card.TFrame")
        card.pack(fill="x", pady=8, padx=2)
        ttk.Label(card, text="LÖSUNG / NÄHERUNG", style="Header.TLabel",
                  padding=(8, 6)).pack(fill="x")

        body = ttk.Frame(card, style="Card.TFrame")
        body.pack(fill="x", pady=(10, 0))

        parts = [f"{self._x_symbol(i+1)} = {self._fmt(v)}"
                 for i, v in enumerate(x)]
        sol = "   |   ".join(parts)

        txt = tk.Text(body, height=3, wrap="word", relief="solid",
                      borderwidth=1, font=("Segoe UI", 10),
                      spacing1=0, spacing3=0)
        txt.pack(fill="x")
        txt.tag_configure("newx", background=self.bg_newx,
                          font=("Segoe UI", 10, "bold"))
        txt.insert("1.0", f"Endwert {self._x_symbol(None, k)}:\n")
        txt.insert("end", sol, ("newx",))
        txt.configure(state="disabled")

        messagebox.showinfo(self._solution_title, sol)

    # ── Reset (98% identisch) ─────────────────────────────────────────────
    def on_reset(self):
        self.stepper = None
        self.started = False
        self.btn_start.configure(text="Start")
        self._set_input_locked(False)
        self.history.clear()
        self.step_count = 0

        for row in self.entries_A:
            for e in row:
                e.delete(0, tk.END); e.insert(0, "0")
                e.configure(bg=self.bg_default)
        for e in self.entries_b:
            e.delete(0, tk.END); e.insert(0, "0")
        for e in self.entries_x0:
            e.delete(0, tk.END); e.insert(0, "0")
        self.tol_var.set("1e-6")

    # ── Start/Weiter (93% identisch) ──────────────────────────────────────
    def on_start_or_next(self):
        if self.started:
            self._do_one_step()
            return

        if self.step_count > 0:
            self.history.clear()
            self.step_count = 0
        self.stepper = None

        try:
            A, b, x0, tol = self._read_inputs()
        except ValueError as e:
            messagebox.showerror("Eingabefehler", str(e))
            return

        self.stepper = self._create_stepper(A, b, x0, tol)
        self.started = True
        self.btn_start.configure(text="Weiter")
        self._set_input_locked(True)
        self.after(0, self._do_one_step)

    # ── Einen Schritt (100% identisch) ────────────────────────────────────
    def _do_one_step(self):
        if not self.stepper:
            return
        step = self.stepper.next_step()
        self._append_step_card(self.stepper.A, self.stepper.b, step)
        if step.kind == "done":
            self._append_final_solution_card()
            self.btn_start.configure(state="disabled")

    # ── Abstrakte Methoden ────────────────────────────────────────────────
    def _create_stepper(self, A, b, x0, tol):
        raise NotImplementedError

    def _append_step_card(self, A, b, step):
        raise NotImplementedError