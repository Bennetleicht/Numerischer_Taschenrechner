from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, List
import sys
import os

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

from gauss_legendre_solver import Gauss_Legendre_Solver

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plotter')))
from integrations_plotter import GaussLegendrePlotter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'base_gui')))
from latex_renderer import render_formula_block, render_formula

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from GUI import maximize_and_lock

#  Sympy-Parser 
_XSYM = sp.Symbol("x")
_LOCALS = {
    "x": _XSYM,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
    "abs": sp.Abs, "pi": sp.pi, "E": sp.E,
}
_TRANS = standard_transformations + (implicit_multiplication_application, convert_xor)

def parse_function(expr_str: str):
    expr = parse_expr(expr_str, local_dict=_LOCALS, transformations=_TRANS)
    f    = sp.lambdify(_XSYM, expr, modules=["numpy"])
    return expr, f


#  Method-Klasse 
class GaussLegendreMethod:
    title = "Gauß-Legendre-Quadratur"

    def __init__(self):
        self.solver    = Gauss_Legendre_Solver()
        self.fx_str    = ""
        self.fx_latex  = ""
        self.f_sym     = None

    def on_start(self, values, parsers, plotter):
        fx = values["fx"].strip()
        if not fx:
            raise ValueError("Bitte f(x) eingeben.")
        sym_expr, f = parsers["parse_function"](fx)

        try:
            a = float(values["a"].replace(",", "."))
            b = float(values["b"].replace(",", "."))
            n = int(values["n"])
        except Exception:
            raise ValueError("a, b müssen Zahlen sein und n eine ganze Zahl (1–5).")

        self.solver.start(f, a, b, n)
        self.fx_str   = fx
        self.fx_latex = sp.latex(sym_expr)
        self.f_sym    = sym_expr

        plotter.set_function(f)
        width = b - a
        pad   = max(5.0, 3.0 * width)
        plotter.set_view(a - pad, b + pad)
        plotter.refresh()

        return "Start ok.", None, False

    def on_step(self, plotter):
        status, row, done = self.solver.step()
        if row is not None:
            a, b, n, xi, nodes, fxi, wi, I = row
            plotter.set_nodes(a, b, nodes, fxi)
            plotter.refresh()
        return status, row, done


#  Scrollbarer LaTeX-Frame 
class LatexScrollFrame(tk.Frame):

    def __init__(self, parent, bg="#f6f7fb", **kw):
        super().__init__(parent, bg=bg, **kw)
        self.bg = bg
        self._images: List = []
        self._scroll_enabled = False

        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        self.sb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.sb.set)

        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner = tk.Frame(self.canvas, bg=bg)
        self._window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Nur scrollen, wenn Maus über diesem Bereich ist
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)

    def _on_inner_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self._update_scrollbar()

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self._window, width=event.width)
        self._update_scrollbar()

    def _update_scrollbar(self):
        self.update_idletasks()

        inner_height = self.inner.winfo_reqheight()
        canvas_height = self.canvas.winfo_height()

        if inner_height > canvas_height:
            if not self._scroll_enabled:
                self.sb.pack(side="right", fill="y")
                self._scroll_enabled = True
        else:
            if self._scroll_enabled:
                self.sb.pack_forget()
                self._scroll_enabled = False
            self.canvas.yview_moveto(0)

    def _on_mousewheel(self, event):
        if not self._scroll_enabled:
            return
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def clear(self):
        for widget in self.inner.winfo_children():
            widget.destroy()
        self._images.clear()
        self.after(10, self._update_scrollbar)

    def add_heading(self, text: str):
        lbl = tk.Label(
            self.inner, text=text,
            font=("Segoe UI", 13, "bold"),
            fg="#1a6b9e", bg=self.bg,
            anchor="w", justify="left",
        )
        lbl.pack(fill="x", padx=10, pady=(12, 2))
        self.after(10, self._update_scrollbar)

    def add_latex(self, latex: str, fontsize: int = 13):
        img = render_formula(latex, bg=self.bg, fontsize=fontsize)
        self._images.append(img)
        lbl = tk.Label(self.inner, image=img, bg=self.bg, anchor="w")
        lbl.pack(fill="x", padx=10, pady=2)
        self.after(10, self._update_scrollbar)

    def add_latex_block(self, lines: list, fontsize: int = 13):
        img = render_formula_block(
            [(line, fontsize) for line in lines],
            bg=self.bg,
        )
        self._images.append(img)
        lbl = tk.Label(self.inner, image=img, bg=self.bg, anchor="w")
        lbl.pack(fill="x", padx=10, pady=2)
        self.after(10, self._update_scrollbar)

    def add_text(self, text: str, fontsize: int = 12, mono: bool = False):
        font = ("Consolas", fontsize) if mono else ("Segoe UI", fontsize)
        lbl = tk.Label(
            self.inner, text=text,
            font=font, fg="black", bg=self.bg,
            anchor="w", justify="left",
        )
        lbl.pack(fill="x", padx=14, pady=1)
        self.after(10, self._update_scrollbar)

    def add_table(self, headers: list, rows: list):
        """Tabelle als gleichmaessig ausgerichtete Zeilen mit Consolas."""
        frame = tk.Frame(self.inner, bg=self.bg)
        frame.pack(fill="x", padx=14, pady=4)

        col_w = 14
        header_cells = [f"{h:^{col_w}}" for h in headers]
        header_line = " | ".join(header_cells)

        tk.Label(
            frame,
            text=header_line,
            font=("Consolas", 11, "bold"),
            fg="#1a6b9e",
            bg=self.bg,
            anchor="w",
            justify="left",
        ).pack(fill="x")

        sep = "-+-".join(["-" * col_w for _ in headers])
        tk.Label(
            frame,
            text=sep,
            font=("Consolas", 11),
            fg="#888",
            bg=self.bg,
            anchor="w",
            justify="left",
        ).pack(fill="x")

        for row in rows:
            cells = [f"{str(v):^{col_w}}" for v in row]
            row_line = " | ".join(cells)
            tk.Label(
                frame,
                text=row_line,
                font=("Consolas", 11),
                fg="black",
                bg=self.bg,
                anchor="w",
                justify="left",
            ).pack(fill="x")

        self.after(10, self._update_scrollbar)

    def add_result(self, latex: str):
        img = render_formula(latex, bg=self.bg, fontsize=20)
        self._images.append(img)
        lbl = tk.Label(self.inner, image=img, bg=self.bg, anchor="w")
        lbl.pack(fill="x", padx=10, pady=(6, 2))
        self.after(10, self._update_scrollbar)

    def scroll_top(self):
        self.canvas.yview_moveto(0)


#  GUI 
class GaussLegendreGUI(tk.Tk):

    BG = "#f6f7fb"

    def __init__(self):
        super().__init__()
        self.method = GaussLegendreMethod()
        self.input_widgets: Dict[str, ttk.Entry] = {}
        self._calc_row  = None
        self._calc_step = 0

        self.title(self.method.title)

        maximize_and_lock(self)

        self._build_ui()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.configure(bg=self.BG)

        top = ttk.Frame(self, padding=15)
        top.grid(row=0, column=0, sticky="ew")

        style = ttk.Style()
        style.configure("TLabel",  font=("Segoe UI", 12))
        style.configure("TEntry",  font=("Segoe UI", 12))
        style.configure("TButton", font=("Segoe UI", 12))

        fields = [
            ("fx", "f(x) =", "", 36),
            ("a",  "a =",    "", 10),
            ("b",  "b =",    "", 10),
        ]
        self.vars: Dict[str, tk.StringVar] = {}
        col = 0
        for key, label, default, width in fields:
            ttk.Label(top, text=label).grid(row=0, column=col, sticky="w")
            col += 1
            v = tk.StringVar(value=str(default))
            self.vars[key] = v
            entry = ttk.Entry(top, textvariable=v, width=width)
            entry.grid(row=0, column=col, sticky="w", padx=(5, 15))
            self.input_widgets[key] = entry
            col += 1

        ttk.Label(top, text="n =").grid(row=0, column=col, sticky="w")
        col += 1
        self.n_var = tk.StringVar(value="2")
        self.vars["n"] = self.n_var
        spinbox = ttk.Spinbox(top, from_=1, to=5, textvariable=self.n_var, width=5)
        spinbox.grid(row=0, column=col, sticky="w", padx=(5, 15))
        self.input_widgets["n"] = spinbox
        col += 1

        self.plot_btn = ttk.Button(top, text="Plotten", command=self._on_plotten)
        self.plot_btn.grid(row=0, column=col, sticky="w", padx=(0, 10))
        col += 1

        self.calc_btn = ttk.Button(top, text="Berechnung", command=self._on_berechnung)
        self.calc_btn.grid(row=0, column=col, sticky="w", padx=(0, 10))
        col += 1

        self.reset_btn = ttk.Button(top, text="Reset", command=self._reset)
        self.reset_btn.grid(row=0, column=col, sticky="w")

        main = ttk.Frame(self, padding=(10, 0, 10, 10))
        main.grid(row=1, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        #  Linke Seite 
        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)

        ttk.Label(left, text="Berechnung",
                  font=("Segoe UI", 14, "bold")).grid(row=0, column=0, sticky="w")

        self.latex_frame = LatexScrollFrame(left, bg=self.BG)
        self.latex_frame.grid(row=1, column=0, sticky="nsew")

        self._show_initial_hint()

        #  Rechte Seite 
        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        ttk.Label(right, text="Plot",
                  font=("Segoe UI", 14, "bold")).grid(row=0, column=0, sticky="w", pady=(30, 0))

        self.plotter = GaussLegendrePlotter(right)
        self.plotter.widget().grid(row=1, column=0, sticky="nsew", pady=(5, 0))

    #  Anzeige-Hilfsmethoden 
    def _show_initial_hint(self):
        lf = self.latex_frame
        lf.clear()
        lf.add_text("Funktion eingeben und Plotten / Berechnung drücken.", fontsize=12)

    def _show_step(self):
        row = self._calc_row
        if row is None:
            return
        a, b, n, xi, nodes, fxi, wi, I = row
        fx_latex = self.method.fx_latex
        h   = 0.5 * (b - a)
        mid = 0.5 * (a + b)
        summe = float(np.dot(wi, fxi))

        lf = self.latex_frame
        lf.clear()

        # Verfahren (immer sichtbar)
        lf.add_heading("Verfahren")
        lf.add_latex_block([
            r"$Q_n = \sum_{i=1}^{n} w_i \, f(x_i)$",
            rf"$n = {n}, \quad \text{{Genauigkeitsgrad: }} {2*n-1}$",
        ], fontsize=13)

        # Schritt 1: Integral
        if self._calc_step >= 1:
            lf.add_heading("Integral")
            lf.add_latex(
                rf"$\int_{{{a:.6g}}}^{{{b:.6g}}} {fx_latex} \, dx$",
                fontsize=16,
            )

        # Schritt 2: Transformation
        if self._calc_step >= 2:
            lf.add_heading("1. Transformation")
            lf.add_latex_block([
                r"$t(x) = \frac{b-a}{2} \cdot x + \frac{a+b}{2}$",
                rf"$t(x) = {h:.6g} \cdot x + {mid:.6g}$",
                rf"$\int_{{{a:.6g}}}^{{{b:.6g}}} {fx_latex} \, dt = {h:.6g} \cdot \int_{{-1}}^{{1}} f({h:.6g} x + {mid:.6g}) \, dx$",
            ], fontsize=13)

        # Schritt 3: Auswertung
        if self._calc_step >= 3:
            lf.add_heading("2. Auswertung der Quadraturformel")
            lf.add_latex(
                r"$Q_n = \frac{b-a}{2} \cdot \sum_{i=1}^{n} w_i \cdot f(t_i)$",
                fontsize=14,
            )
            headers = ["i", "xi in [-1,1]", "ti = t(xi)", "wi", "f(ti)"]
            table_rows = [
                (i+1, f"{x:.8f}", f"{t:.8f}", f"{w:.8f}", f"{ft:.8f}")
                for i, (x, t, w, ft) in enumerate(zip(xi, nodes, wi, fxi))
            ]
            lf.add_table(headers, table_rows)
            lf.add_latex_block([
                rf"$\sum w_i \cdot f(t_i) = {summe:.8f}$",
                rf"$I = {h:.6g} \cdot {summe:.8f}$",
            ], fontsize=13)

        # Schritt 4: Ergebnis
        if self._calc_step >= 4:
            lf.add_heading("Ergebnis")
            lf.add_result(rf"$I \approx {I:.8g}$")

        lf.scroll_top()

    #  Hilfsmethoden 
    def _read_values(self) -> Dict[str, str]:
        return {k: v.get() for k, v in self.vars.items()}

    def _lock_all(self):
        for widget in self.input_widgets.values():
            widget.config(state="disabled")

    def _unlock_all(self):
        for widget in self.input_widgets.values():
            widget.config(state="normal")
        self.plot_btn.config(state="normal")
        self.calc_btn.config(state="normal")

    #  Button-Handler 
    def _on_plotten(self):
        try:
            values = self._read_values()
            _, f = parse_function(values["fx"].strip())
            a = float(values["a"].replace(",", "."))
            b = float(values["b"].replace(",", "."))
            n = int(values["n"])
        except Exception as e:
            messagebox.showerror("Fehler", str(e))
            return

        try:
            self.method.solver.start(f, a, b, n)
        except ValueError as e:
            messagebox.showerror("Fehler", str(e))
            return

        self._lock_all()
        self.plot_btn.config(state="disabled")

        _, row, _ = self.method.solver.step()

        self.plotter.set_function(f)
        width = b - a
        pad = max(5.0, 3.0 * width)
        self.plotter.set_view(a - pad, b + pad)

        if row is not None:
            _, _, _, _, nodes, fxi, _, _ = row
            self.plotter.set_nodes(a, b, nodes, fxi)

        self.plotter.refresh()

    def _on_berechnung(self):
        if self._calc_step == 0:
            try:
                values = self._read_values()

                class _DummyPlotter:
                    def set_function(self, f): pass
                    def set_view(self, *a, **kw): pass
                    def refresh(self): pass
                    def set_nodes(self, *a, **kw): pass

                self.method.fx_str   = ""
                self.method.fx_latex = ""
                self.method.f_sym    = None
                self.method.on_start(
                    values,
                    parsers={"parse_function": parse_function},
                    plotter=_DummyPlotter(),
                )
                _, row, _ = self.method.on_step(plotter=_DummyPlotter())
            except Exception as e:
                messagebox.showerror("Fehler", str(e))
                return

            self._lock_all()
            self._calc_row  = row
            self._calc_step = 1
            self.calc_btn.config(state="normal", text="Weiter")
        else:
            self._calc_step += 1
            if self._calc_step >= 4:
                self.calc_btn.config(state="disabled")

        self._show_step()

    def _reset(self):
        self._unlock_all()
        self._calc_row  = None
        self._calc_step = 0
        self.calc_btn.config(text="Berechnung")
        self._show_initial_hint()
        self.plotter.clear()



def main():
    app = GaussLegendreGUI()
    app.mainloop()


if __name__ == "__main__":
    main()