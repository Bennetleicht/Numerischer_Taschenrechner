from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Any, Dict, Tuple

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

from plotters_mpl import ABPlotter, NewtonPlotter, RegulaFalsiPlotter, SecantPlotter, HeronPlotter

# shared sympy parsing
# allows parsing expressions like "2x + sin(x)" without needing to write "2*x + sin(x)"
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
    f = sp.lambdify(_XSYM, expr, modules=["numpy"])
    return expr, f

def parse_function_with_derivative(expr_str: str):
    expr, f = parse_function(expr_str)
    dexpr = sp.diff(expr, _XSYM)
    df = sp.lambdify(_XSYM, dexpr, modules=["numpy"])
    return expr, f, dexpr, df

# Generic GUI für die verschiedenen Verfahren. 
# Es wird eine Methode übergeben, die die spezifischen Details (Eingabefelder, Tabellenstruktur, Plotverhalten) definiert.
class GenericMethodGUI(tk.Tk):

    def __init__(self, method):
        super().__init__()
        self.method = method
        self.started = False
        self.input_widgets = {}

        self.title(method.title)
        self.minsize(1000, 600)
        self.state("zoomed")
        try:
            self.attributes("-zoomed", True)
        except tk.TclError:
            pass

        self._build_ui()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, padding=10)
        top.grid(row=0, column=0, sticky="ew")

        self.vars: Dict[str, tk.StringVar] = {}
        col = 0
        for key, label, default, width in self.method.input_fields():
            ttk.Label(top, text=label).grid(row=0, column=col, sticky="w")
            col += 1
            v = tk.StringVar(value=str(default))
            self.vars[key] = v

            entry = ttk.Entry(top, textvariable=v, width=width)
            entry.grid(row=0, column=col, sticky="w", padx=(5, 15))

            self.input_widgets[key] = entry
            col += 1

        self.step_btn = ttk.Button(top, text="Start", command=self._on_step_button)
        self.step_btn.grid(row=0, column=col, sticky="w", padx=(0, 10))
        col += 1

        self.reset_btn = ttk.Button(top, text="Reset", command=self._reset)
        self.reset_btn.grid(row=0, column=col, sticky="w")

        main = ttk.Frame(self, padding=(10, 0, 10, 10))
        main.grid(row=1, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(2, weight=1)

        ttk.Label(left, text="Iteration / Werte", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")

        self.status_var = tk.StringVar(value="Bereit. Start drücken.")
        ttk.Label(left, textvariable=self.status_var).grid(row=1, column=0, sticky="w", pady=(4, 8))

        style = ttk.Style()
        style.configure("Big.Treeview", font=("Consolas", 12))
        style.configure("Big.Treeview.Heading", font=("Consolas", 14))

        cols = self.method.table_columns()
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=18, style="Big.Treeview")
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, anchor="e", width=120)
        self.tree.column(cols[0], width=55, anchor="center")
        self.tree.grid(row=2, column=0, sticky="nsew")

        sb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.grid(row=2, column=1, sticky="ns")

        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        ttk.Label(right, text="Plot", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")

        kind = self.method.plotter_kind
        if kind == "ab":
            self.plotter = ABPlotter(right)
            self.plotter.clear_ab()
        elif kind == "newton":
            self.plotter = NewtonPlotter(right)
            self.plotter.clear_state()
        elif kind == "regula":
            self.plotter = RegulaFalsiPlotter(right)
            self.plotter.clear_state()
        elif kind == "secant":
            self.plotter = SecantPlotter(right)
            self.plotter.clear_state()
        elif kind == "heron":
            self.plotter = HeronPlotter(right)
        else:
            raise ValueError(f"Unknown plotter_kind: {kind}")

        self.plotter.widget().grid(row=1, column=0, sticky="nsew", pady=(5, 0))

    # Hilfsfunktionen zum Verwalten der Tabelle, der Eingabefelder und des Plots
    def _clear_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

    # Sperrt oder entsperrt die Eingabefelder, um zu verhindern, dass der Benutzer während des laufenden Verfahrens Änderungen vornimmt
    def _set_inputs_locked(self, locked: bool):
        state = "disabled" if locked else "normal"
        for widget in self.input_widgets.values():
           widget.config(state=state)

    # Fügt eine neue Zeile zur Tabelle hinzu und scrollt automatisch zum Ende, um die neueste Iteration sichtbar zu machen
    def _append_row(self, row: Tuple[Any, ...]):
        self.tree.insert("", "end", values=row)
        kids = self.tree.get_children()
        if kids:
            self.tree.see(kids[-1])

    # Liest die aktuellen Werte aus den Eingabefeldern aus und gibt sie als Dictionary zurück, das an die Methode übergeben werden kann
    def _read_values(self) -> Dict[str, str]:
        return {k: v.get() for k, v in self.vars.items()}

    # Handler für den Start/Weiter-Button. 
    # Je nachdem, ob das Verfahren bereits gestartet wurde oder nicht, wird entweder die Start-Logik oder die Schritt-Logik ausgeführt.
    def _on_step_button(self):
        if not self.started:
            self._start()
        else:
            self._step()

    # Reset-Handler, der den Zustand der GUI zurücksetzt.
    def _reset(self):
        self.started = False
        self.step_btn.config(text="Start")
        self.status_var.set("Bereit. Start drücken.")
        self._clear_table()
        self._set_inputs_locked(False)

        try:
            kind = self.method.plotter_kind
            if kind == "ab":
                self.plotter.clear_ab()
            elif kind in ("newton", "regula", "secant"):
                self.plotter.clear_state()
        except Exception:
            pass

    def _start(self):
        self._clear_table()
        self.status_var.set("Starte...")
        self._set_inputs_locked(True)

        try:
            values = self._read_values()
            status, first_row, done = self.method.on_start(
                values,
                parsers={
                    "parse_function": parse_function,
                    "parse_function_with_derivative": parse_function_with_derivative,
                },
                plotter=self.plotter,
            )
        except Exception as e:
            messagebox.showerror("Fehler", str(e))
            self.status_var.set("Fehler. Start erneut versuchen.")
            self.started = False
            self.step_btn.config(text="Start")
            return

        self.status_var.set(status)

        if first_row is not None:
            self._append_row(first_row)

        if done:
            self.started = False
            self.step_btn.config(text="Start")
        else:
            self.started = True
            self.step_btn.config(text="Weiter")

    # Handler für den Weiter-Button, der einen Schritt im Verfahren ausführt.
    def _step(self):
        try:
            status, row, done = self.method.on_step(plotter=self.plotter)
        except Exception as e:
            messagebox.showerror("Fehler", str(e))
            self.status_var.set("Fehler in Schritt.")
            self.started = False
            self.step_btn.config(text="Start")
            return

        self.status_var.set(status)

        if row is not None:
            self._append_row(row)

        if done:
            self.started = False
            self.step_btn.config(text="Start")
        else:
            self.started = True
            self.step_btn.config(text="Weiter")