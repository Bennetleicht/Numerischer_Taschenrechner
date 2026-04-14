from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict
import numpy as np
import sys
import os

from latex_scroll_frame import LatexScrollFrame
from shared_compare import SharedCompare

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DGL')))
import dgl_common as common

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plotter')))
from dgl_plotter import DGL_Plotter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from GUI import maximize_and_lock


class Base_DGL_GUI(tk.Tk):
    BG = "#f6f7fb"

    def __init__(self, initial_method: str | None = None):
        super().__init__()
        self.minsize(1100, 650)

        maximize_and_lock(self)

        self.configure(bg=self.BG)

        self.vars: Dict[str, tk.StringVar] = {}
        self.input_widgets: Dict[str, tk.Widget] = {}
        self.method_buttons: Dict[str, tk.Button] = {}

        methods = self._get_methods()
        default_method = self._get_default_method()
        self.current_method = initial_method if initial_method in methods else default_method

        self.current_order = None
        self.order_box = None
        self.solver = None
        self.started = False

        self._compare_signatures = set()

        self._shared: SharedCompare | None = None

        self._build_ui()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        style = ttk.Style()
        style.configure("TLabel", font=("Segoe UI", 12))
        style.configure("TEntry", font=("Segoe UI", 12))
        style.configure("TButton", font=("Segoe UI", 12))

        top = ttk.Frame(self, padding=15)
        top.grid(row=0, column=0, sticky="ew")

        button_frame = ttk.Frame(top)
        button_frame.grid(row=0, column=0, columnspan=20, sticky="w", pady=(0, 12))

        for i, method_name in enumerate(self._get_methods()):
            btn = tk.Button(
                button_frame,
                text=method_name,
                width=30,
                height=2,
                font=("Consolas", 11),
                command=lambda m=method_name: self._set_method(m)
            )
            btn.grid(row=0, column=i, padx=2, pady=2)
            self.method_buttons[method_name] = btn

        self._update_method_buttons()

        fields = [
            ("y'(t) =", "f", " ", 30),
            ("a =", "a", " ", 8),
            ("b =", "b", " ", 8),
            ("y(a) =", "y0", " ", 8),
            ("h =", "h", " ", 8),
        ]

        col = 0
        for label, key, default, width in fields:
            ttk.Label(top, text=label).grid(row=1, column=col, sticky="w")
            col += 1

            v = tk.StringVar(value=default)
            self.vars[key] = v

            entry = ttk.Entry(top, textvariable=v, width=width)
            entry.grid(row=1, column=col, sticky="w", padx=(5, 15))
            self.input_widgets[key] = entry
            col += 1

        if self._has_order_selector():
            order_values = self._get_order_values()
            self.current_order = tk.StringVar(value=str(order_values[0]))

            ttk.Label(top, text="Ordnung k =").grid(row=1, column=col, sticky="w")
            col += 1

            self.order_box = ttk.Combobox(
                top,
                textvariable=self.current_order,
                values=[str(v) for v in order_values],
                width=5,
                state="readonly"
            )
            self.order_box.grid(row=1, column=col, sticky="w", padx=(5, 15))
            self.order_box.bind("<<ComboboxSelected>>", lambda _e: self._show_formula())
            col += 1
        else:
            self.current_order = None
            self.order_box = None

        self.step_btn = ttk.Button(top, text="Start", command=self._on_step)
        self.step_btn.grid(row=1, column=col, sticky="w", padx=(10, 8))
        col += 1

        self.reset_btn = ttk.Button(top, text="Reset", command=self._reset)
        self.reset_btn.grid(row=1, column=col, sticky="w", padx=(0, 8))
        col += 1

        self.compare_btn = ttk.Button(top, text="Vergleichen", command=self._on_compare)
        self.compare_btn.grid(row=1, column=col, sticky="w", padx=(0, 8))
        col += 1

        main = ttk.Frame(self, padding=(10, 0, 10, 10))
        main.grid(row=1, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)
        ttk.Label(left, text="Berechnung", font=("Segoe UI", 16, "bold")).grid(row=0, column=0, sticky="w")

        self.latex_frame = LatexScrollFrame(left, bg=self.BG)
        self.latex_frame.grid(row=1, column=0, sticky="nsew")

        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        ttk.Label(right, text="Plot", font=("Segoe UI", 16, "bold")).grid(row=0, column=0, sticky="w", pady=(30, 0))

        self.plotter = DGL_Plotter(right, xlabel="t", ylabel="y(t)")
        self.plotter.widget().grid(row=1, column=0, sticky="nsew", pady=(5, 0))

        self._show_formula()

    def _get_shared_compare(self) -> SharedCompare:
        if self._shared is None:
            self._shared = SharedCompare(self)
        return self._shared

    def _set_method(self, method_name: str):
        if self.started:
            return
        self.current_method = method_name
        self._update_method_buttons()
        self._show_formula()

    def _update_method_buttons(self):
        for name, btn in self.method_buttons.items():
            btn.config(
                bg="lightblue" if name == self.current_method else "lightgray",
                relief="sunken" if name == self.current_method else "raised"
            )

    def _lock_all(self):
        for w in self.input_widgets.values():
            w.config(state="disabled")
        for btn in self.method_buttons.values():
            btn.config(state="disabled")
        if self.order_box is not None:
            self.order_box.config(state="disabled")

    def _unlock_all(self):
        for w in self.input_widgets.values():
            w.config(state="normal")
        for btn in self.method_buttons.values():
            btn.config(state="normal")
        if self.order_box is not None:
            self.order_box.config(state="readonly")

    def _read_inputs(self):
        f_str = self.vars["f"].get().strip()
        if not f_str:
            raise ValueError("Bitte y'(t) eingeben.")

        a = float(self.vars["a"].get().replace(",", "."))
        b = float(self.vars["b"].get().replace(",", "."))
        y0 = float(self.vars["y0"].get().replace(",", "."))
        h = float(self.vars["h"].get().replace(",", "."))

        if b <= a:
            raise ValueError("Es muss gelten: a < b")
        if h <= 0:
            raise ValueError("Schrittweite h muss positiv sein.")

        f_expr, f = common.parse_function(f_str)
        return f_str, f_expr, f, a, b, y0, h

    def _initialize_solver(self):
        _, f_expr, f, a, b, y0, h = self._read_inputs()
        self.solver = self._create_solver(f_expr, f, a, b, y0, h)

        self.started = True
        self._lock_all()
        self.plotter.clear()

        order = self._get_order()
        if order is None:
            self.plotter.set_title(f"{self.current_method}")
        else:
            self.plotter.set_title(f"{self.current_method} (k={order})")

        self.plotter.update_solution(self.solver.ts, self.solver.ys)

    def _on_step(self):
        try:
            if not self.started:
                self._initialize_solver()
            if self.solver is None:
                return

            result = self.solver.step()
            self._format_step_output(result)
            self.plotter.update_solution(self.solver.ts, self.solver.ys)

            if self.solver.is_finished:
                self.step_btn.config(text="Fertig", state="disabled")
                messagebox.showinfo("Fertig", "Intervall fertig berechnet.")
            else:
                self.step_btn.config(text="Weiter")

        except Exception as e:
            messagebox.showerror("Fehler", str(e))

    def _reset(self):
        self.started = False
        self.solver = None
        self.step_btn.config(text="Start", state="normal")
        self._unlock_all()
        self.plotter.clear()
        self._show_formula()

    def _build_shared_compare_signature(self, solver):
        base_signature = self._build_compare_signature(solver)
        if not isinstance(base_signature, tuple):
            base_signature = (base_signature,)
        return (self.__class__.__name__,) + base_signature


    def _on_compare(self):
        try:
            full_solver = self._compute_full_solution_for_compare()
            if full_solver is None or len(full_solver.ts) == 0:
                messagebox.showwarning("Vergleich", "Es gibt keine berechnete Lösung zum Vergleichen.")
                return

            signature = self._build_shared_compare_signature(full_solver)

            source_name = "Einschritt" if "Einschritt" in self.__class__.__name__ else "Mehrschritt"
            label = f"[{source_name}] {self._build_compare_label(full_solver)}"

            shared = self._get_shared_compare()
            shared.add(
                signature,
                label,
                list(full_solver.ts),
                list(full_solver.ys)
            )

        except Exception as e:
            messagebox.showerror("Fehler", str(e))

    def _compute_full_solution_for_compare(self):
        if self.started and self.solver is not None:
            f = self.solver.f
            f_expr = self.solver.f_expr
            a = self.solver.a
            b = self.solver.b
            y0 = self.solver.y0
            h = self.solver.h
        else:
            _, f_expr, f, a, b, y0, h = self._read_inputs()

        temp_solver = self._create_solver(f_expr, f, a, b, y0, h)
        while not temp_solver.is_finished:
            temp_solver.step()
        return temp_solver


    def _get_methods(self):
        raise NotImplementedError

    def _get_default_method(self):
        raise NotImplementedError

    def _has_order_selector(self):
        return False

    def _get_order_values(self):
        return []

    def _get_order(self):
        return None

    def _create_solver(self, f_expr, f, a, b, y0, h):
        raise NotImplementedError

    def _build_compare_label(self, solver):
        raise NotImplementedError

    def _build_compare_signature(self, solver):
        raise NotImplementedError

    def _show_formula(self):
        raise NotImplementedError

    def _format_step_output(self, result):
        raise NotImplementedError