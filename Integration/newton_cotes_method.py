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

from newton_cotes_solver import Newton_Cotes_Solver

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Nullstellen')))
from plotters_mpl import NewtonCotesPlotter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'base_gui')))
from latex_renderer import render_formula_block, render_formula


verfahren: str = "Trapezregel"
modus: str = "Einzelstreifen"

STRIP_COLORS = [
    "#2563eb", "#16a34a", "#dc2626", "#d97706", "#7c3aed",
    "#0891b2", "#be185d", "#65a30d", "#ea580c", "#4f46e5",
]

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


FORMELN_LATEX = {
    "Trapezregel": [
        r"Einfache Trapezregel:",
        r"$I \approx \frac{b-a}{2} \left( f(a) + f(b) \right)$",
        r"Zusammengesetzte Trapezregel:",
        r"$h = \frac{b-a}{m}$",
        r"$I \approx h \left( \frac{1}{2}f(x_0) + \sum_{i=1}^{m-1} f(x_i) + \frac{1}{2}f(x_m) \right)$",
    ],
    "Simpsonregel": [
        r"Einfache Simpsonregel:",
        r"$I \approx \frac{b-a}{6} \left( f(a) + 4 f(m) + f(b) \right)$",
        r"Zusammengesetzte Simpsonregel (m gerade Streifen):",
        r"$h = \frac{b-a}{m}$",
        r"$I \approx \frac{h}{3} \left( f(x_0) + f(x_m) + 4\sum_{i=0}^{m/2-1} f(x_{2i+1}) + 2\sum_{i=1}^{m/2-1} f(x_{2i}) \right)$",
    ],
    "3/8-Regel": [
        r"$I \approx \frac{b-a}{8} \left( f(a) + 3f(m_1) + 3f(m_2) + f(b) \right)$",
        r"$m_1 = a + \frac{b-a}{3}, \quad m_2 = a + \frac{2(b-a)}{3}$",
    ],
    "Milne-Regel": [
        r"$I \approx \frac{b-a}{90} \left( 7f(a) + 32f(m_1) + 12f(m_2) + 32f(m_3) + 7f(b) \right)$",
        r"$m_1 = a+\frac{b-a}{4},\quad m_2 = a+\frac{b-a}{2},\quad m_3 = a+\frac{3(b-a)}{4}$",
    ],
}

#  Schritt-Definitionen 
MAX_STEP = 3


def compute_strip_data(row, f, verfahren_name: str) -> list:
    if row is None or f is None:
        return []

    a, b, I, m, xs_nodes, ys_nodes = row
    strips = []

    if verfahren_name == "Trapezregel":
        h = (b - a) / m
        for i in range(m):
            xi = xs_nodes[i]
            xi1 = xs_nodes[i + 1]
            fxi = float(ys_nodes[i])
            fxi1 = float(ys_nodes[i + 1])
            I_i = h * (fxi + fxi1) / 2
            strips.append({
                "label": f"Streifen {i+1}",
                "x_left": xi,
                "x_right": xi1,
                "f_left": fxi,
                "f_right": fxi1,
                "I_part": I_i,
                "color": STRIP_COLORS[i % len(STRIP_COLORS)],
            })

    elif verfahren_name == "Simpsonregel":
        pairs = m // 2
        for i in range(pairs):
            x0 = xs_nodes[2 * i]
            x1 = xs_nodes[2 * i + 1]
            x2 = xs_nodes[2 * i + 2]
            y0 = float(ys_nodes[2 * i])
            y1 = float(ys_nodes[2 * i + 1])
            y2 = float(ys_nodes[2 * i + 2])
            I_i = (x2 - x0) / 6 * (y0 + 4 * y1 + y2)
            strips.append({
                "label": f"Doppelstreifen {i+1}",
                "x_left": x0,
                "x_right": x2,
                "x_mid": x1,
                "f_left": y0,
                "f_mid": y1,
                "f_right": y2,
                "I_part": I_i,
                "color": STRIP_COLORS[i % len(STRIP_COLORS)],
            })

    elif verfahren_name == "3/8-Regel":
        fa = float(ys_nodes[0])
        fm1 = float(ys_nodes[1])
        fm2 = float(ys_nodes[2])
        fb = float(ys_nodes[3])
        strips.append({
            "label": "Einzelstreifen",
            "x_left": a,
            "x_right": b,
            "f_left": fa,
            "f_right": fb,
            "I_part": I,
            "color": STRIP_COLORS[0],
        })

    elif verfahren_name == "Milne-Regel":
        strips.append({
            "label": "Einzelstreifen",
            "x_left": a,
            "x_right": b,
            "f_left": float(ys_nodes[0]),
            "f_right": float(ys_nodes[-1]),
            "I_part": I,
            "color": STRIP_COLORS[0],
        })

    return strips

#  Latex-Formel-Definitionen
def build_rechnung_latex(row, f_sym, f, verfahren_name: str) -> list:
    if row is None or f_sym is None:
        return []

    a, b, I, m, xs_nodes, ys_nodes = row
    lines = []

    if verfahren_name == "Trapezregel":
        h = (b - a) / m
        fa = float(ys_nodes[0])
        fb = float(ys_nodes[-1])
        if m == 1:
            lines.append(rf"$f(a) = f({a:.4g}) = {fa:.6g}$")
            lines.append(rf"$f(b) = f({b:.4g}) = {fb:.6g}$")
            lines.append(rf"$\frac{{b-a}}{{2}} = {(b-a)/2:.6g}$")
            lines.append(rf"$I = {(b-a)/2:.6g} \cdot ({fa:.6g} + {fb:.6g})$")
        else:
            inner_sum = float(np.sum(ys_nodes[1:-1]))
            lines.append(rf"$m = {m}, \quad h = \frac{{{b:.4g}-{a:.4g}}}{{{m}}} = {h:.6g}$")
            lines.append(rf"$x_0={xs_nodes[0]:.4g},\ x_1={xs_nodes[1]:.4g},\ \ldots,\ x_{{{m}}}={xs_nodes[-1]:.4g}$")
            lines.append(rf"$\sum_{{i=1}}^{{{m-1}}} f(x_i) = {inner_sum:.6g}$")
            lines.append(rf"$I = {h:.6g} \cdot \left(\frac{{1}}{{2}} \cdot {fa:.6g} + {inner_sum:.6g} + \frac{{1}}{{2}} \cdot {fb:.6g}\right)$")

    elif verfahren_name == "Simpsonregel":
        h = (b - a) / m
        fa = float(ys_nodes[0])
        fb = float(ys_nodes[-1])
        if m == 2:
            mid = xs_nodes[1]
            fm = float(ys_nodes[1])
            lines.append(rf"$m = \frac{{a+b}}{{2}} = {mid:.6g}$")
            lines.append(rf"$f(a)={fa:.6g},\ f(m)={fm:.6g},\ f(b)={fb:.6g}$")
            lines.append(rf"$\frac{{b-a}}{{6}} = {(b-a)/6:.6g}$")
            lines.append(rf"$I = {(b-a)/6:.6g} \cdot ({fa:.6g} + 4 \cdot {fm:.6g} + {fb:.6g})$")
        else:
            sum_odd = float(np.sum(ys_nodes[1:-1:2]))
            sum_even = float(np.sum(ys_nodes[2:-2:2]))
            lines.append(rf"$m = {m}\ (\text{{gerade}}), \quad h = {h:.6g}$")
            lines.append(rf"$x_0={xs_nodes[0]:.4g},\ldots,x_{{{m}}}={xs_nodes[-1]:.4g}$")
            lines.append(rf"$\sum f(x_{{2i+1}}) = {sum_odd:.6g}\ \text{{(ungerade Indizes)}}$")
            lines.append(rf"$\sum f(x_{{2i}}) = {sum_even:.6g}\ \text{{(gerade Indizes)}}$")
            lines.append(rf"$I = \frac{{{h:.6g}}}{{3}} \cdot ({fa:.6g} + {fb:.6g} + 4 \cdot {sum_odd:.6g} + 2 \cdot {sum_even:.6g})$")

    elif verfahren_name == "3/8-Regel":
        fa = float(ys_nodes[0])
        fm1 = float(ys_nodes[1])
        fm2 = float(ys_nodes[2])
        fb = float(ys_nodes[3])
        h8 = (b - a) / 8
        lines.append(rf"$m_1 = {xs_nodes[1]:.6g}, \quad m_2 = {xs_nodes[2]:.6g}$")
        lines.append(rf"$f(a)={fa:.6g},\ f(m_1)={fm1:.6g},\ f(m_2)={fm2:.6g},\ f(b)={fb:.6g}$")
        lines.append(rf"$\frac{{b-a}}{{8}} = {h8:.6g}$")
        lines.append(rf"$I = {h8:.6g} \cdot ({fa:.6g} + 3\cdot{fm1:.6g} + 3\cdot{fm2:.6g} + {fb:.6g})$")

    elif verfahren_name == "Milne-Regel":
        fa = float(ys_nodes[0])
        fm1 = float(ys_nodes[1])
        fm2 = float(ys_nodes[2])
        fm3 = float(ys_nodes[3])
        fb = float(ys_nodes[4])
        h90 = (b - a) / 90
        lines.append(rf"$m_1={xs_nodes[1]:.6g},\ m_2={xs_nodes[2]:.6g},\ m_3={xs_nodes[3]:.6g}$")
        lines.append(rf"$f(a)={fa:.6g},\ f(m_1)={fm1:.6g},\ f(m_2)={fm2:.6g}$")
        lines.append(rf"$f(m_3)={fm3:.6g},\ f(b)={fb:.6g}$")
        lines.append(rf"$\frac{{b-a}}{{90}} = {h90:.6g}$")
        lines.append(rf"$I = {h90:.6g}\cdot(7\cdot{fa:.6g}+32\cdot{fm1:.6g}+12\cdot{fm2:.6g}+32\cdot{fm3:.6g}+7\cdot{fb:.6g})$")

    return lines


class NewtonCotesMethod:
    title = "Newton-Cotes-Verfahren"

    def __init__(self):
        self.solver = Newton_Cotes_Solver(verfahren, modus)
        self.fx_str = ""
        self.fx_latex = ""
        self.a_val = None
        self.b_val = None
        self.m_val = 1
        self.f_sym = None

    def input_fields(self):
        return [
            ("fx", "f(x) =", "", 36),
            ("a", "a =", "", 10),
            ("b", "b =", "", 10),
            ("m", "m =", "1", 6),
            ("tol", "Tol =", "0", 10),
        ]

    def on_start(self, values, parsers):
        fx = values["fx"].strip()
        if not fx:
            raise ValueError("Bitte f(x) eingeben.")

        sym_expr, f = parsers["parse_function"](fx)

        try:
            a = float(values["a"].replace(",", "."))
            b = float(values["b"].replace(",", "."))
            m = int(float(values.get("m", "1").replace(",", ".")))
            tol = float(values["tol"].replace(",", "."))
        except Exception:
            raise ValueError("a, b, m, Tol müssen Zahlen sein.")

        if m < 1:
            raise ValueError("m muss >= 1 sein.")

        _m = m
        if verfahren == "Simpsonregel" and m % 2 != 0:
            _m = m + 1

        self.solver = Newton_Cotes_Solver(verfahren, modus)
        self.solver.start(f, a, b, tol, m=_m)

        self.fx_str = fx
        self.fx_latex = sp.latex(sym_expr)
        self.f_sym = sym_expr
        self.a_val = a
        self.b_val = b
        self.m_val = _m

        return "Start ok.", None, False

    def on_step(self):
        return self.solver.step()


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
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)

    def _on_inner_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self._update_scrollbar()

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self._window, width=event.width)
        self._update_scrollbar()

    def _update_scrollbar(self):
        self.update_idletasks()
        inner_h = self.inner.winfo_reqheight()
        canvas_h = self.canvas.winfo_height()

        if inner_h > canvas_h:
            if not self._scroll_enabled:
                self.sb.pack(side="right", fill="y")
                self._scroll_enabled = True
        else:
            if self._scroll_enabled:
                self.sb.pack_forget()
                self._scroll_enabled = False
            self.canvas.yview_moveto(0)

    def _on_mousewheel(self, event):
        if self._scroll_enabled:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def clear(self):
        for w in self.inner.winfo_children():
            w.destroy()
        self._images.clear()
        self.after(10, self._update_scrollbar)

    def add_heading(self, text: str):
        lbl = tk.Label(
            self.inner,
            text=text,
            font=("Segoe UI", 13, "bold"),
            fg="#1a6b9e",
            bg=self.bg,
            anchor="w",
            justify="left"
        )
        lbl.pack(fill="x", padx=10, pady=(12, 2))
        self.after(10, self._update_scrollbar)

    def add_latex(self, latex: str, fontsize: int = 13):
        img = render_formula(latex, bg=self.bg, fontsize=fontsize)
        self._images.append(img)
        tk.Label(self.inner, image=img, bg=self.bg, anchor="w").pack(fill="x", padx=10, pady=2)
        self.after(10, self._update_scrollbar)

    def add_latex_block(self, lines: list, fontsize: int = 13):
        img = render_formula_block([(line, fontsize) for line in lines], bg=self.bg)
        self._images.append(img)
        tk.Label(self.inner, image=img, bg=self.bg, anchor="w").pack(fill="x", padx=10, pady=2)
        self.after(10, self._update_scrollbar)

    def add_text(self, text: str, fontsize: int = 12, mono: bool = False):
        font = ("Consolas", fontsize) if mono else ("Segoe UI", fontsize)
        tk.Label(
            self.inner,
            text=text,
            font=font,
            fg="black",
            bg=self.bg,
            anchor="w",
            justify="left"
        ).pack(fill="x", padx=14, pady=1)
        self.after(10, self._update_scrollbar)

    def add_result(self, latex: str):
        img = render_formula(latex, bg=self.bg, fontsize=20)
        self._images.append(img)
        tk.Label(self.inner, image=img, bg=self.bg, anchor="w").pack(fill="x", padx=10, pady=(6, 2))
        self.after(10, self._update_scrollbar)

    def add_strip_table(self, strips: list, verfahren_name: str):
        if not strips:
            return

        frame = tk.Frame(self.inner, bg=self.bg)
        frame.pack(fill="x", padx=10, pady=(4, 8))

        if verfahren_name == "Trapezregel":
            headers = ["", "x_i", "x_{i+1}", "f(x_i)", "f(x_{i+1})", "Teilintegral"]
        elif verfahren_name == "Simpsonregel":
            headers = ["", "x_{2i}", "Mitte", "x_{2i+2}", "f links", "f Mitte", "f rechts", "Teilintegral"]
        else:
            headers = ["", "a", "b", "Teilintegral"]

        header_font = ("Segoe UI", 10, "bold")
        cell_font = ("Consolas", 10)

        for col, h in enumerate(headers):
            tk.Label(
                frame,
                text=h,
                font=header_font,
                bg="#dbeafe",
                fg="#1e3a8a",
                relief="flat",
                borderwidth=1,
                padx=6,
                pady=3,
                anchor="center"
            ).grid(row=0, column=col, sticky="nsew", padx=1, pady=1)

        for row_i, s in enumerate(strips):
            row_bg = self.bg

            color_box = tk.Label(frame, bg=s["color"], text="  ", width=2, relief="flat", padx=4, pady=3)
            color_box.grid(row=row_i + 1, column=0, sticky="nsew", padx=1, pady=1)

            def _cell(text, col):
                tk.Label(
                    frame,
                    text=text,
                    font=cell_font,
                    bg=row_bg,
                    fg="#111827",
                    padx=6,
                    pady=3,
                    anchor="e",
                    relief="flat",
                    borderwidth=1
                ).grid(row=row_i + 1, column=col, sticky="nsew", padx=1, pady=1)

            if verfahren_name == "Trapezregel":
                _cell(f"{s['x_left']:.5g}", 1)
                _cell(f"{s['x_right']:.5g}", 2)
                _cell(f"{s['f_left']:.5g}", 3)
                _cell(f"{s['f_right']:.5g}", 4)
                _cell(f"{s['I_part']:.6g}", 5)

            elif verfahren_name == "Simpsonregel":
                _cell(f"{s['x_left']:.5g}", 1)
                _cell(f"{s['x_mid']:.5g}", 2)
                _cell(f"{s['x_right']:.5g}", 3)
                _cell(f"{s['f_left']:.5g}", 4)
                _cell(f"{s['f_mid']:.5g}", 5)
                _cell(f"{s['f_right']:.5g}", 6)
                _cell(f"{s['I_part']:.6g}", 7)

            else:
                _cell(f"{s['x_left']:.5g}", 1)
                _cell(f"{s['x_right']:.5g}", 2)
                _cell(f"{s['I_part']:.6g}", 3)

        for col in range(len(headers)):
            frame.columnconfigure(col, weight=1)

        self.after(10, self._update_scrollbar)

    def scroll_top(self):
        self.canvas.yview_moveto(0)


class NewtonCotesGUI(tk.Tk):
    BG = "#f6f7fb"

    def __init__(self):
        super().__init__()
        self.method = NewtonCotesMethod()
        self.input_widgets: Dict[str, ttk.Entry] = {}
        self.modus_buttons: Dict[str, tk.Button] = {}
        self._calc_row = None
        self._calc_step = 0
        self._plot_ready = False

        self.title(self.method.title)
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
        self.configure(bg=self.BG)

        top = ttk.Frame(self, padding=15)
        top.grid(row=0, column=0, sticky="ew")

        style = ttk.Style()
        style.configure("TLabel", font=("Segoe UI", 12))
        style.configure("TEntry", font=("Segoe UI", 12))
        style.configure("TButton", font=("Segoe UI", 12))

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

        self.info_var = tk.StringVar(value="")
        ttk.Label(
            top,
            textvariable=self.info_var,
            font=("Segoe UI", 10, "italic"),
            foreground="#b45309"
        ).grid(row=1, column=0, columnspan=col, sticky="w", pady=(2, 0))

        self.plot_btn = ttk.Button(top, text="Plotten", command=self._on_plotten)
        self.plot_btn.grid(row=0, column=col, sticky="w", padx=(0, 10))
        col += 1

        self.calc_btn = ttk.Button(top, text="Berechnung", command=self._on_berechnung)
        self.calc_btn.grid(row=0, column=col, sticky="w", padx=(0, 10))
        col += 1

        self.reset_btn = ttk.Button(top, text="Reset", command=self._reset)
        self.reset_btn.grid(row=0, column=col, sticky="w")

        self.vars["m"].trace_add("write", lambda *_: self._update_m_info())

        main = ttk.Frame(self, padding=(10, 0, 10, 10))
        main.grid(row=1, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)

        ttk.Label(left, text="Berechnung", font=("Segoe UI", 14, "bold")).grid(row=0, column=0, sticky="w")

        self.latex_frame = LatexScrollFrame(left, bg=self.BG)
        self.latex_frame.grid(row=1, column=0, sticky="nsew")
        self._show_formel()

        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        ttk.Label(right, text="Plot", font=("Segoe UI", 14, "bold")).grid(row=0, column=0, sticky="w", pady=(30, 0))

        self.plotter = NewtonCotesPlotter(right)

        button_frame = ttk.Frame(right, padding=(0, 5))
        button_frame.grid(row=0, column=0, sticky="ne")

        self.verfahren_buttons: Dict[str, tk.Button] = {}
        for i, vname in enumerate(["Trapezregel", "Simpsonregel", "3/8-Regel", "Milne-Regel"]):
            btn = tk.Button(
                button_frame,
                text=vname,
                width=18,
                height=2,
                font=("Consolas", 12),
                command=lambda vn=vname: self._set_verfahren(vn),
            )
            btn.grid(row=0, column=i, padx=2)
            self.verfahren_buttons[vname] = btn
        self._update_verfahren_buttons()

        self.plotter.widget().grid(row=1, column=0, sticky="nsew", pady=(5, 0))

    def _draw_plot(self, f, a, b, m, xs_nodes, ys_nodes, force_view=False):
        already_has_function = self.plotter.f is not None

        self.plotter.set_function(f)

        if force_view or not already_has_function:
            width = b - a
            pad = max(5.0, 3.0 * width)
            self.plotter.set_view(a - pad, b + pad)

        self.plotter.set_a_b(
            a, b,
            verfahren=verfahren,
            m=m,
            xs_nodes=xs_nodes,
            ys_nodes=ys_nodes,
        )
        self.plotter.refresh()
        self._plot_ready = True

    def _update_m_info(self):
        try:
            m = int(float(self.vars["m"].get()))
        except Exception:
            self.info_var.set("")
            return

        if verfahren == "Simpsonregel" and m % 2 != 0:
            self.info_var.set(f"Simpsonregel: m muss gerade sein → wird auf m={m+1} aufgerundet.")
        elif verfahren == "Simpsonregel":
            self.info_var.set(f"Simpsonregel: {m//2} Doppelstreifen")
        elif verfahren == "Trapezregel":
            self.info_var.set(f"Trapezregel: {m} Teilstreifen")
        else:
            self.info_var.set("")

    def _show_formel(self):
        lf = self.latex_frame
        lf.clear()
        lf.add_heading(verfahren)
        lf.add_latex_block(FORMELN_LATEX.get(verfahren, []), fontsize=13)
        lf.add_text("Funktion eingeben und Berechnung drücken.", fontsize=11)

    def _show_step(self):
        row = self._calc_row
        if row is None:
            return

        a, b, I, m, xs_nodes, ys_nodes = row
        fx_latex = self.method.fx_latex

        lf = self.latex_frame
        lf.clear()

        #  Verfahren + Grundformeln 
        lf.add_heading(verfahren)
        for line in FORMELN_LATEX.get(verfahren, []):
            line = line.strip()
            if line.startswith("$") and line.endswith("$"):
                lf.add_latex(line, fontsize=13)
            else:
                lf.add_text(line, fontsize=12)

        #  Schritt 1: Integral-Übersicht + Teilstreifen 
        if self._calc_step >= 1:
            lf.add_heading("Integral")
            lf.add_latex(
                rf"$\int_{{{a:.6g}}}^{{{b:.6g}}} {fx_latex} \, dx$",
                fontsize=16,
            )

            if m > 1 and verfahren in ("Trapezregel", "Simpsonregel"):
                suffix = "Doppelstreifen" if verfahren == "Simpsonregel" else "Teilstreifen"
                lf.add_text(f"m = {m} {suffix}", fontsize=11)

            strips = compute_strip_data(row, self.method.solver.f, verfahren)
            if strips and m > 1:
                lf.add_heading("Teilstreifen")
                lf.add_strip_table(strips, verfahren)

        #  Schritt 2: Gesamtrechnung 
        if self._calc_step >= 2:
            lf.add_heading("Gesamtrechnung")
            rechnung_lines = build_rechnung_latex(
                row, self.method.f_sym, self.method.solver.f, verfahren
            )
            for line in rechnung_lines:
                line = line.strip()
                if line.startswith("$") and line.endswith("$"):
                    lf.add_latex(line, fontsize=13)
                else:
                    lf.add_text(line, fontsize=12)

        #  Schritt 3: Ergebnis 
        if self._calc_step >= 3:
            lf.add_heading("Ergebnis")
            lf.add_result(rf"$I \approx {I:.8g}$")

        lf.scroll_top()

    def _update_calc_button(self):
        """Button-Beschriftung und -Zustand nach aktuellem Schritt setzen."""
        if self._calc_step == 0:
            self.calc_btn.config(text="Berechnung", state="normal")
        elif self._calc_step < MAX_STEP:
            self.calc_btn.config(text="Weiter", state="normal")
        else:
            self.calc_btn.config(text="Weiter", state="disabled")

    def _read_values(self) -> Dict[str, str]:
        return {k: v.get() for k, v in self.vars.items()}

    def _lock_all(self):
        for w in self.input_widgets.values():
            w.config(state="disabled")
        for btn in self.verfahren_buttons.values():
            btn.config(state="disabled")
        for btn in self.modus_buttons.values():
            btn.config(state="disabled")

    def _unlock_all(self):
        for w in self.input_widgets.values():
            w.config(state="normal")
        for btn in self.verfahren_buttons.values():
            btn.config(state="normal")
        for btn in self.modus_buttons.values():
            btn.config(state="normal")
        self.plot_btn.config(state="normal")
        self.calc_btn.config(state="normal")

    def _on_plotten(self):
        try:
            values = self._read_values()
            _, f = parse_function(values["fx"].strip())
            a = float(values["a"].replace(",", "."))
            b = float(values["b"].replace(",", "."))
            m = int(float(values.get("m", "1").replace(",", ".")))
        except Exception as e:
            messagebox.showerror("Fehler", str(e))
            return

        if not (a < b):
            messagebox.showerror("Fehler", "Es muss gelten: a < b")
            return
        if m < 1:
            messagebox.showerror("Fehler", "m muss >= 1 sein.")
            return

        if verfahren == "Simpsonregel" and m % 2 != 0:
            m += 1

        xs_nodes, ys_nodes = NewtonCotesPlotter._compute_nodes(a, b, f, verfahren, m)

        self._lock_all()
        self.plot_btn.config(state="disabled")

        self._draw_plot(f, a, b, m, xs_nodes, ys_nodes, force_view=True)

    def _on_berechnung(self):
        if self._calc_step == 0:
            try:
                values = self._read_values()
                self.method.on_start(
                    values,
                    parsers={"parse_function": parse_function},
                )
                status, row, done = self.method.on_step()
            except Exception as e:
                messagebox.showerror("Fehler", str(e))
                return

            if row is None:
                messagebox.showerror("Fehler", "Keine Berechnungsdaten erzeugt.")
                return

            a, b, I, m, xs_nodes, ys_nodes = row
            f = self.method.solver.f

            if self.plotter.f is None:
                self.plotter.set_function(f)
                width = b - a
                pad = max(5.0, 3.0 * width)
                self.plotter.set_view(a - pad, b + pad)

            self.plotter.set_a_b(
                a, b,
                verfahren=verfahren,
                m=m,
                xs_nodes=xs_nodes,
                ys_nodes=ys_nodes,
            )
            self.plotter.refresh()
            self._plot_ready = True

            self._calc_row = row
            self._calc_step = 1

            self._lock_all()
            self._update_calc_button()
            self._show_step()
            return

        if self._calc_row is None:
            messagebox.showerror("Fehler", "Keine Berechnung vorhanden.")
            return

        if self._calc_step < MAX_STEP:
            self._calc_step += 1

        self._update_calc_button()
        self._show_step()

    def _reset(self):
        self._unlock_all()
        self._calc_row = None
        self._calc_step = 0
        self._plot_ready = False
        self._update_calc_button()
        self.info_var.set("")
        self._show_formel()
        self.plotter.clear_ab()
        self.plotter.set_function(None)
        self.plotter.refresh()

    def _set_verfahren(self, name: str):
        global verfahren
        verfahren = name
        self._update_verfahren_buttons()
        self._update_m_info()
        self._show_formel()

        if name in ("3/8-Regel", "Milne-Regel"):
            self.vars["m"].set("1")
            self.input_widgets["m"].config(state="disabled")
        else:
            self.input_widgets["m"].config(state="normal")

    def _update_verfahren_buttons(self):
        for name, btn in self.verfahren_buttons.items():
            btn.config(
                bg="lightblue" if name == verfahren else "lightgray",
                relief="sunken" if name == verfahren else "raised",
            )


def main():
    app = NewtonCotesGUI()
    app.mainloop()


if __name__ == "__main__":
    main()