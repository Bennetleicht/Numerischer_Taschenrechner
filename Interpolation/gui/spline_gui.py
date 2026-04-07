from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from matplotlib.lines import Line2D
from io import BytesIO
import matplotlib.figure as mfigure

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application
)

from spline_method import SplineMethod
from base_gui.gui_utils import ScrollableFrame
from gui.base_interp_gui import (BaseInterpGUI, render_formula_block,
                                  render_matrix_img)


# Rendering                                 

def render_symbolic_tridiag(boundary: str, bg: str = "#f6f7fb",
                             dpi: int = 100) -> ImageTk.PhotoImage:
    # Zeichnet die allgemeine Tridiagonalmatrix symbolisch
    import matplotlib.pyplot as plt

    if boundary == "natural":
        rows = [
            [r"$1$",             r"$0$",              r"",                 r"",                 r"$0$"],
            [r"$\frac{h}{6}$",  r"$\frac{2h}{3}$",  r"$\frac{h}{6}$",  r"",                 r""],
            [r"",                r"$\ddots$",         r"$\ddots$",        r"$\ddots$",        r""],
            [r"",                r"",                 r"$\frac{h}{6}$",   r"$\frac{2h}{3}$",  r"$\frac{h}{6}$"],
            [r"$0$",             r"",                 r"",                 r"$0$",              r"$1$"],
        ]
        # M0=Mn=0 direkt eingesetzt
        mvec = [r"$0$", r"$M_1$", r"$\vdots$", r"$M_{n-1}$", r"$0$"]
        rhs  = [r"$0$",
                r"$\frac{y_2-y_1}{h}-\frac{y_1-y_0}{h}$",
                r"$\vdots$",
                r"$\frac{y_n-y_{n-1}}{h}-\frac{y_{n-1}-y_{n-2}}{h}$",
                r"$0$"]
    else:
        rows = [
            [r"$\alpha_0$",       r"$\beta_0$",             r"",                       r"",                         r"$0$"],
            [r"$\frac{h_0}{6}$",  r"$\frac{h_0+h_1}{3}$",   r"$\frac{h_1}{6}$",       r"",                         r""],
            [r"",                  r"$\ddots$",               r"$\ddots$",              r"$\ddots$",                r""],
            [r"",                  r"",                  r"$\frac{h_{n-1}}{6}$",   r"$\frac{h_{n-1}+h_n}{3}$",  r"$\frac{h_n}{6}$"],
            [r"$0$",               r"",                  r"",                       r"$\beta_n$",                r"$\alpha_n$"],
        ]
        mvec = [r"$M_0$", r"$M_1$", r"$\vdots$", r"$M_{n-1}$", r"$M_n$"]
        rhs  = [r"$b_0$",
                r"$\frac{y_2-y_1}{h_1}-\frac{y_1-y_0}{h_0}$",
                r"$\vdots$",
                r"$\frac{y_n-y_{n-1}}{h_{n-1}}-\frac{y_{n-1}-y_{n-2}}{h_{n-2}}$",
                r"$b_n$"]

    n  = len(rows); nc = len(rows[0])
    cell_w = 0.78; cell_h = 0.30
    pad_l  = 0.26; eq_gap = 0.22
    mvec_w = 0.82; rhs_w  = 3.1
    fig_w = pad_l + nc*cell_w + eq_gap + mvec_w + eq_gap + rhs_w + 0.28
    fig_h = (n + 0.55) * cell_h

    fig = mfigure.Figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w); ax.set_ylim(0, fig_h)
    ax.set_axis_off(); ax.set_facecolor(bg)

    def cx(j): return pad_l + (j + 0.5) * cell_w
    def ry(i): return fig_h - (i + 0.95) * cell_h
    by_top = fig_h - 0.42 * cell_h
    by_bot = fig_h - (n + 0.42) * cell_h
    mid_y  = (by_top + by_bot) / 2
    lw = 1.6

    def bracket(xl, xr, yt, yb, open_=True):
        d = 0.055
        xs_ = [xl+d, xl, xl, xl+d] if open_ else [xr-d, xr, xr, xr-d]
        ax.plot(xs_, [yt, yt, yb, yb], color="#374151", lw=lw, solid_capstyle="round")

    bx_l = pad_l - 0.055; bx_r = pad_l + nc * cell_w + 0.055
    bracket(bx_l, bx_r, by_top, by_bot, True)
    bracket(bx_l, bx_r, by_top, by_bot, False)
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            if cell:
                ax.text(cx(j), ry(i), cell, ha="center", va="center",
                        fontsize=11, color="#1e293b")

    dot_x = bx_r + eq_gap * 0.35
    ax.text(dot_x, mid_y, r"$\cdot$", ha="center", va="center", fontsize=13, color="#374151")

    mvl = dot_x + eq_gap * 0.55; mvr = mvl + mvec_w
    bracket(mvl, mvr, by_top, by_bot, True)
    bracket(mvl, mvr, by_top, by_bot, False)
    mvc = (mvl + mvr) / 2
    for i, lbl in enumerate(mvec):
        ax.text(mvc, ry(i), lbl, ha="center", va="center", fontsize=10, color="#1e293b")

    eq_x = mvr + eq_gap * 0.35
    ax.text(eq_x, mid_y, r"$=$", ha="center", va="center", fontsize=13, color="#374151")

    fvl = eq_x + eq_gap * 0.55; fvr = fvl + rhs_w
    bracket(fvl, fvr, by_top, by_bot, True)
    bracket(fvl, fvr, by_top, by_bot, False)
    fvc = (fvl + fvr) / 2
    for i, lbl in enumerate(rhs):
        ax.text(fvc, ry(i), lbl, ha="center", va="center", fontsize=9, color="#1e293b")


    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=bg, pad_inches=0.04)
    plt.close(fig); buf.seek(0)
    return ImageTk.PhotoImage(Image.open(buf).convert("RGBA"))

def render_symbolic_system(boundary: str, bg: str = "#f6f7fb",
                            dpi: int = 100) -> ImageTk.PhotoImage:
    lines = [
        (r"$h = \frac{b-a}{n}$", 12),
        (r"$s_i(x) = \frac{1}{6h}\!\left((x_{i+1}-x)^3 M_i + (x-x_i)^3 M_{i+1}\right) + c_i(x-x_i) + d_i$", 11),
        (r"$d_i = y_i - \frac{h}{6}M_i, \qquad c_i = \frac{y_{i+1}-y_i}{h} - \frac{h}{6}(M_{i+1}-M_i)$", 11),
    ]
    if boundary == "natural":
        lines += [
            (r"$\mathbf{Nat\ddot{u}rliche\ RB:}\quad M_0=0,\quad M_n=0$", 11),
            (r"$\beta_0=\beta_n=b_0=b_n=0,\quad \alpha_0=\alpha_n=1$", 11),
        ]
    else:
        lines += [
            (r"$\mathbf{Hermite\ RB:}\quad s^{\prime}_0(x_0)=f^{\prime}(x_0),\quad s^{\prime}_{n-1}(x_n)=f^{\prime}(x_n)$", 11),
            (r"$\alpha_0=\frac{h}{3},\quad \beta_0=\frac{h}{6},\quad b_0=\frac{y_1-y_0}{h}-f^{\prime}(x_0)$", 11),
            (r"$\alpha_n=\frac{h}{3},\quad \beta_n=\frac{h}{6},\quad b_n=-\frac{y_n-y_{n-1}}{h}+f^{\prime}(x_n)$", 11),
        ]
    return render_formula_block(lines, bg=bg, dpi=dpi, width_in=8.0)

# Hauptfenster
class SplineGUI(BaseInterpGUI):

    def __init__(self):
        self.method = SplineMethod()
        super().__init__()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.configure(bg=self.bg_app)

        # Kopfzeile
        top = ttk.Frame(self, padding=(10, 8), style="App.TFrame")
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)

        ttk.Label(top, text="Kubische Spline-Interpolation",
                  style="Title.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

        # Zeile 1: Eingaben
        row1 = ttk.Frame(top, style="App.TFrame")
        row1.grid(row=1, column=0, sticky="w")

        ttk.Label(row1, text="f(x) =", style="Hint.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 4))
        self._func_var = tk.StringVar(value="1/(1+x**2)")
        self._func_entry = ttk.Entry(row1, textvariable=self._func_var, width=22)
        self._func_entry.grid(row=0, column=1, sticky="w", padx=(0, 14))

        ttk.Label(row1, text="x ∈ [", style="Hint.TLabel").grid(
            row=0, column=2, sticky="w")
        self._a_var = tk.StringVar(value="-5")
        ttk.Entry(row1, textvariable=self._a_var, width=6).grid(
            row=0, column=3, sticky="w", padx=(0, 2))
        self._a_entry = row1.winfo_children()[-1]
        ttk.Label(row1, text=",", style="Hint.TLabel").grid(row=0, column=4)
        self._b_var = tk.StringVar(value="5")
        ttk.Entry(row1, textvariable=self._b_var, width=6).grid(
            row=0, column=5, sticky="w", padx=(0, 2))
        self._b_entry = row1.winfo_children()[-1]
        ttk.Label(row1, text="]", style="Hint.TLabel").grid(
            row=0, column=6, sticky="w", padx=(0, 14))

        ttk.Label(row1, text="n+1 Punkte =", style="Hint.TLabel").grid(
            row=0, column=7, sticky="w", padx=(0, 4))
        self._n_var = tk.StringVar(value="7")
        self._n_entry = ttk.Entry(row1, textvariable=self._n_var, width=5)
        self._n_entry.grid(row=0, column=8, sticky="w", padx=(0, 14))

        btn_frame = ttk.Frame(top, style="App.TFrame")
        btn_frame.grid(row=1, column=1, sticky="e", padx=(10, 0))
        self._btn_step = ttk.Button(btn_frame, text="Start",
                                    command=self._on_step_btn)
        self._btn_step.pack(side="left", padx=(0, 6))
        ttk.Button(btn_frame, text="Reset",
                   command=self._on_reset).pack(side="left")

        # Zeile 2: Randbedingung + Knoten
        row2 = ttk.Frame(top, style="App.TFrame")
        row2.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

        col2 = 0
        ttk.Label(row2, text="Randbedingung:", style="Hint.TLabel").grid(
            row=0, column=col2, sticky="w", padx=(0, 6)); col2 += 1

        self._boundary_var = tk.StringVar(value="natural")
        self._btn_natural = tk.Button(
            row2, text="Natürlich", font=("Segoe UI", 10),
            relief="solid", bd=1, bg="#3b82f6", fg="white", cursor="hand2",
            command=lambda: self._set_boundary("natural"))
        self._btn_natural.grid(row=0, column=col2, padx=(0, 4)); col2 += 1

        self._btn_hermite_toggle = tk.Button(
            row2, text="Hermite", font=("Segoe UI", 10),
            relief="solid", bd=1, bg=self.bg_card, fg="#374151", cursor="hand2",
            command=lambda: self._set_boundary("hermite"))
        self._btn_hermite_toggle.grid(row=0, column=col2, padx=(0, 14)); col2 += 1

        ttk.Label(row2, text="f'(a) =", style="Hint.TLabel").grid(
            row=0, column=col2, sticky="w", padx=(0, 4)); col2 += 1
        self._df0_var = tk.StringVar(value="0")
        self._df0_entry = ttk.Entry(row2, textvariable=self._df0_var, width=7)
        self._df0_entry.grid(row=0, column=col2, sticky="w", padx=(0, 10)); col2 += 1
        ttk.Label(row2, text="f'(b) =", style="Hint.TLabel").grid(
            row=0, column=col2, sticky="w", padx=(0, 4)); col2 += 1
        self._dfn_var = tk.StringVar(value="0")
        self._dfn_entry = ttk.Entry(row2, textvariable=self._dfn_var, width=7)
        self._dfn_entry.grid(row=0, column=col2, sticky="w", padx=(0, 14)); col2 += 1

        self._hermite_cols = list(range(col2 - 4, col2))
        for c in self._hermite_cols:
            ws = row2.grid_slaves(row=0, column=c)
            for w in ws:
                w.grid_remove()

        # Hauptbereich
        main = ttk.Frame(self, padding=(10, 0, 10, 10), style="App.TFrame")
        main.grid(row=1, column=0, sticky="nsew")
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main, style="App.TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)

        # Header: allgemeines System als LaTeX
        hdr_frame = ttk.Frame(left, style="App.TFrame")
        hdr_frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(hdr_frame, text="Spline-System",
                  style="Hint.TLabel").pack(anchor="w")
        self._hdr_canvas = tk.Canvas(hdr_frame, bg=self.bg_app,
                                     highlightthickness=0, height=120)
        self._hdr_canvas.pack(fill="x")
        self._render_header()

        self._history = ScrollableFrame(left)
        self._history.grid(row=1, column=0, sticky="nsew")

        # Plot
        right = ttk.Frame(main, style="App.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        ttk.Label(right, text="Plot", style="Title.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 6))

        from Nullstellen.plotters_mpl import InterpolationPlotter

        self.plotter = InterpolationPlotter(right)
        self.plotter.widget().grid(row=1, column=0, sticky="nsew")
        self._draw_empty_plot()

    # Header
    def _render_header(self):
        boundary = self._boundary_var.get() if hasattr(self, "_boundary_var") \
                   else "natural"
        # Matrix-Bild
        mat_img  = render_symbolic_tridiag(boundary, bg=self.bg_app, dpi=100)
        # Formelzeilen darunter
        form_img = render_symbolic_system(boundary, bg=self.bg_app, dpi=100)

        self._imgs_hdr_mat  = mat_img
        self._imgs_hdr_form = form_img

        total_h = mat_img.height() + form_img.height() + 4
        self._hdr_canvas.configure(height=total_h)
        self._hdr_canvas.delete("all")
        self._hdr_canvas.create_image(0, 0, anchor="nw", image=mat_img)
        self._hdr_canvas.create_image(0, mat_img.height() + 4,
                                       anchor="nw", image=form_img)

    # Toggle   
    def _set_boundary(self, mode: str):
        if self.started:
            return
        self._boundary_var.set(mode)
        if mode == "natural":
            self._btn_natural.config(bg="#3b82f6", fg="white")
            self._btn_hermite_toggle.config(bg=self.bg_card, fg="#374151")
            for c in self._hermite_cols:
                for w in self._df0_entry.master.grid_slaves(row=0, column=c):
                    w.grid_remove()
        else:
            self._btn_hermite_toggle.config(bg="#3b82f6", fg="white")
            self._btn_natural.config(bg=self.bg_card, fg="#374151")
            for c in self._hermite_cols:
                for w in self._df0_entry.master.grid_slaves(row=0, column=c):
                    w.grid()
        self._render_header()


    # Plot
    def _draw_empty_plot(self):
        self.plotter.clear_plot()
        self.plotter.set_title("Kubischer Spline")

        ax = self.plotter.ax
        ax.set_facecolor(self.bg_app)
        ax.set_xlabel("x")
        ax.set_ylabel("s(x)")
        ax.grid(True, linestyle="--", alpha=0.5)

        ax.text(
            0.5, 0.5,
            "Funktion und Intervall eingeben\nund Start drücken",
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=11, color="#9ca3af"
        )

        self.plotter.points_x = []
        self.plotter.points_y = []
        self.plotter.curve_x = np.array([], dtype=float)
        self.plotter.curve_y = np.array([], dtype=float)
        self.plotter.curve_line.set_data([], [])
        self.plotter.scatter_pts.set_offsets(np.empty((0, 2)))
        self.plotter.scatter_pts.set_visible(False)

        self.plotter.refresh()
        self._set_dynamic_legend(
            show_points=True,
            show_polygon=False,
            show_spline=False,
            active_interval=None
        )
        self.plotter.canvas.draw_idle()

    def _draw_points_up_to(self, up_to: int):
        solver = self.method.solver
        x_all, f_all = solver.x, solver.f
        xs = x_all[:up_to + 1]
        fs = f_all[:up_to + 1]

        self.plotter.clear_plot()
        self.plotter.set_title("Kubischer Spline")
        self.plotter.set_points(xs, fs)

        ax = self.plotter.ax
        ax.set_facecolor(self.bg_app)
        ax.set_xlabel("x")
        ax.set_ylabel("s(x)")
        ax.grid(True, linestyle="--", alpha=0.4)

        if len(xs) >= 2:
            ax.plot(
                xs, fs,
                linestyle="--",
                linewidth=1.0,
                alpha=0.55,
                color="#6b7280",
                zorder=2
            )

        for xi, fi in zip(xs, fs):
            ax.annotate(
                f"({xi:.3f}, {fi:.3f})",
                (xi, fi),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7
            )

        self.plotter.curve_x = np.array([], dtype=float)
        self.plotter.curve_y = np.array([], dtype=float)
        self.plotter.curve_line.set_data([], [])

        if x_all:
            xmin, xmax = min(x_all), max(x_all)
            pad = max(0.3, (xmax - xmin) * 0.08)
            ax.set_xlim(xmin - pad, xmax + pad)

            ys = [y for y in f_all if np.isfinite(y)]
            if ys:
                ymin, ymax = min(ys), max(ys)
                ypad = max(0.3, (ymax - ymin) * 0.12 if ymax != ymin else 1.0)
                ax.set_ylim(ymin - ypad, ymax + ypad)

        self.plotter.refresh()
        self._set_dynamic_legend(
            show_points=True,
            show_polygon=(len(xs) >= 2),
            show_spline=False,
            active_interval=None
        )
        self.plotter.canvas.draw_idle()

    def _update_plot(self, highlight_interval: int | None = None,
                    build_up_to: int | None = None, **kwargs):
        solver = self.method.solver
        x, f = solver.x, solver.f

        if not x or len(x) < 2:
            self._draw_empty_plot()
            return

        base_color = "#2563eb"
        highlight_color = "#fde047"

        self.plotter.clear_plot()
        self.plotter.set_title("Kubischer Spline")
        self.plotter.set_points(x, f)

        ax = self.plotter.ax
        ax.set_facecolor(self.bg_app)
        ax.set_xlabel("x")
        ax.set_ylabel("s(x)")
        ax.grid(True, linestyle="--", alpha=0.4)

        # Stützpolygon
        ax.plot(
            x, f,
            linestyle="--",
            linewidth=1.0,
            alpha=0.55,
            color="#6b7280",
            zorder=2
        )

        for xi, fi in zip(x, f):
            ax.annotate(
                f"({xi:.3f}, {fi:.3f})",
                (xi, fi),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8
            )

        all_curve_x = []
        all_curve_y = []

        if build_up_to is not None:
            max_seg = min(build_up_to, len(x) - 2)

            for i in range(max_seg + 1):
                xa, xb = x[i], x[i + 1]
                seg_t = np.linspace(xa, xb, 220)
                seg_y = solver.evaluate_array(seg_t)

                all_curve_x.extend(seg_t.tolist())
                all_curve_y.extend(seg_y.tolist())

                if i == highlight_interval:
                    col = highlight_color
                    lw = 3.6
                    alpha = 1.0
                else:
                    col = base_color
                    lw = 2.0
                    alpha = 0.95

                ax.plot(
                    seg_t, seg_y,
                    color=col,
                    linewidth=lw,
                    alpha=alpha,
                    zorder=4
                )

        # plotter state aktualisiern
        if all_curve_x and all_curve_y:
            self.plotter.set_curve(all_curve_x, all_curve_y, label=None)
        else:
            self.plotter.curve_x = np.array([], dtype=float)
            self.plotter.curve_y = np.array([], dtype=float)
            self.plotter.curve_line.set_data([], [])

        self._apply_plot_limits(
            x, f,
            curve_x=all_curve_x if all_curve_x else None,
            curve_y=all_curve_y if all_curve_y else None
        )

        self.plotter.refresh()
        self._set_dynamic_legend(
            show_points=True,
            show_polygon=True,
            show_spline=(build_up_to is not None and build_up_to >= 0),
            active_interval=highlight_interval
        )
        self.plotter.canvas.draw_idle()

    def _set_dynamic_legend(
        self,
        show_points: bool = False,
        show_polygon: bool = False,
        show_spline: bool = False,
        active_interval: int | None = None
    ):
        ax = self.plotter.ax

        old_legend = ax.get_legend()
        if old_legend is not None:
            old_legend.remove()

        legend_artists = []

        if show_points:
            artist_points = Line2D(
                [0], [0],
                linestyle="None",
                marker="o",
                markersize=6,
                markerfacecolor="red",
                markeredgecolor="red",
                color="red",
                label="Stützpunkte"
            )
            legend_artists.append(artist_points)

        if show_polygon:
            artist_polygon = Line2D(
                [0], [0],
                linestyle="--",
                linewidth=1.2,
                color="#6b7280",
                label="Stützpolygon"
            )
            legend_artists.append(artist_polygon)

        if show_spline:
            artist_spline = Line2D(
                [0], [0],
                linestyle="-",
                linewidth=2.2,
                color="#2563eb",
                label="Spline"
            )
            legend_artists.append(artist_spline)

        if active_interval is not None:
            artist_active = Line2D(
                [0], [0],
                linestyle="-",
                linewidth=3.0,
                color="#fde047",
                label=rf"Aktives Segment $s_{{{active_interval}}}(x)$"
            )
            legend_artists.append(artist_active)

        if legend_artists:
            leg = ax.legend(
                handles=legend_artists,
                fontsize=8,
                loc="upper right",
                frameon=True,
                fancybox=True,
                framealpha=0.95,
                borderpad=0.6,
                handlelength=2.2,
                labelspacing=0.5
            )
            leg.get_frame().set_edgecolor("#d1d5db")
            leg.get_frame().set_linewidth(0.8)

    # LaTeX-Zeilen pro Schritt-Art                    
    def _fmt(self, v: float) -> str:
        # Formatiert float kompakt
        s = f"{v:.4f}".rstrip("0").rstrip(".")
        return s if s else "0"

    def _latex_setup(self) -> list[tuple[str, int]]:
        solver = self.method.solver
        n_intervals = len(solver.x) - 1
        a, b = solver.x[0], solver.x[-1]
        h = (b - a) / n_intervals

        func_ltx = self._expr_to_latex(self.method.func_str)

        return [
            (rf"$f(x) = {func_ltx}$", 12),
            (rf"$x \in [{a:.4g},\ {b:.4g}],\quad n = {n_intervals}$", 11),
            (rf"$h = \frac{{b-a}}{{n}} = \frac{{{b:.4g} - ({a:.4g})}}{{{n_intervals}}} = {h:.6f}$", 12),
            (r"$x_i = a + i \cdot h$", 10),
        ]
    
    def _expr_to_latex(self, expr: str) -> str:
        expr = expr.strip().replace("^", "**")

        transformations = standard_transformations + (
            implicit_multiplication_application,
        )

        local_dict = {
            "x": sp.Symbol("x"),
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "exp": sp.exp,
            "log": sp.log,
            "sqrt": sp.sqrt,
            "abs": sp.Abs,
            "pi": sp.pi,
            "e": sp.E,
        }

        try:
            parsed = parse_expr(expr, local_dict=local_dict, transformations=transformations)
            return sp.latex(parsed)
        except Exception:
            # Fallback, falls Parsing fehlschlägt
            return expr.replace("**", "^")

    def _latex_coeffs(self, i: int) -> list[tuple[str, int]]:
        solver = self.method.solver

        xi = solver.x[i]
        xi1 = solver.x[i + 1]
        hi = solver.h[i]

        ai = solver.a[i]
        bi = solver.b[i]
        ci = solver.c[i]
        di = solver.d[i]

        simplified = self._simplified_si(i)

        return [
            (rf"$\mathrm{{Segment}}\ [{xi:.4f},\ {xi1:.4f}]$", 11),
            (rf"$h_{{{i}}} = x_{{{i+1}}} - x_{{{i}}} = {xi1:.4f} - {xi:.4f} = {hi:.6f}$", 10),
            (rf"$a_{{{i}}} = y_{{{i}}} = {ai:.6f}$", 10),
            (rf"$b_{{{i}}} = {bi:.6f}$", 10),
            (rf"$c_{{{i}}} = \frac{{M_{{{i}}}}}{{2}} = \frac{{{solver.M[i]:.6f}}}{{2}} = {ci:.6f}$", 10),
            (rf"$d_{{{i}}} = \frac{{M_{{{i+1}}} - M_{{{i}}}}}{{6h_{{{i}}}}} = {di:.6f}$", 10),
            (rf"$s_{{{i}}}(x) = a_{{{i}}} + b_{{{i}}}(x-x_{{{i}}}) + c_{{{i}}}(x-x_{{{i}}})^2 + d_{{{i}}}(x-x_{{{i}}})^3$", 10),
            (rf"$= {ai:.6f} + {bi:.6f}(x-{xi:.4f}) + {ci:.6f}(x-{xi:.4f})^2 + {di:.6f}(x-{xi:.4f})^3$", 10),
            (rf"$\Rightarrow\ s_{{{i}}}(x) = {simplified}$", 10),
        ]
    
    def _latex_knot(self, i: int) -> list[tuple[str, int]]:
        solver = self.method.solver
        n_intervals = len(solver.x) - 1
        xi = solver.x[i]
        yi = solver.f[i]
        a = solver.x[0]
        h = (solver.x[-1] - solver.x[0]) / n_intervals

        return [
            (rf"$x_{{{i}}} = a + {i} \cdot h = {a:.4g} + {i} \cdot {h:.4f} = {xi:.6f}$", 11),
            (rf"$y_{{{i}}} = f(x_{{{i}}}) = f({xi:.6f}) = {yi:.6f}$", 11),
        ]

    def _latex_boundary(self) -> list[tuple[str, int]]:
        bnd = self.method.boundary
        solver = self.method.solver
        if bnd == "natural":
            return [
                (r"$\mathbf{Nat\ddot{u}rliche\ Randbedingung:}$", 12),
                (r"$s^{\prime\prime}_0(x_0) = 0, \qquad s^{\prime\prime}_{n-1}(x_n) = 0$", 11),
                (r"$M_0 = 0, \quad M_n = 0$", 11),
                (r"$\beta_0 = \beta_n = b_0 = b_n = 0$", 11),
                (r"$\alpha_0 = \alpha_n = 1$", 11),
            ]
        else:
            df0 = solver.df0; dfn = solver.dfn
            h0 = solver.h[0]; hn = solver.h[-1]
            y0, y1 = solver.f[0], solver.f[1]
            yn_1, yn = solver.f[-2], solver.f[-1]
            b0_val = (y1 - y0) / h0 - df0
            bn_val = -(yn - yn_1) / hn + dfn
            return [
                (r"$\mathbf{Hermite\ Randbedingung:}$", 12),
                (r"$s^{\prime}_0(x_0) = f^{\prime}(x_0), \qquad s^{\prime}_{n-1}(x_n) = f^{\prime}(x_n)$", 11),
                (rf"$f^{{\prime}}(x_0) = {df0},\qquad f^{{\prime}}(x_n) = {dfn}$", 11),
                (rf"$\alpha_0 = \frac{{h_0}}{{3}} = \frac{{{h0:.4f}}}{{3}} = {h0/3:.4f},\quad"
                 rf"\beta_0 = \frac{{h_0}}{{6}} = {h0/6:.4f}$", 11),
                (rf"$b_0 = \frac{{y_1-y_0}}{{h_0}} - f^{{\prime}}(x_0) = "
                 rf"\frac{{{y1:.4f}-{y0:.4f}}}{{{h0:.4f}}} - {df0} = {b0_val:.6f}$", 11),
                (rf"$\alpha_n = \frac{{h_{{n-1}}}}{{3}} = {hn/3:.4f},\quad"
                 rf"\beta_n = \frac{{h_{{n-1}}}}{{6}} = {hn/6:.4f}$", 11),
                (rf"$b_n = -\frac{{y_n-y_{{n-1}}}}{{h_{{n-1}}}} + f^{{\prime}}(x_n) = "
                 rf"-\frac{{{yn:.4f}-{yn_1:.4f}}}{{{hn:.4f}}} + {dfn} = {bn_val:.6f}$", 11),
            ]

    def _latex_system(self) -> list[tuple[str, int]]:
        bnd = self.method.boundary
        solver = self.method.solver
        x, f, h = solver.x, solver.f, solver.h
        n = len(x)
        A, rhs = self.method._A, self.method._rhs

        if bnd == "natural":
            lines = [
                (r"$\alpha_i = 2(h_{i-1}+h_i),\quad \beta_i = h_{i-1},\quad \gamma_i = h_i$", 11),
                (r"$d_i = 6\!\left(\frac{f_{i+1}-f_i}{h_i} - \frac{f_i-f_{i-1}}{h_{i-1}}\right),\quad i=1,\ldots,n-1$", 11),
                (r"$\mathrm{Randzeilen\ (aus\ RB):}\quad \alpha_0=\alpha_n=1,\quad \beta_0=\beta_n=0,\quad d_0=d_n=0$", 10),
            ]
            # Randzeile i=0: α=1, β=0, d=0  (M₀=0)
            lines.append((
                rf"$i=0:\quad \alpha_0 = 1,\quad \beta_0 = 0,\quad d_0 = 0\quad\Rightarrow M_0 = 0$", 10
            ))
            nr = A.shape[0]  # = n-2 (nur innere)
            for k in range(nr):
                i = k + 1
                ai_v = 2 * (h[k] + h[k+1])
                bi_v = h[k]
                gi_v = h[k+1]
                di_v = rhs[k] / 6.0  # /6 to match matrix scaling
                lines.append((
                    rf"$i={i}:\quad \alpha_{{{i}}} = {ai_v:.4f},\quad"
                    rf"\beta_{{{i}}} = {bi_v:.4f},\quad"
                    rf"\gamma_{{{i}}} = {gi_v:.4f},\quad"
                    rf"d_{{{i}}} = {di_v:.4f}$", 10
                ))
            # Randzeile i=n: α=1, β=0, d=0  (Mₙ=0)
            lines.append((
                rf"$i=n:\quad \alpha_n = 1,\quad \beta_n = 0,\quad d_n = 0\quad\Rightarrow M_n = 0$", 10
            ))
        else:
            # Hermite: Matrix hat Größe n×n
            # Zeile 0: α₀=h[0]/3, β₀=h[0]/6, rhs[0]=(y1-y0)/h0 - df0
            # Zeile k=1..n-2: αk=2*(h[k-1]+h[k]), βk=h[k-1], γk=h[k]
            # Zeile n-1: αn=h[-1]/3, βn=h[-1]/6
            df0, dfn = solver.df0, solver.dfn
            h0, hn = h[0], h[-1]
            lines = [
                (r"$\mathrm{Zeile\ 0\ (Hermite\ links):}$", 11),
                (rf"$\alpha_0 = \frac{{h_0}}{{3}} = {h0/3:.4f},\quad"
                 rf"\beta_0 = \frac{{h_0}}{{6}} = {h0/6:.4f},\quad"
                 rf"d_0 = {rhs[0]:.4f}$", 10),
            ]
            for k in range(1, n - 1):
                ai_v = 2*(h[k-1]+h[k]); bi_v = h[k-1]; gi_v = h[k]; di_v = rhs[k]
                lines.append((
                    rf"$i={k}:\quad \alpha_{{{k}}}={ai_v:.4f},\quad"
                    rf"\beta_{{{k}}}={bi_v:.4f},\quad"
                    rf"\gamma_{{{k}}}={gi_v:.4f},\quad"
                    rf"d_{{{k}}}={di_v:.4f}$", 10))
            lines.append((r"$\mathrm{Zeile\ n\ (Hermite\ rechts):}$", 11))
            lines.append((
                rf"$\alpha_n = \frac{{h_{{n-1}}}}{{3}} = {hn/3:.4f},\quad"
                rf"\beta_n = \frac{{h_{{n-1}}}}{{6}} = {hn/6:.4f},\quad"
                rf"d_n = {rhs[-1]:.4f}$", 10))
        return lines

    def _latex_moment(self) -> list[tuple[str, int]]:
        solver = self.method.solver
        lines = [
            (r"$\mathrm{L\ddot{o}sung\ via\ Gau\ss\text{-}Elimination:}\quad A \cdot M = d$", 11),
        ]
        for i, mi in enumerate(solver.M):
            lines.append((rf"$M_{{{i}}} = {mi:.6f}$", 10))
        return lines

    def _simplified_si(self, i: int) -> str:
        """Gibt die vereinfachte si(x)-Gleichung zurück."""
        solver = self.method.solver
        xi = solver.x[i]; xi1 = solver.x[i+1]
        hi = solver.h[i]
        Mi, Mi1 = solver.M[i], solver.M[i+1]
        fi, fi1 = solver.f[i], solver.f[i+1]
        ci = (fi1 - fi) / hi - hi / 6 * (Mi1 - Mi)
        di = fi - hi**2 / 6 * Mi
        bi = solver.b[i]
        ai = solver.a[i]
        di_coeff = solver.d[i]
        A3 = di_coeff
        A2 = ci - 3*xi*di_coeff
        A1 = bi - 2*xi*ci + 3*xi**2*di_coeff
        A0 = ai - xi*bi + xi**2*ci - xi**3*di_coeff
        parts = []
        for exp, coef in [(3, A3), (2, A2), (1, A1), (0, A0)]:
            if abs(coef) < 1e-10:
                continue
            c_str = f"{abs(coef):.4f}".rstrip("0").rstrip(".")
            if exp == 3:   term = rf"{c_str} x^3"
            elif exp == 2: term = rf"{c_str} x^2"
            elif exp == 1: term = rf"{c_str} x"
            else:          term = c_str
            if not parts:
                parts.append(f"-{term}" if coef < 0 else term)
            else:
                parts.append(rf" - {term}" if coef < 0 else rf" + {term}")
        return "".join(parts) if parts else "0"

    def _latex_all_segments(self) -> list[tuple[str, int]]:
        # Alle vereinfachten si(x) untereinander
        solver = self.method.solver
        n = len(solver.x) - 1
        lines = [(r"$\mathrm{\ddot{U}bersicht\ aller\ Segmente\ (vereinfacht):}$", 11)]
        for i in range(n):
            xi = solver.x[i]; xi1 = solver.x[i+1]
            simplified = self._simplified_si(i)
            lines.append((
                rf"$s_{{{i}}}(x) = {simplified},\quad x \in [{xi:.4f},\ {xi1:.4f}]$",
                10
            ))
        return lines

    def _latex_done(self) -> list[tuple[str, int]]:
        solver = self.method.solver
        n = len(solver.x) - 1
        bnd_str = r"nat\ddot{u}rlich" if self.method.boundary == "natural" else "Hermite"
        return [
            (r"$\checkmark\quad \mathrm{Kubischer\ Spline\ vollst\ddot{a}ndig}$", 12),
            (rf"$n = {n}\ \mathrm{{Segmente}},\quad \mathrm{{RB:\ {bnd_str}}}$", 11),
        ]

    def _latex_system_reduced(self) -> list[tuple[str, int]]:
        # Zeigt die Reduktion: Randzeilen streichen wegen M0=Mn=0
        solver = self.method.solver
        h = solver.h; f = solver.f; n = len(solver.x)
        A = self.method._A; rhs = self.method._rhs
        nr = A.shape[0]
        lines = [
            (r"$M_0=0\ \mathrm{und}\ M_n=0\ \Rightarrow\ \mathrm{Randzeilen\ streichen}$", 11),
            (r"$\mathrm{Reduziertes\ System\ der\ Gr\ddot{o}\ss e}\ "
             rf"({nr} \times {nr}):$", 11),
        ]
        for k in range(nr):
            i = k + 1
            ai_v = 2*(h[k]+h[k+1]); bi_v = h[k]; gi_v = h[k+1]
            di_v = rhs[k] / 6.0
            lines.append((
                rf"$i={i}:\quad \alpha_{{{i}}}={ai_v:.4f},\quad"
                rf"\beta_{{{i}}}={bi_v:.4f},\quad"
                rf"\gamma_{{{i}}}={gi_v:.4f},\quad"
                rf"d_{{{i}}}={di_v:.4f}$", 10
            ))
        return lines

    def _make_reduced_img(self, bg: str) -> ImageTk.PhotoImage:
        # Rendert nur das innere (n-2)×(n-2) System
        A   = self.method._A
        rhs = self.method._rhs
        solver = self.method.solver
        nr = A.shape[0]
        # Innere Matrix /6 skaliert, M-Labels M₁..Mₙ₋₁
        m_labels = [f"M_{{{i+1}}}" for i in range(nr)]
        img = render_matrix_img(A / 6.0, rhs / 6.0, m_labels, bg=bg, dpi=85)
        self._imgs.append(img)
        return img

    # Matrix-Bild für System-Schritt                   
    def _make_system_img(self, bg: str,
                         show_m_values: bool = False) -> ImageTk.PhotoImage:
        A_inner = self.method._A
        rhs_inner = self.method._rhs
        solver = self.method.solver
        bnd = solver.boundary
        n_inner = A_inner.shape[0]

        if bnd == "natural":
            # Vollständige Matrix inkl. h/6 an den Randpositionen
            n_full = n_inner + 2
            h = solver.h
            A_full = np.zeros((n_full, n_full))
            rhs_full = np.zeros(n_full)
            # Randzeile oben: alpha0=1
            A_full[0, 0] = 1.0
            # Innere Zeilen (skaliert /6): mit h/6 an den Außenpositionen
            A_full[1:n_inner+1, 1:n_inner+1] = A_inner / 6.0
            rhs_full[1:n_inner+1] = rhs_inner / 6.0
            # Erste innere Zeile: linker Nachbar M0 hat Koeffizient h[0]/6
            A_full[1, 0] = h[0] / 6.0
            # Letzte innere Zeile: rechter Nachbar Mn hat Koeffizient h[-1]/6
            A_full[n_inner, n_full-1] = h[-1] / 6.0
            # Randzeile unten: alpha0=1
            A_full[n_full-1, n_full-1] = 1.0

            if show_m_values:
                m_labels = [f"{solver.M[i]:.4f}" for i in range(n_full)]
            else:
                m_labels = [f"M_{{{i}}}" for i in range(n_full)]
            img = render_matrix_img(A_full, rhs_full, m_labels, bg=bg, dpi=85)
        else:
            if show_m_values:
                m_labels = [f"{solver.M[i]:.4f}" for i in range(n_inner)]
            else:
                m_labels = [f"M_{{{i}}}" for i in range(n_inner)]
            img = render_matrix_img(A_inner / 6.0, rhs_inner / 6.0, m_labels, bg=bg, dpi=85)

        self._imgs.append(img)
        return img

    # Button-Handler                           
    def _on_reset(self):
        self._on_reset_base()

    def _lock_controls(self, locked: bool):
        state = "disabled" if locked else "normal"
        for w in (self._func_entry, self._a_entry, self._b_entry,
                  self._n_entry, self._df0_entry, self._dfn_entry):
            try: w.config(state=state)
            except Exception: pass

    def _do_start(self):
        try:
            func_str    = self._func_var.get().strip()
            a           = float(self._a_var.get().replace(",", "."))
            b           = float(self._b_var.get().replace(",", "."))
            n_intervals = int(self._n_var.get()) - 1
            if n_intervals < 2:
                raise ValueError("n+1 muss mindestens 3 sein.")
            boundary  = self._boundary_var.get()
            df0 = float(self._df0_var.get().replace(",", ".")) \
                if boundary == "hermite" else 0.0
            dfn = float(self._dfn_var.get().replace(",", ".")) \
                if boundary == "hermite" else 0.0
        except ValueError as e:
            messagebox.showerror("Eingabefehler", str(e))
            return

        try:
            self.method.on_start(func_str, a, b, n_intervals,
                                boundary=boundary, df0=df0, dfn=dfn)
        except Exception as e:
            messagebox.showerror("Fehler", str(e))
            return

        self.started = True
        self.step_count = 0
        self._lock_controls(True)
        self._btn_step.config(text="Weiter")
        self._history.clear()
        self._imgs.clear()
        self._render_header()

        # Kein extra Leerplot hier – der setup-Schritt zeichnet den Startzustand sauber.
        self._do_step()

    def _do_step(self):
        if self.method.is_done():
            return
        try:
            status, step_obj, done = self.method.on_step()
        except Exception as e:
            messagebox.showerror("Fehler", str(e))
            return

        self.step_count += 1
        is_final = step_obj is not None and step_obj.kind == "done"
        bg = self.c_done if is_final else self.c_hi

        if step_obj is None:
            flines = [(r"$\mathrm{...}$", 11)]
            title  = f"Schritt {self.step_count}"
            mat    = None

        elif step_obj.kind == "setup":
            flines = self._latex_setup()
            title  = f"Schritt {self.step_count}: Schrittweite"
            mat    = None
            self._draw_empty_plot()

        elif step_obj.kind == "knot":
            i = step_obj.index
            flines = self._latex_knot(i)
            title  = f"Schritt {self.step_count}: Stützpunkt x_{i}"
            mat    = None
            self._draw_points_up_to(i)

        elif step_obj.kind == "boundary":
            flines = self._latex_boundary()
            title  = f"Schritt {self.step_count}: Randbedingung"
            mat    = None

        elif step_obj.kind == "system":
            flines = self._latex_system()
            title  = f"Schritt {self.step_count}: Tridiagonalsystem (vollständig)"
            mat    = self._make_system_img(bg)

        elif step_obj.kind == "system_reduced":
            flines = self._latex_system_reduced()
            title  = f"Schritt {self.step_count}: Reduziertes System (M₀=Mₙ=0 eingesetzt)"
            mat    = self._make_reduced_img(bg)

        elif step_obj.kind == "moment":
            flines = self._latex_moment()
            title  = f"Schritt {self.step_count}: Momente lösen"
            mat    = self._make_system_img(bg, show_m_values=True)

        elif step_obj.kind == "coeffs":
            flines = self._latex_coeffs(step_obj.index)
            title  = f"Schritt {self.step_count}: Segment {step_obj.index}"
            mat    = None

        elif step_obj.kind == "all_segments":
            flines = self._latex_all_segments()
            title  = f"Schritt {self.step_count}: Alle Segmente"
            mat    = None

        elif is_final:
            flines = self._latex_done()
            title  = "✓ Ergebnis"
            mat    = None

        else:
            flines = [(r"$\mathrm{...}$", 11)]
            title  = f"Schritt {self.step_count}"
            mat    = None

        self._add_step_card(self.step_count, flines, title, bg, matrix_img=mat)

        if step_obj is not None:
            if step_obj.kind == "coeffs":
                self._update_plot(
                    highlight_interval=step_obj.index,
                    build_up_to=step_obj.index
                )
            elif step_obj.kind in ("all_segments", "done"):
                self._update_plot(
                    highlight_interval=None,
                    build_up_to=len(self.method.solver.x) - 2
                )

        if done:
            self.started = False
            self._btn_step.config(text="Start", state="normal")
            messagebox.showinfo("Fertig", "Kubischer Spline vollständig berechnet.")

    def _apply_plot_limits(self, x_vals, y_vals, curve_x=None, curve_y=None):
        xs = list(map(float, x_vals))
        ys = [float(v) for v in y_vals if np.isfinite(v)]

        if curve_x is not None:
            xs.extend(float(v) for v in curve_x if np.isfinite(v))
        if curve_y is not None:
            ys.extend(float(v) for v in curve_y if np.isfinite(v))

        if not xs or not ys:
            return

        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        xpad = max(0.5, (xmax - xmin) * 0.12 if xmax != xmin else 1.0)
        ypad = max(0.5, (ymax - ymin) * 0.12 if ymax != ymin else 1.0)

        self.plotter.ax.set_xlim(xmin - xpad, xmax + xpad)
        self.plotter.ax.set_ylim(ymin - ypad, ymax + ypad)

    def _add_step_card(
        self,
        step_nr: int,
        formula_lines: list[tuple[str, int]],
        title: str,
        bg: str,
        matrix_img: ImageTk.PhotoImage | None = None
    ):
        card = tk.Frame(self._history.inner, bg=bg, relief="solid", bd=1)
        card.pack(fill="x", padx=4, pady=3)

        tk.Label(
            card,
            text=title,
            bg=self.bg_head,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
            padx=8,
            pady=3
        ).pack(fill="x")

        body = tk.Frame(card, bg=bg)
        body.pack(fill="x", expand=True, padx=6, pady=5)
        body.columnconfigure(0, weight=1)

        current_row = 0

        if formula_lines:
            width_in = 8.4
            fimg = render_formula_block(
                formula_lines,
                bg=bg,
                dpi=100,
                width_in=width_in
            )
            self._imgs.append(fimg)

            tk.Label(
                body,
                image=fimg,
                bg=bg,
                anchor="nw",
                justify="left"
            ).grid(row=current_row, column=0, sticky="w", padx=2, pady=(0, 6))

            current_row += 1

        if matrix_img is not None:
            sep = tk.Frame(body, bg="#d1d5db", height=1)
            sep.grid(row=current_row, column=0, sticky="ew", pady=(0, 6))
            current_row += 1

            tk.Label(
                body,
                image=matrix_img,
                bg=bg,
                anchor="w",
                justify="left"
            ).grid(row=current_row, column=0, sticky="w", padx=2)

        self._history.canvas.update_idletasks()
        self._history.canvas.yview_moveto(1.0)

def main():
    app = SplineGUI()
    app.mainloop()


if __name__ == "__main__":
    main()