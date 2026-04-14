from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

from bezier_method import BezierMethod
from bezier_solver import BezierSolver

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'plotter')))
from plotter.interpolations_plotter import InterpolationPlotter

from base_gui.gui_utils import ScrollableFrame
from gui.base_interp_gui import (BaseInterpGUI, render_formula_block)

_BASIS_COLORS = [
    "#e11d48", "#d97706", "#16a34a", "#2563eb",
    "#7c3aed", "#0891b2", "#be185d", "#65a30d",
]


# LaTeX Rendering

def render_bezier_header(algorithm: str, bg: str = "#f6f7fb",
                          dpi: int = 100):
    if algorithm == "casteljau":
        lines = [
            (r"$\mathbf{De\ Casteljau:}\quad Q_i^{(r)}(t) = "
             r"(1-t)\cdot Q_i^{(r-1)} + t\cdot Q_{i+1}^{(r-1)}$", 11),
            (r"$Q_i^{(0)} = P_i,\qquad "
             r"B(t) = Q_0^{(n)}(t)$", 11),
        ]
    else:
        lines = [
            (r"$\mathbf{Bernstein:}\quad "
             r"B(t) = \sum_{i=0}^{n} b_{i,n}(t)\cdot P_i$", 11),
            (r"$b_{i,n}(t) = \binom{n}{i} t^i (1-t)^{n-i},"
             r"\qquad t \in [0,1]$", 11),
        ]
    return render_formula_block(lines, bg=bg, dpi=dpi, width_in=7.5)


# Haupt GUI

class BezierGUI(BaseInterpGUI):

    def __init__(self):
        self.method = BezierMethod()
        self._mode = "calc"
        self._interactive_solver = BezierSolver()
        self._interactive_t = 0.5
        self._drag_active = False
        self._legend_map = {}
        self._hidden_levels: set = set()
        super().__init__()

    # UI
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.configure(bg=self.bg_app)

        # Kopfzeile
        self._top = ttk.Frame(self, padding=(10, 8), style="App.TFrame")
        self._top.grid(row=0, column=0, sticky="ew")
        self._build_header()

        # Hauptbereich -> wird bei Moduswechsel neu gebaut
        self._main_frame = ttk.Frame(self, padding=(10, 0, 10, 10),
                                     style="App.TFrame")
        self._main_frame.grid(row=1, column=0, sticky="nsew")
        self._build_calc_ui()

    def _build_header(self):
        top = self._top
        self._point_entries = []

        for w in top.winfo_children():
            w.destroy()

        ttk.Label(top, text="Bézierkurven",
                  style="Title.TLabel").grid(row=0, column=0, sticky="w",
                                             padx=(0, 16))

        # Modus Toggle
        mode_frame = ttk.Frame(top, style="App.TFrame")
        mode_frame.grid(row=0, column=1, sticky="w", padx=(0, 16))
        ttk.Label(mode_frame, text="Modus:", style="Hint.TLabel").pack(
            side="left", padx=(0, 6))

        self._btn_calc_mode = tk.Button(
            mode_frame, text="⚙ Berechnen",
            font=("Segoe UI", 10), relief="solid", bd=1,
            bg="#3b82f6", fg="white", cursor="hand2",
            command=lambda: self._set_mode("calc"))
        self._btn_calc_mode.pack(side="left", padx=(0, 4))

        self._btn_interactive_mode = tk.Button(
            mode_frame, text="↔ Interaktiv",
            font=("Segoe UI", 10), relief="solid", bd=1,
            bg=self.bg_card, fg="#374151", cursor="hand2",
            state="normal", command=lambda: self._set_mode("interactive"))
        self._btn_interactive_mode.pack(side="left")

        if self._mode == "calc":
            self._build_calc_header_controls(top, start_col=2)
        else:
            self._build_interactive_header_controls(top, start_col=2)

    def _build_calc_header_controls(self, top, start_col):
        # Algorithmus Toggle
        algo_frame = ttk.Frame(top, style="App.TFrame")
        algo_frame.grid(row=0, column=start_col, sticky="w", padx=(0, 16))
        ttk.Label(algo_frame, text="Algorithmus:", style="Hint.TLabel").pack(
            side="left", padx=(0, 6))

        self._algo_var = tk.StringVar(value=self.method.algorithm)
        self._btn_casteljau = tk.Button(
            algo_frame, text="De Casteljau",
            font=("Segoe UI", 10), relief="solid", bd=1,
            bg="#3b82f6" if self.method.algorithm == "casteljau" else self.bg_card,
            fg="white" if self.method.algorithm == "casteljau" else "#374151",
            cursor="hand2", command=lambda: self._set_algo("casteljau"))
        self._btn_casteljau.pack(side="left", padx=(0, 4))

        self._btn_bernstein = tk.Button(
            algo_frame, text="Bernstein",
            font=("Segoe UI", 10), relief="solid", bd=1,
            bg="#3b82f6" if self.method.algorithm == "bernstein" else self.bg_card,
            fg="white" if self.method.algorithm == "bernstein" else "#374151",
            cursor="hand2", command=lambda: self._set_algo("bernstein"))
        self._btn_bernstein.pack(side="left")

        col = start_col + 1
        self._input_vars = {"t": tk.StringVar(value="0.5")}
        self._input_widgets = {}

        ttk.Label(top, text="Parameter t ∈ [0,1]:", style="Hint.TLabel").grid(
            row=0, column=col, sticky="w", padx=(0, 4))
        col += 1
        t_entry = ttk.Entry(top, textvariable=self._input_vars["t"], width=6)
        t_entry.grid(row=0, column=col, sticky="w", padx=(0, 10))
        self._input_widgets["t"] = t_entry
        col += 1

        self._btn_step = ttk.Button(top, text="Start", command=self._on_step_btn)
        self._btn_step.grid(row=0, column=col, padx=(0, 6))
        col += 1
        ttk.Button(top, text="Reset", command=self._on_reset).grid(row=0, column=col)

        if not hasattr(self, "_point_count_var") or not isinstance(self._point_count_var, tk.StringVar):
            self._point_count_var = tk.StringVar(value="4")

        self._build_shared_header_content(top, col + 1)

    def _build_interactive_header_controls(self, top, start_col):
        col = start_col

        ttk.Label(top, text="Parameter t ∈ [0,1]:", style="Hint.TLabel").grid(
            row=0, column=col, sticky="w", padx=(0, 4))
        col += 1
        self._t_slider_var = tk.DoubleVar(value=getattr(self, "_interactive_t", 0.5))
        self._t_slider = ttk.Scale(top, from_=0.0, to=1.0,
                                   variable=self._t_slider_var,
                                   orient="horizontal", length=220,
                                   command=self._on_slider_change)
        self._t_slider.grid(row=0, column=col, sticky="w", padx=(0, 6))
        col += 1

        self._t_entry_var = tk.StringVar(value=f"{getattr(self, '_interactive_t', 0.5):.3f}")
        self._t_entry = ttk.Entry(top, textvariable=self._t_entry_var, width=6)
        self._t_entry.grid(row=0, column=col, sticky="w", padx=(0, 8))
        self._t_entry.bind("<Return>", self._on_t_entry)
        self._t_entry.bind("<FocusOut>", self._on_t_entry)
        col += 1

        ttk.Button(top, text="Anwenden", command=self._interactive_apply).grid(
            row=0, column=col, padx=(0, 6))
        col += 1
        ttk.Button(top, text="Reset", command=self._interactive_reset).grid(
            row=0, column=col)

        if not hasattr(self, "_point_count_var") or not isinstance(self._point_count_var, tk.StringVar):
            self._point_count_var = tk.StringVar(value="4")

        self._build_shared_header_content(top, col + 1)

    def _build_shared_header_content(self, top, colspan):
        self._header_content = ttk.Frame(top, style="App.TFrame")
        self._header_content.grid(row=1, column=0, columnspan=colspan, sticky="w", pady=(0, 0))
        self._header_content.columnconfigure(0, weight=0)
        self._header_content.columnconfigure(1, weight=0)

        self._points_box = ttk.Frame(self._header_content, style="App.TFrame")
        self._points_box.grid(row=0, column=0, sticky="nw")
        self._build_point_grid_controls(self._points_box)

        hdr_frame = ttk.Frame(self._header_content, style="App.TFrame")
        hdr_frame.grid(row=0, column=1, sticky="nw", padx=(12, 0), pady=(0, 0))
        self._hdr_canvas = tk.Canvas(hdr_frame, bg=self.bg_app, highlightthickness=0, height=56)
        self._hdr_canvas.pack(anchor="nw")
        self._render_latex_header()

    def _render_latex_header(self):
        if not hasattr(self, "_hdr_canvas"):
            return
        algo = "casteljau" if self._mode == "interactive" else self.method.algorithm
        img = render_bezier_header(algo, bg=self.bg_app, dpi=100)
        self._img_hdr = img
        self._hdr_canvas.configure(width=img.width(), height=img.height())
        self._hdr_canvas.delete("all")
        self._hdr_canvas.create_image(0, 0, anchor="nw", image=img)

    def _default_points(self):
        return [(0, 0), (1, 2), (3, 2), (4, 0)]

    # setzt Modus
    def _set_mode(self, mode: str):
        if mode == self._mode:
            return

        try:
            current_pts = self._read_point_pairs()
        except Exception:
            current_pts = None

        self._mode = mode
        self.started = False
        self.step_count = 0
        self._imgs.clear()
        self._drag_active = False

        self._build_header()
        self._apply_mode_button_styles()

        for w in self._main_frame.winfo_children():
            w.destroy()

        if mode == "calc":
            self._build_calc_ui()
        else:
            self._build_interactive_ui()

        if current_pts:
            self._restore_point_entries(current_pts)

        if mode == "interactive":
            try:
                pts = self._read_point_pairs()
                self._interactive_solver.start(pts)
            except Exception:
                self._interactive_solver.control_points = []
            self._draw_interactive()
        else:
            self._draw_empty_plot()

    def _apply_mode_button_styles(self):
        if self._mode == "calc":
            self._btn_calc_mode.config(bg="#3b82f6", fg="white")
            self._btn_interactive_mode.config(bg=self.bg_card, fg="#374151")
        else:
            self._btn_interactive_mode.config(bg="#3b82f6", fg="white")
            self._btn_calc_mode.config(bg=self.bg_card, fg="#374151")

    def _restore_point_entries(self, points):
        self._point_count_var.set(str(len(points)))
        self._build_point_grid_controls(self._points_box)

        for i, (px, py) in enumerate(points):
            if i >= len(self._point_entries):
                break
            ex, ey = self._point_entries[i]
            ex.delete(0, "end")
            ex.insert(0, str(px))
            ey.delete(0, "end")
            ey.insert(0, str(py))

    # Berechnungs UI
    def _build_calc_ui(self):
        main = self._main_frame
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main, style="App.TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)

        ttk.Label(left, text="Schrittdetails",
                  style="Title.TLabel").grid(row=0, column=0, sticky="w",
                                              pady=(0, 6))
        self._history = ScrollableFrame(left)
        self._history.grid(row=1, column=0, sticky="nsew")

        right = ttk.Frame(main, style="App.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        ttk.Label(right, text="Plot",
                  style="Title.TLabel").grid(row=0, column=0, sticky="w",
                                              pady=(0, 6))
        from plotter.interpolations_plotter import InterpolationPlotter

        self.plotter = InterpolationPlotter(right)
        self.plotter.widget().grid(row=1, column=0, sticky="nsew")
        self._draw_empty_plot()

    # Interaktiv UI
    def _build_interactive_ui(self):
        main = self._main_frame
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        self.plotter = InterpolationPlotter(main)
        self.plotter.widget().grid(row=0, column=0, sticky="nsew")

        # Maus Events für Drag auf der Kurve
        self.plotter.canvas.mpl_connect("button_press_event",   self._i_press)
        self.plotter.canvas.mpl_connect("motion_notify_event",  self._i_motion)
        self.plotter.canvas.mpl_connect("button_release_event", self._i_release)
        self.plotter.canvas.mpl_connect("pick_event",           self._i_legend_pick)

    # Interaktiv
    def _interactive_apply(self):
        try:
            pts = self._read_point_pairs()
            self._interactive_solver.start(pts)
        except Exception as e:
            messagebox.showerror("Eingabefehler", str(e))
            return
        self._hidden_levels.clear()
        self._draw_interactive()

    def _interactive_reset(self):
        self._hidden_levels.clear()
        self._interactive_t = 0.5
        if hasattr(self, "_t_slider_var"):
            self._t_slider_var.set(self._interactive_t)
        if hasattr(self, "_t_entry_var"):
            self._t_entry_var.set(f"{self._interactive_t:.3f}")
        try:
            pts = self._read_point_pairs()
            self._interactive_solver.start(pts)
        except Exception:
            self._interactive_solver.control_points = []
        self._draw_interactive()

    # Interaktiven Slider bauen
    def _on_slider_change(self, val):
        self._interactive_t = float(val)
        if hasattr(self, "_t_entry_var"):
            self._t_entry_var.set(f"{self._interactive_t:.3f}")
        self._draw_interactive()

    def _on_t_entry(self, event=None):
        try:
            val = float(self._t_entry_var.get().replace(",", "."))
            val = max(0.0, min(1.0, val))
            self._interactive_t = val
            self._t_entry_var.set(f"{val:.3f}")
            if hasattr(self, "_t_slider_var"):
                self._t_slider_var.set(val)
        except ValueError:
            self._t_entry_var.set(f"{self._interactive_t:.3f}")
        self._draw_interactive()

    # Interaktiver Maus-Drag
    def _i_press(self, event):
        if not hasattr(self, "plotter") or event.inaxes != self.plotter.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        if not self._interactive_solver.control_points:
            return

        bt_x, bt_y = self._interactive_solver.evaluate_casteljau(self._interactive_t)

        x0, x1 = self.plotter.ax.get_xlim()
        y0, y1 = self.plotter.ax.get_ylim()

        tol_x = 0.03 * max(abs(x1 - x0), 1e-9)
        tol_y = 0.03 * max(abs(y1 - y0), 1e-9)

        near_x = abs(event.xdata - bt_x) <= tol_x
        near_y = abs(event.ydata - bt_y) <= tol_y

        if near_x and near_y:
            self._drag_active = True
            try:
                self.plotter._panning = False
                self.plotter._pan_press = None
            except Exception:
                pass

    def _i_motion(self, event):
        if not self._drag_active or not hasattr(self, "plotter"):
            return
        if event.inaxes != self.plotter.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        if not self._interactive_solver.control_points:
            return

        # Finde das t, dessen Kurvenpunkt B(t) dem Mauspunkt am nächsten liegt
        xs, ys = self._interactive_solver.curve_points("casteljau", 500)
        if xs.size == 0:
            return

        mx = event.xdata
        my = event.ydata
        dists = (xs - mx) ** 2 + (ys - my) ** 2
        best = int(np.argmin(dists))
        self._interactive_t = best / max(len(xs) - 1, 1)

        if hasattr(self, "_t_slider_var"):
            self._t_slider_var.set(self._interactive_t)
        if hasattr(self, "_t_entry_var"):
            self._t_entry_var.set(f"{self._interactive_t:.3f}")

        self._draw_interactive()

    def _i_release(self, event):
        self._drag_active = False

    def _i_legend_pick(self, event):
        r = self._legend_map.get(event.artist)
        if r is None:
            return

        if r in self._hidden_levels:
            self._hidden_levels.discard(r)
        else:
            self._hidden_levels.add(r)

        self._draw_interactive()

    # Interaktives Zeichnen
    def _draw_interactive(self):
        if not hasattr(self, "plotter"):
            return
        solver = self._interactive_solver
        self.plotter.clear_plot()
        ax = self.plotter.ax
        ax.set_facecolor(self.bg_app)
        self.plotter.fig.patch.set_facecolor(self.bg_app)

        if not solver.control_points:
            ax.text(0.5, 0.5, "Kontrollpunkte eingeben und Anwenden drücken",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=12, color="#9ca3af")
            self.plotter.refresh()
            return

        t = self._interactive_t
        cpts = solver.control_points

        xs, ys = solver.curve_points("casteljau", 400)
        self.plotter.set_curve(xs, ys, label="Bézierkurve")

        cx = [p[0] for p in cpts]
        cy = [p[1] for p in cpts]
        self.plotter.set_points(cx, cy)
        ax.plot(cx, cy, "o--", color="#9ca3af", linewidth=1.2,
                markersize=8, zorder=2, label="Kontrollpolygon")
        ax.scatter([cx[0], cx[-1]], [cy[0], cy[-1]], s=120,
                   color="#dc2626", edgecolors="white", linewidths=1.2,
                   zorder=7, label="Endpunkte")
        
        self._legend_map = {}
        for i, (px, py) in enumerate(cpts):
            ax.annotate(f"$P_{i}$", (px, py),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=10, color="#374151")

        tableau = solver.de_casteljau_full(t)
        level_colors = ["#fbbf24", "#f97316", "#ef4444",
                        "#a855f7", "#06b6d4", "#84cc16"]

        for r in range(1, len(tableau) - 1):
            pts_r = tableau[r]
            rx = [p[0] for p in pts_r]
            ry = [p[1] for p in pts_r]
            col = level_colors[(r - 1) % len(level_colors)]
            visible = r not in self._hidden_levels
            ax.plot(rx, ry, "o-", color=col, linewidth=1.2,
                    markersize=6, alpha=0.75 if visible else 0.0,
                    zorder=4, visible=visible, label=f"Stufe {r}")

        bt = tableau[-1][0]
        self.plotter.set_eval_point(bt[0], bt[1], label=f"B(t={t:.3f})")
        self.plotter.vline_eval.set_visible(False)
        ax.scatter([bt[0]], [bt[1]], color="white", s=50,
                marker="o", zorder=9)
        ax.annotate(f"  $B(t={t:.3f})$\n  $=({bt[0]:.3f}, {bt[1]:.3f})$",
                    (bt[0], bt[1]),
                    textcoords="offset points", xytext=(10, -20),
                    fontsize=9, color="#16a34a",
                    bbox=dict(boxstyle="round,pad=0.3",
                            facecolor="white", alpha=0.85))

        ax.set_title("Bézierkurve – De Casteljau", fontsize=10, color="#374151")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, linestyle="--", alpha=0.35)

        self.plotter.auto_view()
        self.plotter.refresh()

        self.plotter.vline_eval.set_visible(False)

        self._legend_map = {}
        leg = ax.get_legend()
        if leg is not None:
            handles = getattr(leg, "legend_handles", [])
            texts = leg.get_texts()

            for handle, text in zip(handles, texts):
                txt = text.get_text()
                if txt.startswith("Stufe "):
                    try:
                        r = int(txt.split()[1])
                        handle.set_picker(True)
                        self._legend_map[handle] = r
                    except (IndexError, ValueError):
                        pass

        self.plotter.canvas.draw_idle()

    # Berechnungsplot
    def _draw_empty_plot(self):
        self.plotter.clear_plot()
        self.plotter.ax.set_facecolor(self.bg_app)
        self.plotter.ax.set_title("Bézierkurve", fontsize=11)
        self.plotter.ax.grid(True, linestyle="--", alpha=0.5)
        self.plotter.ax.text(
            0.5, 0.5,
            "Kontrollpunkte eingeben\nund Start drücken",
            ha="center", va="center",
            transform=self.plotter.ax.transAxes,
            fontsize=11, color="#9ca3af"
        )
        self.plotter.canvas.draw_idle()

    def _update_plot(self, step=None, **kwargs):
        solver = self.method.solver
        pts = list(solver.control_points)

        if not pts:
            self._draw_empty_plot()
            return

        x = [p[0] for p in pts]
        y = [p[1] for p in pts]

        bx, by = solver.curve_points(self.method.algorithm, 500)

        self.plotter.clear_plot()
        self.plotter.set_curve(bx, by, label="Bézierkurve")
        self.plotter.set_points(x, y)

        if len(x) >= 2:
            self.plotter.ax.plot(
                x, y,
                linestyle="--",
                linewidth=1.2,
                marker="o",
                alpha=0.8,
                label="Kontrollpolygon"
            )
            self.plotter.ax.scatter(
                [x[0], x[-1]], [y[0], y[-1]],
                s=120,
                color="#dc2626",
                edgecolors="white",
                linewidths=1.2,
                zorder=8,
                label="Endpunkte"
            )

        if step is not None and self.method.algorithm == "casteljau":
            self._plot_casteljau_step(self.plotter.ax, step, self.method.t_val)
        elif step is not None and self.method.algorithm == "bernstein":
            self._plot_bernstein_step(self.plotter.ax, step, self.method.t_val)

        if step is not None and step.kind == "done":
            px, py = solver.evaluate(self.method.algorithm, self.method.t_val)
            self.plotter.set_eval_point(px, py, label=f"B({self.method.t_val})")
            self.plotter.vline_eval.set_visible(False)
        else:
            self.plotter.clear_eval_point()

        self.plotter.ax.set_xlabel("x")
        self.plotter.ax.set_ylabel("y")
        self.plotter.ax.grid(True, linestyle="--", alpha=0.4)
        self.plotter.ax.set_title("Bézierkurve", fontsize=11)

        for i, (xi, yi) in enumerate(zip(x, y)):
            self.plotter.ax.annotate(
                f"P{i} = ({xi:.3f}, {yi:.3f})",
                (xi, yi),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8
            )

        self.plotter.auto_view()
        self.plotter.refresh()
        self.plotter.vline_eval.set_visible(False)
        self.plotter.canvas.draw_idle()

    def _plot_casteljau_step(self, ax, step, t):
        if step is None or step.kind == "done":
            return
        if step.kind == "level" and step.level_or_i > 0:
            pts = step.data
            lx = [p[0] for p in pts]; ly = [p[1] for p in pts]
            ax.plot(lx, ly, "o-", color="#f59e0b", linewidth=1.3,
                    markersize=6, zorder=5, label=f"Stufe {step.level_or_i}")

    def _plot_bernstein_step(self, ax, step, t):
        if step is None:
            return
        if step.kind == "bernstein_basis":
            pts = self.method.solver.control_points
            i = step.level_or_i
            b_val = step.data["b_val"]
            px, py = pts[i]
            contrib_x = b_val * px; contrib_y = b_val * py
            color = _BASIS_COLORS[i % len(_BASIS_COLORS)]
            ax.annotate(f"$b_{i}\\cdot P_{i}$",
                        xy=(contrib_x, contrib_y), xytext=(px, py),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                        fontsize=8, color=color)
            ax.scatter([contrib_x], [contrib_y], color=color, s=60, zorder=6)

    # Schritt Karten mit LaTeX 
    def _add_step_card(self, step_nr: int, step_obj, is_final: bool = False):
        bg = self.c_done if is_final else self.c_hi
        card = tk.Frame(self._history.inner, bg=bg, relief="solid", bd=1)
        card.pack(fill="x", padx=4, pady=3)

        title = "✓ Ergebnis" if is_final else f"Schritt {step_nr}"
        tk.Label(card, text=title, bg=self.bg_head,
                 font=("Segoe UI", 10, "bold"), anchor="w",
                 padx=8, pady=3).pack(fill="x")

        body = tk.Frame(card, bg=bg); body.pack(fill="x", padx=4, pady=3)

        # LaTeX-Formeln je nach Schritt-Art
        flines = self._latex_for_step(step_obj)
        if flines:
            img = render_formula_block(flines, bg=bg, dpi=100, width_in=6.5)
            self._imgs.append(img)
            tk.Label(body, image=img, bg=bg, anchor="nw").pack(
                anchor="nw")

        self._history.canvas.update_idletasks()
        self._history.canvas.yview_moveto(1.0)

    def _latex_for_step(self, step_obj) -> list[tuple[str, int]]:
        if step_obj is None:
            return []
        t = self.method.t_val
        algo = self.method.algorithm

        if step_obj.kind == "done":
            x, y = step_obj.data[0]
            return [
                (rf"$\checkmark\quad B(t={t}) = ({x:.6f},\ {y:.6f})$", 12),
            ]

        if algo == "casteljau":
            return self._latex_casteljau(step_obj, t)
        else:
            return self._latex_bernstein(step_obj, t)

    def _latex_casteljau(self, step, t) -> list[tuple[str, int]]:
        lines = []
        if step.kind == "level":
            r = step.level_or_i
            pts = step.data
            tableau = self.method.solver.de_casteljau_full(t)
            if r == 0:
                lines.append((r"$\mathrm{Stufe\ 0:\ Kontrollpunkte}$", 11))
                for i, p in enumerate(pts):
                    lines.append(
                        (rf"$Q_{{{i}}}^{{(0)}} = P_{{{i}}} = ({p[0]:.4f},\ {p[1]:.4f})$", 10))
            else:
                lines.append(
                    (rf"$\mathrm{{Stufe\ {r}:}}\quad Q_i^{{({r})}} = "
                     rf"(1-{t})\cdot Q_i^{{({r-1})}} + {t}\cdot Q_{{i+1}}^{{({r-1})}}$", 11))
                prev = tableau[r - 1]
                for i, p in enumerate(pts):
                    a, b = prev[i], prev[i + 1]
                    lines.append((
                        rf"$Q_{{{i}}}^{{({r})}} = "
                        rf"{1-t:.3f}\cdot({a[0]:.3f},{a[1]:.3f}) + "
                        rf"{t:.3f}\cdot({b[0]:.3f},{b[1]:.3f}) = "
                        rf"({p[0]:.4f},{p[1]:.4f})$", 10))
        return lines

    def _latex_bernstein(self, step, t) -> list[tuple[str, int]]:
        if step.kind != "bernstein_basis":
            return []
        i   = step.level_or_i
        n   = step.data["n"]
        b   = step.data["b_val"]
        cx, cy = step.data["contrib"]
        sx, sy = step.data["partial"]
        return [
            (rf"$b_{{{i},{n}}}(t={t}) = \binom{{{n}}}{{{i}}}"
             rf"\cdot {t}^{{{i}}}\cdot (1-{t})^{{{n-i}}} = {b:.6f}$", 11),
            (rf"$P_{{{i}}} \cdot b_{{{i},{n}}} = "
             rf"({step.data['contrib'][0]/b if b != 0 else 0:.3f},\ "
             rf"{step.data['contrib'][1]/b if b != 0 else 0:.3f}) "
             rf"\cdot {b:.4f} = ({cx:.4f},\ {cy:.4f})$", 10),
            (rf"$\mathrm{{Teilsumme:}}\quad ({sx:.4f},\ {sy:.4f})$", 10),
        ]

    # Algorithmus Toggle (Berechnungsmodus)
    def _set_algo(self, algo: str):
        if self.started:
            return
        self.method.set_algorithm(algo)
        self._algo_var.set(algo)
        if algo == "casteljau":
            self._btn_casteljau.config(bg="#3b82f6", fg="white")
            self._btn_bernstein.config(bg=self.bg_card, fg="#374151")
        else:
            self._btn_bernstein.config(bg="#3b82f6", fg="white")
            self._btn_casteljau.config(bg=self.bg_card, fg="#374151")
        self._render_latex_header()

    # Button Handler
    def _on_reset(self):
        if self._mode == "interactive":
            self._interactive_reset()
            return
        self._on_reset_base()

    def _do_start(self):
        try:
            pts = self._read_point_pairs()
            values = {
                "points": "; ".join(f"{x},{y}" for x, y in pts),
                "t": self._input_vars["t"].get(),
            }
            self.method.on_start(values)
        except Exception as e:
            messagebox.showerror("Eingabefehler", str(e)); return

        self.started = True
        self.step_count = 0
        self._imgs.clear()
        self._lock_controls(True)
        self._btn_step.config(text="Weiter")
        self._history.clear()
        self._render_latex_header()
        self._update_plot(step=None)

    def _do_step(self):
        if self.method.is_done():
            self.started = False
            self._btn_step.config(text="Start", state="normal")
            return

        try:
            status, step_obj, done = self.method.on_step()
        except Exception as e:
            messagebox.showerror("Fehler", str(e))
            return

        self.step_count += 1

        is_final = step_obj is not None and step_obj.kind == "done"
        self._add_step_card(self.step_count, step_obj, is_final=is_final)
        self._update_plot(step=step_obj)

        if done:
            self.started = False
            self._btn_step.config(text="Start", state="normal")
            result = self.method.get_result()
            if result:
                messagebox.showinfo(
                    "Ergebnis",
                    f"B(t = {self.method.t_val}) = ({result[0]:.6f}, {result[1]:.6f})"
                )


def main():
    app = BezierGUI()
    app.mainloop()


if __name__ == "__main__":
    main()