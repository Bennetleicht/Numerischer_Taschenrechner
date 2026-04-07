from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tkinter import messagebox
from io import BytesIO

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.figure as mfigure
import matplotlib.patches as mpatches
from PIL import Image, ImageTk

from polynom_method import PolynomMethod
from gui.base_interp_gui import BaseInterpGUI


# Render-Funktionen
def render_matrix(cell_rows: list[list[str]],
                  f_vals: list[str],
                  a_vals: list[str] | None = None,
                  revealed_rows: int = -1,
                  bg: str = "#fef9c3",
                  dpi: int = 100,
                  highlight: bool = True) -> ImageTk.PhotoImage:
    # Zeichnet V · a = f als strukturiertes Matrizenbild
    import matplotlib.pyplot as plt

    n  = len(cell_rows)
    nc = len(cell_rows[0])
    ph = "?"
    CHAR_W = 0.072

    def vec_width(vals: list[str]) -> float:
        return max(max(len(v) for v in vals) * CHAR_W + 0.14, 0.32)

    a_display = a_vals if a_vals else [ph] * n
    avec_w = vec_width(a_display)
    fvec_w = vec_width(f_vals)
    cell_w = 0.42; cell_h = 0.27; pad_l = 0.32; eq_gap = 0.22; pad_r = 0.88

    fig_w = pad_l + nc*cell_w + eq_gap + avec_w + eq_gap + fvec_w + pad_r
    fig_h = (n + 1.5) * cell_h

    fig = mfigure.Figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w); ax.set_ylim(0, fig_h)
    ax.set_axis_off(); ax.set_facecolor(bg)

    def cx(j):   return pad_l + (j + 0.5) * cell_w
    def ry(i):   return fig_h - (i + 2.0) * cell_h
    def hdr_y(): return fig_h - 1.2 * cell_h

    by_top = fig_h - 1.5 * cell_h
    by_bot = fig_h - (n + 1.5) * cell_h
    bx_l   = pad_l - 0.06; bx_r = pad_l + nc * cell_w + 0.06
    mid_y  = (by_top + by_bot) / 2
    lw = 1.6

    def bracket(xl, xr, yt, yb, open_=True):
        d = 0.06
        xs_ = [xl+d, xl, xl, xl+d] if open_ else [xr-d, xr, xr, xr-d]
        ax.plot(xs_, [yt, yt, yb, yb], color="#374151", lw=lw,
                solid_capstyle="round")

    # V-Matrix
    bracket(bx_l, bx_r, by_top, by_bot, True)
    bracket(bx_l, bx_r, by_top, by_bot, False)
    for j in range(nc):
        lbl = "$1$" if j == 0 else "$x$" if j == 1 else f"$x^{{{j}}}$"
        ax.text(cx(j), hdr_y(), lbl, ha="center", va="center",
                fontsize=8, color="#374151", fontweight="bold")
    for i in range(n):
        vis = (revealed_rows == -1) or (i < revealed_rows)
        if highlight and revealed_rows != -1 and i == revealed_rows - 1:
            rect = mpatches.FancyBboxPatch(
                (pad_l - 0.02, ry(i) - cell_h * 0.45),
                nc * cell_w + 0.04, cell_h * 0.90,
                boxstyle="round,pad=0.02",
                facecolor="#fde68a", edgecolor="none", zorder=0)
            ax.add_patch(rect)
        for j in range(nc):
            val   = cell_rows[i][j] if vis else ph
            color = "#1e293b" if vis else "#9ca3af"
            ax.text(cx(j), ry(i), val, ha="center", va="center",
                    fontsize=9, color=color)

    dot_x = bx_r + eq_gap * 0.38
    ax.text(dot_x, mid_y, r"$\cdot$", ha="center", va="center",
            fontsize=11, color="#374151")

    avl = dot_x + eq_gap * 0.55; avr = avl + avec_w
    bracket(avl, avr, by_top, by_bot, True)
    bracket(avl, avr, by_top, by_bot, False)
    avc = (avl + avr) / 2
    ax.text(avc, hdr_y(), "$a$", ha="center", va="center",
            fontsize=8, color="#374151", fontweight="bold")
    for i in range(n):
        ax.text(avc, ry(i), a_display[i], ha="center", va="center",
                fontsize=9, color="#1e293b" if a_vals else "#9ca3af")

    eq_x = avr + eq_gap * 0.38
    ax.text(eq_x, mid_y, "$=$", ha="center", va="center",
            fontsize=11, color="#374151")

    fvl = eq_x + eq_gap * 0.55; fvr = fvl + fvec_w
    bracket(fvl, fvr, by_top, by_bot, True)
    bracket(fvl, fvr, by_top, by_bot, False)
    fvc = (fvl + fvr) / 2
    ax.text(fvc, hdr_y(), "$f(x)$", ha="center", va="center",
            fontsize=8, color="#374151", fontweight="bold")
    for i in range(n):
        ax.text(fvc, ry(i), f_vals[i], ha="center", va="center",
                fontsize=9, color="#1e293b")

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=bg, pad_inches=0.35)
    plt.close(fig); buf.seek(0)
    return ImageTk.PhotoImage(Image.open(buf).convert("RGBA"))


def render_symbolic_vandermonde(bg: str = "#f6f7fb",
                                dpi: int = 100) -> ImageTk.PhotoImage:
    # Symbolische V·a=f als Header-Bild
    sym_rows = [
        ["$1$", "$x_0$", "$x_0^2$", "$\\cdots$", "$x_0^n$"],
        ["$1$", "$x_1$", "$x_1^2$", "$\\cdots$", "$x_1^n$"],
        ["$\\vdots$", "$\\vdots$", "$\\vdots$", "$\\ddots$", "$\\vdots$"],
        ["$1$", "$x_n$", "$x_n^2$", "$\\cdots$", "$x_n^n$"],
    ]
    sym_f = ["$f_0$", "$f_1$", "$\\vdots$", "$f_n$"]
    sym_a = ["$a_0$", "$a_1$", "$\\vdots$", "$a_n$"]

    import matplotlib.pyplot as plt
    n = len(sym_rows); nc = len(sym_rows[0])
    cell_w = 0.72; cell_h = 0.40; pad_l = 0.40; pad_r = 0.58
    avec_w = 0.55; eq_gap = 0.30; fvec_w = 0.55
    fig_w = pad_l + nc*cell_w + eq_gap + avec_w + eq_gap + fvec_w + pad_r
    fig_h = (n + 1.5) * cell_h

    fig = mfigure.Figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w); ax.set_ylim(0, fig_h)
    ax.set_axis_off(); ax.set_facecolor(bg)

    def cx(j): return pad_l + (j + 0.5) * cell_w
    def ry(i): return fig_h - (i + 2.0) * cell_h
    def hdr_y(): return fig_h - 1.2 * cell_h

    by_top = fig_h - 1.55 * cell_h
    by_bot = fig_h - (n + 1.55) * cell_h
    lw = 1.8

    def bracket(xl, xr, yt, yb, open_=True):
        d = 0.07
        xs_ = [xl+d, xl, xl, xl+d] if open_ else [xr-d, xr, xr, xr-d]
        ax.plot(xs_, [yt, yt, yb, yb], color="#374151", lw=lw,
                solid_capstyle="round")

    bx_l = pad_l - 0.07; bx_r = pad_l + nc*cell_w + 0.07
    bracket(bx_l, bx_r, by_top, by_bot, True)
    bracket(bx_l, bx_r, by_top, by_bot, False)
    for j, lbl in enumerate(["$1$", "$x$", "$x^2$", "$\\cdots$", "$x^n$"]):
        ax.text(cx(j), hdr_y(), lbl, ha="center", va="center",
                fontsize=9, color="#374151", fontweight="bold")
    for i in range(n):
        for j in range(nc):
            ax.text(cx(j), ry(i), sym_rows[i][j], ha="center", va="center",
                    fontsize=10, color="#1e293b")

    dot_x = pad_l + nc*cell_w + eq_gap * 0.35
    ax.text(dot_x, (by_top+by_bot)/2, r"$\cdot$", ha="center", va="center",
            fontsize=13, color="#374151")

    avl = dot_x + eq_gap * 0.6; avr = avl + avec_w * 0.85
    bracket(avl, avr, by_top, by_bot, True)
    bracket(avl, avr, by_top, by_bot, False)
    avc = (avl + avr) / 2
    for i in range(n):
        ax.text(avc, ry(i), sym_a[i], ha="center", va="center",
                fontsize=10, color="#1e293b")

    eq_x = avr + eq_gap * 0.4
    ax.text(eq_x, (by_top+by_bot)/2, "$=$", ha="center", va="center",
            fontsize=12, color="#374151")

    fvl = eq_x + eq_gap * 0.55; fvr = fvl + fvec_w * 0.85
    bracket(fvl, fvr, by_top, by_bot, True)
    bracket(fvl, fvr, by_top, by_bot, False)
    fvc = (fvl + fvr) / 2
    ax.text(fvc, hdr_y(), "$f(x)$", ha="center", va="center",
            fontsize=9, color="#374151", fontweight="bold")
    for i in range(n):
        ax.text(fvc, ry(i), sym_f[i], ha="center", va="center",
                fontsize=10, color="#1e293b")

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi,
                facecolor=bg, pad_inches=0.28)
    plt.close(fig); buf.seek(0)
    return ImageTk.PhotoImage(Image.open(buf).convert("RGBA"))


def _poly_descending(coeffs) -> str:
    if coeffs is None:
        return "—"
    n = len(coeffs) - 1
    parts = []
    for j in range(n, -1, -1):
        a = float(coeffs[j])
        if abs(a) < 1e-10:
            continue
        a_abs = f"{abs(a):.4f}".rstrip("0").rstrip(".") or "0"
        term = (a_abs if j == 0
                else f"{a_abs}·x" if j == 1
                else f"{a_abs}·x^{j}")
        if not parts:
            parts.append(f"-{term}" if a < 0 else term)
        else:
            parts.append(f" - {term}" if a < 0 else f" + {term}")
    return "p(x) = " + "".join(parts) if parts else "p(x) = 0"

def _poly_title_latex(coeffs) -> str:
    if coeffs is None:
        return r"$p(x)=0$"

    parts = []
    for j in range(len(coeffs) - 1, -1, -1):
        a = float(coeffs[j])
        if abs(a) < 1e-10:
            continue

        a_abs = f"{abs(a):.4f}".rstrip("0").rstrip(".") or "0"

        if j == 0:
            term = a_abs
        elif j == 1:
            term = rf"{a_abs}x"
        else:
            term = rf"{a_abs}x^{{{j}}}"

        if not parts:
            parts.append(f"-{term}" if a < 0 else term)
        else:
            parts.append(rf" - {term}" if a < 0 else rf" + {term}")

    expr = "".join(parts) if parts else "0"
    return rf"$p(x)={expr}$"


# GUI
class PolynomGUI(BaseInterpGUI):

    def __init__(self):
        self.method = PolynomMethod()
        super().__init__()

    # Header                               
    def _render_header(self):
        img = render_symbolic_vandermonde(bg=self.bg_app, dpi=100)
        self._imgs_hdr = img 
        self._hdr_canvas.configure(width=img.width(), height=img.height())
        self._hdr_canvas.delete("all")
        self._hdr_canvas.create_image(0, 0, anchor="nw", image=img)

    # Matrix-Hilfsmethode
    def _make_matrix_img(self, xs, fs, revealed_rows: int, bg: str,
                         a_vals: list[str] | None = None) -> ImageTk.PhotoImage:
        n = len(xs)
        cell_rows = [[f"{xs[i] ** j:.4g}" for j in range(n)] for i in range(n)]
        f_strs = [f"{fi:.4g}" for fi in fs]
        return render_matrix(cell_rows, f_strs, a_vals=a_vals,
                             revealed_rows=revealed_rows, bg=bg, dpi=100,
                             highlight=(revealed_rows != -1))

    # Plot                                
    def _draw_points_only(self):
        solver = self.method.solver
        x, f = solver.x, solver.f

        self.plotter.clear_plot()
        self.plotter.ax.set_facecolor(self.bg_app)
        self.plotter.set_points(x, f)
        self.plotter.ax.set_xlabel("x")
        self.plotter.ax.set_ylabel("p(x)")
        self.plotter.ax.grid(True, linestyle="--", alpha=0.4)
        self.plotter.set_title("Vandermondepolynom")

        for xi, fi in zip(x, f):
            self.plotter.ax.annotate(
                f"({xi:.3f}, {fi:.3f})", (xi, fi),
                textcoords="offset points", xytext=(5, 5), fontsize=8
            )

        self.plotter.auto_view()
        self.plotter.refresh()

    def _update_plot(self, step=None, **kwargs):
        solver = self.method.solver
        if not solver.x or solver.coeffs is None:
            self._draw_empty_plot()
            return

        x, f = solver.x, solver.f
        x_min, x_max = min(x), max(x)
        pad = max(0.5, (x_max - x_min) * 0.15)
        t = np.linspace(x_min - pad, x_max + pad, 400)
        p_t = solver.evaluate_array(t)

        self.plotter.clear_plot()
        self.plotter.ax.set_facecolor(self.bg_app)
        self.plotter.set_curve(t, p_t, label="Interpolationspolynom")
        self.plotter.set_points(x, f)

        self.plotter.ax.set_xlabel("x")
        self.plotter.ax.set_ylabel("p(x)")
        self.plotter.ax.grid(True, linestyle="--", alpha=0.4)
        title = "Vandermondepolynom\n" + _poly_title_latex(solver.coeffs)
        self.plotter.set_title(title)

        for xi, fi in zip(x, f):
            self.plotter.ax.annotate(
                f"({xi:.3f}, {fi:.3f})", (xi, fi),
                textcoords="offset points", xytext=(5, 5), fontsize=8
            )

        self.plotter.auto_view()
        self.plotter.refresh()

    # LaTeX Schritt Inhalte
    def _latex_vandermonde_row(self, i: int) -> list[tuple[str, int]]:
        x = self.method.solver.x
        n = len(x)
        lines = [(rf"$\mathrm{{Zeile\ {i}:}}\quad x_{{{i}}} = {x[i]}$", 11)]
        for j in range(n):
            val = x[i] ** j
            lines.append((rf"$x_{{{i}}}^{{{j}}} = {x[i]}^{{{j}}} = {val:.4g}$", 10))
        return lines

    def _latex_coeffs(self) -> list[tuple[str, int]]:
        coeffs = self.method.solver.coeffs
        n = len(coeffs)
        lines = [(r"$\mathrm{Gel\ddot{o}ste\ Koeffizienten\ (V \cdot a = f):}$", 11)]
        for j in range(n):
            lines.append((rf"$a_{{{j}}} = {coeffs[j]:.6f}$", 10))
        parts = []
        for j in range(n - 1, -1, -1):
            a = float(coeffs[j])
            if abs(a) < 1e-10: continue
            a_abs = f"{abs(a):.4f}".rstrip("0").rstrip(".") or "0"
            term = (a_abs if j == 0 else rf"{a_abs} \cdot x" if j == 1
                    else rf"{a_abs} \cdot x^{{{j}}}")
            if not parts: parts.append(f"-{term}" if a < 0 else term)
            else: parts.append(rf" - {term}" if a < 0 else rf" + {term}")
        poly = r"p(x) = " + "".join(parts) if parts else r"p(x) = 0"
        lines.append((rf"${poly}$", 12))
        return lines

    def _latex_done(self) -> list[tuple[str, int]]:
        coeffs = self.method.solver.coeffs
        parts = []
        for j in range(len(coeffs) - 1, -1, -1):
            a = float(coeffs[j])
            if abs(a) < 1e-10:
                continue
            a_abs = f"{abs(a):.4f}".rstrip("0").rstrip(".") or "0"
            term = (a_abs if j == 0 else rf"{a_abs} \cdot x" if j == 1
                    else rf"{a_abs} \cdot x^{{{j}}}")
            if not parts:
                parts.append(f"-{term}" if a < 0 else term)
            else:
                parts.append(rf" - {term}" if a < 0 else rf" + {term}")
        poly = r"p(x) = " + "".join(parts) if parts else r"p(x) = 0"
        return [(rf"${poly}$", 13)]
    
    # standard methode, reset, start, step
    def _on_reset(self):
        self._on_reset_base()

    def _do_start(self):
        try:
            pts = self._read_point_pairs()
            values = {
                "points": pts
            }
            self.method.on_start(values)
        except Exception as e:
            messagebox.showerror("Eingabefehler", str(e))
            return

        self.started = True
        self.step_count = 0
        self._lock_controls(True)
        self._btn_step.config(text="Weiter")
        self._history.clear()
        self._imgs.clear()
        self._render_header()
        self._draw_points_only()

    def _do_step(self):
        if self.method.is_done():
            return
        try:
            step_obj, done = self.method.on_step()
        except Exception as e:
            messagebox.showerror("Fehler", str(e))
            return

        self.step_count += 1
        is_final = step_obj.kind == "done"

        if step_obj.kind == "vandermonde":
            flines = self._latex_vandermonde_row(step_obj.row_i)
        elif step_obj.kind == "coeffs":
            flines = self._latex_coeffs()
        elif is_final:
            flines = self._latex_done()
        else:
            flines = [(rf"$\mathrm{{{step_obj.message}}}$", 11)]

        mat_img = None
        xs = self.method.solver.x
        fs = self.method.solver.f
        if xs and not is_final:
            bg = self.c_hi
            if step_obj.kind == "vandermonde":
                revealed = step_obj.row_i + 1; a_vals = None
            else:
                revealed = -1
                coeffs = self.method.solver.coeffs
                a_vals = ([f"{coeffs[i]:.4g}" for i in range(len(xs))]
                          if coeffs is not None else None)
            mat_img = self._make_matrix_img(xs, fs, revealed, bg, a_vals=a_vals)
            self._imgs.append(mat_img)

        title = "✓ Ergebnis" if is_final else f"Schritt {self.step_count}"
        bg = self.c_done if is_final else self.c_hi
        self._add_step_card(self.step_count, flines, title, bg,
                            matrix_img=mat_img)

        if step_obj.kind in ("coeffs", "eval", "done"):
            self._update_plot(step=step_obj)

        if done:
            self.started = False
            self._btn_step.config(text="Start", state="normal")
            poly = _poly_descending(self.method.solver.coeffs)
            messagebox.showinfo("Ergebnis", poly)

    def _uses_point_grid(self) -> bool:
        return True

    def _default_points(self) -> list[tuple[float, float]]:
        return [(0, 1), (1, 2), (2, 0), (3, 3)]

def main():
    app = PolynomGUI()
    app.mainloop()


if __name__ == "__main__":
    main()