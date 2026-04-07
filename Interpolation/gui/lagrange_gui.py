from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tkinter import messagebox
import numpy as np

from lagrange_method import LagrangeMethod
from gui.base_interp_gui import BaseInterpGUI, render_formula_block


# Hilfsfunktionen
def _fmt(v: float) -> str:
    return f"{v:.4g}"


def _poly_latex(coeffs) -> str:
    n = len(coeffs) - 1
    parts = []
    for j in range(n, -1, -1):
        a = float(coeffs[j])
        if abs(a) < 1e-10:
            continue
        a_abs = f"{abs(a):.4f}".rstrip("0").rstrip(".")
        if not a_abs:
            a_abs = "0"
        if j == 0:
            term = a_abs
        elif j == 1:
            term = rf"{a_abs} \cdot x"
        else:
            term = rf"{a_abs} \cdot x^{{{j}}}"
        if not parts:
            parts.append(f"-{term}" if a < 0 else term)
        else:
            parts.append(rf" - {term}" if a < 0 else rf" + {term}")
    return r"p(x) = " + "".join(parts) if parts else r"p(x) = 0"

def _poly_title_latex(coeffs) -> str:
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



#GUI       
class LagrangeGUI(BaseInterpGUI):

    def __init__(self):
        self.method = LagrangeMethod()
        super().__init__()

    # Header-Formel
    def _render_header(self):
        lines = [
            (r"$p(x) = \sum_{i=0}^{n} f_i \cdot L_i(x) \qquad "
             r"L_i(x) = \prod_{j \neq i} \dfrac{x - x_j}{x_i - x_j}$", 13),
        ]
        img = render_formula_block(lines, bg=self.bg_app, dpi=110, width_in=7.2)
        self._imgs.append(img)
        self._hdr_canvas.configure(width=img.width(), height=img.height())
        self._hdr_canvas.delete("all")
        self._hdr_canvas.create_image(0, 0, anchor="nw", image=img)

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
        title = "Lagrange-Interpolation"
        self.plotter.set_title(title)

        for xi, fi in zip(x, f):
            self.plotter.ax.annotate(
                f"({xi:.3f}, {fi:.3f})", (xi, fi),
                textcoords="offset points", xytext=(6, 6), fontsize=8
            )

        self.plotter.auto_view()
        self.plotter.refresh()

    def _update_plot(self):
        solver = self.method.solver
        x = solver.x
        f = solver.f

        x_min, x_max = min(x), max(x)
        pad = max(0.5, (x_max - x_min) * 0.15)
        t = np.linspace(x_min - pad, x_max + pad, 400)
        p_t = solver.evaluate_array(t)
        coeffs = solver.polynomial_coeffs()

        self.plotter.clear_plot()
        self.plotter.ax.set_facecolor(self.bg_app)
        self.plotter.set_curve(t, p_t, label="Interpolationspolynom")
        self.plotter.set_points(x, f)
        self.plotter.ax.set_xlabel("x")
        self.plotter.ax.set_ylabel("p(x)")
        self.plotter.ax.grid(True, linestyle="--", alpha=0.4)
        title = "Lagrange-Interpolation\n" + _poly_title_latex(coeffs)
        self.plotter.set_title(title)

        for xi, fi in zip(x, f):
            self.plotter.ax.annotate(
                f"({xi:.3f}, {fi:.3f})", (xi, fi),
                textcoords="offset points", xytext=(6, 6), fontsize=8
            )

        self.plotter.auto_view()
        self.plotter.refresh()

    # LaTeX Schritte Inhalt
    def _latex_basis(self, i: int) -> list[tuple[str, int]]:
        x = self.method.solver.x
        n = len(x)
        num = r" \cdot ".join(f"(x - {_fmt(x[j])})" for j in range(n) if j != i)
        den = r" \cdot ".join(f"({_fmt(x[i])} - {_fmt(x[j])})" for j in range(n) if j != i)
        den_val = 1.0
        for j in range(n):
            if j != i:
                den_val *= (x[i] - x[j])
        return [
            (rf"$L_{{{i}}}(x) = \dfrac{{{num}}}{{{den}}}$", 12),
            (rf"$\quad\quad\quad = \dfrac{{{num}}}{{{den_val:.4g}}}$", 12),
        ]

    def _latex_expand(self) -> list[tuple[str, int]]:
        x = self.method.solver.x
        f = self.method.solver.f
        n = len(x)
        lines: list[tuple[str, int]] = [
            (rf"$p(x) = \sum_{{i=0}}^{{{n-1}}} f_i \cdot L_i(x)$", 11),
        ]
        for i in range(n):
            den_val = 1.0
            for j in range(n):
                if j != i:
                    den_val *= (x[i] - x[j])
            num_str = r" \cdot ".join(
                f"(x - {_fmt(x[j])})" for j in range(n) if j != i
            )
            prefix = r"\quad =" if i == 0 else r"\quad +"
            lines.append((
                rf"${prefix}\; {_fmt(f[i])} \cdot \dfrac{{{num_str}}}{{{den_val:.4g}}}$",
                10
            ))
        return lines

    def _latex_done(self) -> list[tuple[str, int]]:
        coeffs = self.method.solver.polynomial_coeffs()
        return [(rf"${_poly_latex(coeffs)}$", 13)]

    # standard methoden, reset, start, step
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
        is_expand = step_obj.kind == "expand"
        hi_i = step_obj.index if step_obj.kind == "basis" and step_obj.index >= 0 else None

        if hi_i is not None:
            flines = self._latex_basis(hi_i)
            title = f"Schritt {self.step_count}: L_{hi_i}(x)"
            bg = self.c_hi
        elif is_expand:
            flines = self._latex_expand()
            title = f"Schritt {self.step_count}: Ausmultipliziert"
            bg = "#dbeafe"
        elif is_final:
            flines = self._latex_done()
            title = "✓ Ergebnis"
            bg = self.c_done
        else:
            flines = [(rf"$\mathrm{{{step_obj.message}}}$", 11)]
            title = f"Schritt {self.step_count}"
            bg = self.c_hi

        self._add_step_card(self.step_count, flines, title, bg)

        if is_final or is_expand:
            self._update_plot()

        if done:
            self.started = False
            self._btn_step.config(text="Start", state="normal")
            messagebox.showinfo("Ergebnis", step_obj.message)

    def _uses_point_grid(self) -> bool:
        return True

    def _default_points(self) -> list[tuple[float, float]]:
        return [(0, 1), (1, 2), (2, 0), (3, 3)]

def main():
    app = LagrangeGUI()
    app.mainloop()


if __name__ == "__main__":
    main()