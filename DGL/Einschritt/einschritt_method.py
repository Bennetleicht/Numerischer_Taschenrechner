from __future__ import annotations
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'base_gui')))
from dgl_base_gui import Base_DGL_GUI

from einschritt_solver import (
    EinschrittSolver,
    METHOD_EE,
    METHOD_HE,
    METHOD_ME,
    METHOD_RK,
    METHOD_IE,
    METHODS,
)
# Latex-Formeln
FORMULAS_LATEX = {
    METHOD_EE: [
        r"$y_{n+1} = y_n + h\,f(t_n,y_n)$",
    ],
    METHOD_HE: [
        r"$k_1 = f(t_n,y_n)$",
        r"$k_2 = f(t_n+h,\; y_n + h\,k_1)$",
        r"$y_{n+1} = y_n + \frac{h}{2}(k_1+k_2)$",
    ],
    METHOD_ME: [
        r"$k_1 = f(t_n,y_n)$",
        r"$y_{n+1} = y_n + h\,f\!\left(t_n+\frac{h}{2},\; y_n+\frac{h}{2}k_1\right)$",
    ],
    METHOD_RK: [
        r"$k_1 = f(t_n,y_n)$",
        r"$k_2 = f\!\left(t_n+\frac{h}{2},\; y_n+\frac{h}{2}k_1\right)$",
        r"$k_3 = f\!\left(t_n+\frac{h}{2},\; y_n+\frac{h}{2}k_2\right)$",
        r"$k_4 = f(t_n+h,\; y_n+h\,k_3)$",
        r"$y_{n+1} = y_n + \frac{h}{6}(k_1+2k_2+2k_3+k_4)$",
    ],
    METHOD_IE: [
        r"$y_{n+1} = y_n + h\,f(t_{n+1}, y_{n+1})$",
    ],
}


class EinschrittGUI(Base_DGL_GUI):
    def __init__(self, initial_method=None):
        super().__init__(initial_method=initial_method)
        self.title("Einschrittverfahren für Anfangswertprobleme")

    def _get_methods(self):
        return METHODS

    def _get_default_method(self):
        return METHOD_EE

    def _create_solver(self, f_expr, f, a, b, y0, h):
        return EinschrittSolver(self.current_method, f, f_expr, a, b, y0, h)

    def _build_compare_label(self, solver):
        return f"{self.current_method} | h={solver.h:.5g} | [{solver.a:.5g}, {solver.b:.5g}]"

    def _build_compare_signature(self, solver):
        return (
            self.current_method,
            float(solver.h),
            float(solver.a),
            float(solver.b),
            tuple(np.asarray(solver.ts, dtype=float)),
            tuple(np.asarray(solver.ys, dtype=float)),
        )

    #Zeigt die Formel für das aktuelle Verfahren an
    def _show_formula(self):
        lf = self.latex_frame
        lf.clear()
        display_name = self.current_method
        if display_name == METHOD_RK:
            display_name = "Runge-Kutta-Verfahren 4. Ordnung"

        lf.add_heading(display_name)
        lf.add_latex_block(FORMULAS_LATEX.get(self.current_method, []), fontsize=13)

    #Formatiert die Ausgabe für einen Schritt des Verfahrens
    def _format_step_output(self, result):
        lf = self.latex_frame
        v = result.values
        method = v.get("method", self.current_method)
        h = v.get("h", result.t_next - result.t_n)

        lf.add_heading(f"Schritt {result.n + 1}")

        #  Expliziter Euler 
        if method == METHOD_EE:
            yprime_n = v["yprime_n"]

            lf.add_latex_block([
                rf"$t_{{{result.n}}} = {result.t_n:.5g}, \quad "
                rf"y_{{{result.n}}} = {result.y_n:.5g}, \quad "
                rf"y'_{{{result.n}}} = {yprime_n:.5g}$",

                rf"$t_{{{result.n+1}}} = t_{{{result.n}}} + h$",

                rf"$t_{{{result.n+1}}} = {result.t_n:.5g} + {h:.5g} = {result.t_next:.5g}$",

                rf"$y_{{{result.n+1}}} = y_{{{result.n}}} + h \cdot y'_{{{result.n}}}$",

                rf"$y_{{{result.n+1}}} = {result.y_n:.5g} + {h:.5g} \cdot {yprime_n:.5g}$",
            ], fontsize=14)

        #  Heun 
        elif method == METHOD_HE:
            k1 = v["k1"]
            y_star = v["y_star"]
            k2 = v["k2"]

            lf.add_latex_block([
                rf"$t_{{{result.n}}} = {result.t_n:.5g}, \quad "
                rf"y_{{{result.n}}} = {result.y_n:.5g}, \quad "
                rf"y'_{{{result.n}}} = {k1:.5g}$",

                rf"$t_{{{result.n+1}}} = t_{{{result.n}}} + h$",

                rf"$t_{{{result.n+1}}} = {result.t_n:.5g} + {h:.5g} = {result.t_next:.5g}$",

                rf"$k_1 = y'_{{{result.n}}} = {k1:.5g}$",

                rf"$y^* = y_{{{result.n}}} + h \cdot k_1 = {result.y_n:.5g} + {h:.5g} \cdot {k1:.5g} = {y_star:.5g}$",

                rf"$k_2 = y'(t_{{{result.n+1}}}, y^*) = y'({result.t_next:.5g}, {y_star:.5g}) = {k2:.5g}$",

                rf"$y_{{{result.n+1}}} = y_{{{result.n}}} + \frac{{h}}{{2}}(k_1+k_2)$",

                rf"$y_{{{result.n+1}}} = {result.y_n:.5g} + \frac{{{h:.5g}}}{{2}}({k1:.5g}+{k2:.5g})$",
            ], fontsize=14)

        #  Modifizierter Euler 
        elif method == METHOD_ME:
            k1 = v["k1"]
            t_mid = v["t_mid"]
            y_mid = v["y_mid"]
            yprime_mid = v["yprime_mid"]

            lf.add_latex_block([
                rf"$t_{{{result.n}}} = {result.t_n:.5g}, \quad "
                rf"y_{{{result.n}}} = {result.y_n:.5g}, \quad "
                rf"y'_{{{result.n}}} = {k1:.5g}$",

                rf"$t_{{{result.n+1}}} = t_{{{result.n}}} + h$",

                rf"$t_{{{result.n+1}}} = {result.t_n:.5g} + {h:.5g} = {result.t_next:.5g}$",

                rf"$k_1 = y'_{{{result.n}}} = {k1:.5g}$",

                rf"$t_m = t_{{{result.n}}} + \frac{{h}}{{2}} = {result.t_n:.5g} + \frac{{{h:.5g}}}{{2}} = {t_mid:.5g}$",

                rf"$y_m = y_{{{result.n}}} + \frac{{h}}{{2}}k_1 = {result.y_n:.5g} + \frac{{{h:.5g}}}{{2}} \cdot {k1:.5g} = {y_mid:.5g}$",

                rf"$y'(t_m,y_m) = y'({t_mid:.5g}, {y_mid:.5g}) = {yprime_mid:.5g}$",

                rf"$y_{{{result.n+1}}} = y_{{{result.n}}} + h \cdot y'(t_m, y_m)$",

                rf"$y_{{{result.n+1}}} = {result.y_n:.5g} + {h:.5g} \cdot {yprime_mid:.5g}$",
            ], fontsize=14)

        #  RK4 
        elif method == METHOD_RK:
            k1 = v["k1"]
            k2 = v["k2"]
            k3 = v["k3"]
            k4 = v["k4"]

            lf.add_latex_block([
                rf"$t_{{{result.n}}} = {result.t_n:.5g}, \quad y_{{{result.n}}} = {result.y_n:.5g}$",

                rf"$t_{{{result.n+1}}} = t_{{{result.n}}} + h$",

                rf"$t_{{{result.n+1}}} = {result.t_n:.5g} + {h:.5g} = {result.t_next:.5g}$",

                rf"$k_1 = y'({result.t_n:.5g}, {result.y_n:.5g}) = {k1:.5g}$",

                rf"$k_2 = y'\!\left({result.t_n + h/2:.5g}, {result.y_n + h/2*k1:.5g}\right) = {k2:.5g}$",

                rf"$k_3 = y'\!\left({result.t_n + h/2:.5g}, {result.y_n + h/2*k2:.5g}\right) = {k3:.5g}$",

                rf"$k_4 = y'({result.t_next:.5g}, {result.y_n + h*k3:.5g}) = {k4:.5g}$",

                rf"$y_{{{result.n+1}}} = y_{{{result.n}}} + \frac{{h}}{{6}}(k_1 + 2k_2 + 2k_3 + k_4)$",

                rf"$y_{{{result.n+1}}} = {result.y_n:.5g} + \frac{{{h:.5g}}}{{6}}({k1:.5g} + 2\cdot {k2:.5g} + 2\cdot {k3:.5g} + {k4:.5g})$",
            ], fontsize=14)

        #  Impliziter Euler
        elif method == METHOD_IE:
            lf.add_latex_block([
                rf"$t_{{{result.n}}} = {result.t_n:.5g}, \quad y_{{{result.n}}} = {result.y_n:.5g}$",

                rf"$t_{{{result.n+1}}} = t_{{{result.n}}} + h$",

                rf"$t_{{{result.n+1}}} = {result.t_n:.5g} + {h:.5g} = {result.t_next:.5g}$",

                rf"$y_{{{result.n+1}}} = y_{{{result.n}}} + h \cdot y'(t_{{{result.n+1}}}, y_{{{result.n+1}}})$",

                rf"$y_{{{result.n+1}}} = {result.y_n:.5g} + {h:.5g} \cdot y'({result.t_next:.5g}, y_{{{result.n+1}}})$",
            ], fontsize=14)

        #  Ergebnis 
        lf.add_latex(
            rf"$\mathbf{{t_{{{result.n+1}}} = {result.t_next:.5g}, \quad y_{{{result.n+1}}} = {result.y_next:.5g}}}$",
            fontsize=14
        )

        lf.scroll_bottom()


def _method_from_argv():
    import sys

    if len(sys.argv) > 1:
        return sys.argv[1]
    return None


def main():
    app = EinschrittGUI(initial_method=_method_from_argv())
    app.mainloop()


if __name__ == "__main__":
    main()