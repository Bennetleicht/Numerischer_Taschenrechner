from __future__ import annotations
from typing import List
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'base_gui')))
from dgl_base_gui import Base_DGL_GUI

from mehrschritt_solver import (
    MehrschrittSolver,
    METHOD_AB,
    METHOD_AM,
    METHOD_BDF,
    METHODS,
    ALLOWED_ORDERS,
    AB_COEFFS,
    AM_COEFFS,
    BDF_ALPHAS,
    fraction_to_latex,
)

# Adams-Bashforth-Formel für die Mehrschrittverfahren generieren
def _ab_formula(order: int) -> str:
    coeffs = AB_COEFFS[order]
    parts = []
    for i, coeff in enumerate(coeffs):
        sign = "-" if coeff < 0 else "+"
        coeff_abs = fraction_to_latex(abs(coeff))
        term = rf"{coeff_abs}f_{{j-{i}}}"
        parts.append((sign, term))
    expr = parts[0][1] if parts[0][0] == "+" else rf"-{parts[0][1]}"
    for sign, term in parts[1:]:
        expr += rf" {sign} {term}"
    return rf"y_{{j+1}} = y_j + h\left({expr}\right)"

# Adams-Moulton-Formel für die Mehrschrittverfahren generieren
def _am_formula(order: int) -> str:
    coeffs = AM_COEFFS[order]
    parts = [("+" if coeffs[0] >= 0 else "-", rf"{fraction_to_latex(abs(coeffs[0]))}f_{{j+1}}")]
    for i, coeff in enumerate(coeffs[1:]):
        sign = "-" if coeff < 0 else "+"
        parts.append((sign, rf"{fraction_to_latex(abs(coeff))}f_{{j-{i}}}"))
    expr = parts[0][1] if parts[0][0] == "+" else rf"-{parts[0][1]}"
    for sign, term in parts[1:]:
        expr += rf" {sign} {term}"
    return rf"y_{{j+1}} = y_j + h\left({expr}\right)"

# BDF-Formel für die Mehrschrittverfahren generieren
def _bdf_formula(order: int) -> str:
    alphas = BDF_ALPHAS[order]
    parts = []
    for i, alpha in enumerate(alphas):
        idx = "j+1" if i == 0 else f"j+1-{i}"
        sign = "-" if alpha < 0 else "+"
        coeff_abs = fraction_to_latex(abs(alpha))
        parts.append((sign, rf"{coeff_abs}y_{{{idx}}}"))
    lhs = parts[0][1] if parts[0][0] == "+" else rf"-{parts[0][1]}"
    for sign, term in parts[1:]:
        lhs += rf" {sign} {term}"
    return rf"{lhs} = h f(t_{{j+1}},y_{{j+1}})"

# Generiert die Anzeigetexte für die Formeln basierend auf der gewählten Methode und Ordnung
def get_formula_lines(method: str, order: int) -> List[str]:
    common_general = {
        METHOD_AB: [
            r"$y_{j+1} = y_j + h\sum_{i=0}^{k-1} \beta_i f_{j-i}$",
        ],
        METHOD_AM: [
            r"$y_{j+1} = y_j + h\left(\beta_{-1}f(t_{j+1},y_{j+1}) + \sum_{i=0}^{k-1}\beta_i f_{j-i}\right)$",
        ],
        METHOD_BDF: [
            r"$\sum_{i=0}^{k} a_i\,y_{j+1-i} = h\,b\,f(t_{j+1},y_{j+1})$",
        ],
    }

    current = {
        METHOD_AB: rf"${_ab_formula(order)}$",
        METHOD_AM: rf"${_am_formula(order)}$",
        METHOD_BDF: rf"${_bdf_formula(order)}$",
    }

    if order > 1:
        boot_info = rf"Startwerte werden mit RK4 bis $y_{{{order-1}}}$ erzeugt"
    else:
        boot_info = "Kein zusätzlicher Startwert nötig"

    return [
        *common_general[method],
        f"Aktuelle Implementierung: Ordnung k={order}",
        current[method],
        boot_info,
    ]


class MehrschrittGUI(Base_DGL_GUI):
    def __init__(self, initial_method=None):
        super().__init__(initial_method=initial_method)
        self.title("Mehrschrittverfahren für Anfangswertprobleme")
    
    #Hilfsfunktionen
    def _get_methods(self):
        return METHODS

    def _get_default_method(self):
        return METHOD_AB

    def _has_order_selector(self):
        return True

    def _get_order_values(self):
        return ALLOWED_ORDERS

    def _get_order(self):
        return int(self.current_order.get())

    def _create_solver(self, f_expr, f, a, b, y0, h):
        return MehrschrittSolver(self.current_method, self._get_order(), f, f_expr, a, b, y0, h)

    def _build_compare_label(self, solver):
        return f"{self.current_method} | k={solver.order} | h={solver.h:.5g} | [{solver.a:.5g}, {solver.b:.5g}]"

    def _build_compare_signature(self, solver):
        return (
            self.current_method,
            int(solver.order),
            float(solver.h),
            float(solver.a),
            float(solver.b),
            tuple(np.asarray(solver.ts, dtype=float)),
            tuple(np.asarray(solver.ys, dtype=float)),
        )
    
    # Aktualisiert die angezeigten Formeln basierend auf der aktuellen Methode und Ordnung
    def _show_formula(self):
        lf = self.latex_frame
        lf.clear()
        order = self._get_order()
        lf.add_heading(f"{self.current_method} (Ordnung {order})")
        lf.add_latex_block(get_formula_lines(self.current_method, order), fontsize=13)

    # Generiert die Anzeigetexte für die Historie der vorherigen Schritte
    def _history_block(self, hist_t, hist_y, current_index):
        lines = []
        for offset, (t_val, y_val) in enumerate(zip(hist_t, hist_y)):
            idx = current_index - offset
            lines.append(rf"$t_{{{idx}}} = {t_val:.5g},\; y_{{{idx}}} = {y_val:.5g}$")
        return lines

    # Aktualisiert die Anzeige nach jedem Schritt mit den relevanten Informationen und Berechnungen
    # printing
    def _format_step_output(self, result):
        lf = self.latex_frame
        v = result.values
        method = v.get("method", self.current_method)
        order = v.get("order", self._get_order())
        h = v.get("h", result.t_next - result.t_n)

        
        i = result.n
        i_next = i + 1

        if not hasattr(self, "_printed_general_formulas"):
            self._printed_general_formulas = set()

        formula_key = (method, order)

        if v.get("bootstrap", False):
            idx = v.get("bootstrap_index", 0)
            lf.add_heading(f"Startschritt {idx + 1}")

            lf.add_latex_block([
                rf"$\text{{RK4-Startwert für Ordnung }}k={order}$",
                rf"$t_{{{idx}}} = {result.t_n:.5g}, \quad y_{{{idx}}} = {result.y_n:.5g}$",
                rf"$k_1 = {v['k1']:.5g}$",
                rf"$k_2 = {v['k2']:.5g}$",
                rf"$k_3 = {v['k3']:.5g}$",
                rf"$k_4 = {v['k4']:.5g}$",
                rf"$y_{{{idx+1}}} = y_{{{idx}}} + \frac{{h}}{{6}}(k_1 + 2k_2 + 2k_3 + k_4)$",
                rf"$y_{{{idx+1}}} = {result.y_next:.5g}, \quad t_{{{idx+1}}} = {result.t_next:.5g}$",
            ], fontsize=14)

            lf.add_latex(
                rf"$\mathbf{{t_{{{idx+1}}} = {result.t_next:.5g}, \quad y_{{{idx+1}}} = {result.y_next:.5g}}}$",
                fontsize=14
            )
            lf.scroll_bottom()
            return

        lf.add_heading(f"Schritt {i_next}")

        hist_t = v.get("hist_t", [])
        hist_y = v.get("hist_y", [])

        if method == METHOD_AB:
            coeffs = v["coeffs_fraction"]
            coeff_line = r",\; ".join(
                [rf"\beta_{{{m}}} = {fraction_to_latex(c)}" for m, c in enumerate(coeffs)]
            )

            terms = []
            for m, c in enumerate(coeffs):
                val = v["hist_f"][m]
                sign = "-" if c < 0 else "+"
                coeff = fraction_to_latex(abs(c))
                term = rf"{coeff}\cdot {val:.5g}"
                terms.append((sign, term))

            expr = terms[0][1] if terms[0][0] == "+" else rf"-{terms[0][1]}"
            for sign, term in terms[1:]:
                expr += rf" {sign} {term}"

            lf.add_latex_block([
                *self._history_block(hist_t, hist_y, i),
                rf"$t_{{{i_next}}} = t_{{{i}}} + h = {result.t_n:.5g} + {h:.5g} = {result.t_next:.5g}$",
                rf"${coeff_line}$",
                rf"$\sum_{{m=0}}^{{{order-1}}}\beta_m f_{{{i-m}}} = {expr} = {v['weighted_sum']:.5g}$",
                rf"$y_{{{i_next}}} = y_{{{i}}} + h\cdot \sum_{{m=0}}^{{{order-1}}}\beta_m f_{{{i-m}}}$",
                rf"$y_{{{i_next}}} = {result.y_n:.5g} + {h:.5g}\cdot {v['weighted_sum']:.5g} = {result.y_next:.5g}$",
            ], fontsize=14)

        elif method == METHOD_AM:
            coeffs = v["coeffs_fraction"]
            coeff_items = (
                [rf"\beta_{{-1}} = {fraction_to_latex(coeffs[0])}"] +
                [rf"\beta_{{{m}}} = {fraction_to_latex(c)}" for m, c in enumerate(coeffs[1:])]
            )

            lf.add_latex_block([
                *self._history_block(hist_t, hist_y, i),
                rf"$t_{{{i_next}}} = t_{{{i}}} + h = {result.t_n:.5g} + {h:.5g} = {result.t_next:.5g}$",
                rf"$" + r",\; ".join(coeff_items) + "$",
                rf"$\tilde y_{{{i_next}}}\;(\text{{Prädiktor}}) = {v['predictor']:.5g}$",
                rf"$f_{{{i_next}}} = f(t_{{{i_next}}}, y_{{{i_next}}}) = {v['f_next']:.5g}$",
                rf"$\beta_{{-1}}f_{{{i_next}}} + \sum_{{m=0}}^{{{order-1}}}\beta_m f_{{{i-m}}} = {v['weighted_sum']:.5g}$",
                rf"$y_{{{i_next}}} = y_{{{i}}} + h\left(\beta_{{-1}}f_{{{i_next}}} + \sum_{{m=0}}^{{{order-1}}}\beta_m f_{{{i-m}}}\right)$",
                rf"$y_{{{i_next}}} = {result.y_n:.5g} + {h:.5g}\cdot {v['weighted_sum']:.5g} = {result.y_next:.5g}$",
            ], fontsize=14)

        elif method == METHOD_BDF:
            alphas = v["alphas_fraction"]
            alpha_line = r",\; ".join(
                [rf"a_{{{m}}} = {fraction_to_latex(a)}" for m, a in enumerate(alphas)]
            )

            lf.add_latex_block([
                *self._history_block(hist_t, hist_y, i),
                rf"$t_{{{i_next}}} = t_{{{i}}} + h = {result.t_n:.5g} + {h:.5g} = {result.t_next:.5g}$",
                rf"${alpha_line}$",
                rf"$\tilde y_{{{i_next}}}\;(\text{{Startwert}}) = {v['predictor']:.5g}$",
                rf"$f_{{{i_next}}} = f(t_{{{i_next}}}, y_{{{i_next}}}) = {v['f_next']:.5g}$",
                rf"$y_{{{i_next}}} = {result.y_next:.5g}$",
            ], fontsize=14)

        lf.add_latex(
            rf"$\mathbf{{t_{{{i_next}}} = {result.t_next:.5g}, \quad y_{{{i_next}}} = {result.y_next:.5g}}}$",
            fontsize=14
        )
        lf.scroll_bottom()

# Ermöglicht die Auswahl der Anfangsmethode
def _method_from_argv():
    if len(sys.argv) < 2:
        return None
    arg = sys.argv[1].strip().lower()
    mapping = {
        "ab": METHOD_AB,
        "am": METHOD_AM,
        "bdf": METHOD_BDF,
    }
    return mapping.get(arg)


def _method_from_argv():
    import sys

    if len(sys.argv) > 1:
        return sys.argv[1]
    return None


def main():
    app = MehrschrittGUI(initial_method=_method_from_argv())
    app.mainloop()


if __name__ == "__main__":
    main()