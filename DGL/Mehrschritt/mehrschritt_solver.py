from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction
import sympy as sp


METHOD_AB = "Adams-Bashforth explizit"
METHOD_AM = "Adams-Moulton implizit"
METHOD_BDF = "BDF-Verfahren"
METHODS = [METHOD_AB, METHOD_AM, METHOD_BDF]

ALLOWED_ORDERS = [1, 2, 3, 4, 5, 6]

#Adams Bashforth Koeffizienten
AB_COEFFS = {
    1: [Fraction(1, 1)],
    2: [Fraction(3, 2), Fraction(-1, 2)],
    3: [Fraction(23, 12), Fraction(-16, 12), Fraction(5, 12)],
    4: [Fraction(55, 24), Fraction(-59, 24), Fraction(37, 24), Fraction(-9, 24)],
    5: [Fraction(1901, 720), Fraction(-1387, 360), Fraction(109, 30), Fraction(-637, 360), Fraction(251, 720)],
    6: [Fraction(4277, 1440), Fraction(-2641, 480), Fraction(4991, 720), Fraction(-3649, 720), Fraction(959, 480), Fraction(-95, 288)],
}

#Adams Moulton Koeffizienten
AM_COEFFS = {
    1: [Fraction(1, 1)],
    2: [Fraction(1, 2), Fraction(1, 2)],
    3: [Fraction(5, 12), Fraction(8, 12), Fraction(-1, 12)],
    4: [Fraction(9, 24), Fraction(19, 24), Fraction(-5, 24), Fraction(1, 24)],
    5: [Fraction(251, 720), Fraction(323, 360), Fraction(-11, 30), Fraction(53, 360), Fraction(-19, 720)],
    6: [Fraction(95, 288), Fraction(1427, 1440), Fraction(-133, 240), Fraction(241, 720), Fraction(-173, 1440), Fraction(3, 160)],
}

# BDF Koeffizienten 
BDF_ALPHAS = {
    1: [Fraction(1, 1), Fraction(-1, 1)],
    2: [Fraction(3, 2), Fraction(-2, 1), Fraction(1, 2)],
    3: [Fraction(11, 6), Fraction(-3, 1), Fraction(3, 2), Fraction(-1, 3)],
    4: [Fraction(25, 12), Fraction(-4, 1), Fraction(3, 1), Fraction(-4, 3), Fraction(1, 4)],
    5: [Fraction(137, 60), Fraction(-5, 1), Fraction(5, 1), Fraction(-10, 3), Fraction(5, 4), Fraction(-1, 5)],
    6: [Fraction(49, 20), Fraction(-6, 1), Fraction(15, 2), Fraction(-20, 3), Fraction(15, 4), Fraction(-6, 5), Fraction(1, 6)],
}

# Hilfsfunktion zur Umwandlung von Fraction in LaTeX-String
def fraction_to_latex(fr: Fraction) -> str:
    fr = Fraction(fr)
    if fr.denominator == 1:
        return str(fr.numerator)
    return rf"\frac{{{fr.numerator}}}{{{fr.denominator}}}"


@dataclass
class StepResult:
    n: int
    t_n: float
    y_n: float
    t_next: float
    y_next: float
    text: str = ""
    values: dict = field(default_factory=dict)


class MehrschrittSolver:
    def __init__(self, method_name: str, order: int, f, f_expr, a: float, b: float, y0: float, h: float):
        self.method_name = method_name
        self.order = int(order)
        self.f = f
        self.f_expr = f_expr

        self.a = float(a)
        self.b = float(b)
        self.y0 = float(y0)
        self.h = float(h)

        if self.method_name not in METHODS:
            raise ValueError(f"Unbekanntes Verfahren: {self.method_name}")
        if self.order not in ALLOWED_ORDERS:
            raise ValueError(f"Ordnung muss in {ALLOWED_ORDERS} liegen.")
        if self.h <= 0:
            raise ValueError("Schrittweite h muss positiv sein.")
        if self.b <= self.a:
            raise ValueError("b muss größer als a sein.")

        self.ts = [self.a]
        self.ys = [self.y0]
        self.n = 0

        self._t_sym = sp.Symbol("t")
        self._y_sym = sp.Symbol("y")

    @property
    def is_finished(self) -> bool:
        return self.ts[-1] >= self.b - 1e-14

    #passt die Schrittweite an, um genau bei b zu enden
    def current_step_size(self) -> float:
        return min(self.h, self.b - self.ts[-1])

    # Benötigt wird die Anzahl vorheriger Werte, die für den nächsten Schritt notwendig ist
    def _required_history(self) -> int:
        return self.order

    # Berechnet den nächsten Schritt und gibt alle relevanten Informationen zurück
    def step(self) -> StepResult:
        if self.is_finished:
            raise RuntimeError("Intervall bereits vollständig berechnet.")

        if len(self.ts) < self._required_history():
            y_next, text, values = self._bootstrap_step()
        else:
            t_n = float(self.ts[-1])
            y_n = float(self.ys[-1])
            h_n = float(self.current_step_size())

            # Je nach Methode den nächsten Wert berechnen
            if self.method_name == METHOD_AB:
                y_next, text, values = self._step_ab(t_n, y_n, h_n)
            elif self.method_name == METHOD_AM:
                y_next, text, values = self._step_am(t_n, y_n, h_n)
            elif self.method_name == METHOD_BDF:
                y_next, text, values = self._step_bdf(t_n, y_n, h_n)
            else:
                raise ValueError(f"Unbekanntes Verfahren: {self.method_name}")

        t_n = float(self.ts[-1])
        t_next = min(t_n + float(self.current_step_size()), self.b)
        self.ts.append(float(t_next))
        self.ys.append(float(y_next))

        result = StepResult(
            n=self.n,
            t_n=t_n,
            y_n=float(self.ys[-2]),
            t_next=t_next,
            y_next=float(y_next),
            text=text,
            values=values,
        )
        self.n += 1
        return result

    # Berechnet die ersten Schritte mit einem Runge-Kutta-Verfahren, um genügend Werte für das Mehrschrittverfahren zu haben
    def _bootstrap_step(self):
        t_n = float(self.ts[-1])
        y_n = float(self.ys[-1])
        h = float(self.current_step_size())
        t_next = t_n + h

        k1 = float(self.f(t_n, y_n))
        k2 = float(self.f(t_n + h / 2.0, y_n + h / 2.0 * k1))
        k3 = float(self.f(t_n + h / 2.0, y_n + h / 2.0 * k2))
        k4 = float(self.f(t_n + h, y_n + h * k3))
        y_next = y_n + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        values = {
            "method": self.method_name,
            "order": self.order,
            "bootstrap": True,
            "h": h,
            "k1": k1,
            "k2": k2,
            "k3": k3,
            "k4": k4,
            "t_prev": t_n,
            "y_prev": y_n,
            "bootstrap_index": len(self.ts) - 1,
            "bootstrap_needed": self._required_history() - 1,
        }
        return y_next, "", values

    # Berechnet den nächsten Schritt mit Adams-Bashforth explizit
    def _step_ab(self, t_n: float, y_n: float, h: float):
        coeffs = AB_COEFFS[self.order]
        hist_y = [float(self.ys[-1 - i]) for i in range(self.order)]
        hist_t = [float(self.ts[-1 - i]) for i in range(self.order)]
        hist_f = [float(self.f(hist_t[i], hist_y[i])) for i in range(self.order)]
        weighted_sum = sum(float(coeffs[i]) * hist_f[i] for i in range(self.order))
        y_next = y_n + h * weighted_sum

        values = {
            "method": self.method_name,
            "order": self.order,
            "bootstrap": False,
            "h": h,
            "hist_t": hist_t,
            "hist_y": hist_y,
            "hist_f": hist_f,
            "coeffs": [float(c) for c in coeffs],
            "coeffs_fraction": coeffs,
            "weighted_sum": weighted_sum,
        }
        return y_next, "", values

    # Berechnet den nächsten Schritt mit Adams-Moulton implizit
    def _step_am(self, t_n: float, y_n: float, h: float):
        coeffs = AM_COEFFS[self.order]
        hist_count = max(0, self.order - 1)
        hist_y = [float(self.ys[-1 - i]) for i in range(hist_count)]
        hist_t = [float(self.ts[-1 - i]) for i in range(hist_count)]
        hist_f = [float(self.f(hist_t[i], hist_y[i])) for i in range(hist_count)]
        t_next = t_n + h

        predictor = y_n + h * float(self.f(t_n, y_n))
        if self.order > 1:
            ab_coeffs = AB_COEFFS[self.order - 1]
            predictor = y_n + h * sum(float(ab_coeffs[i]) * hist_f[i] for i in range(self.order - 1))

        y_next_sym = sp.Symbol("y_next")
        expr = y_n + h * (
            float(coeffs[0]) * self.f_expr.subs({self._t_sym: t_next, self._y_sym: y_next_sym})
            + sum(float(coeffs[i + 1]) * hist_f[i] for i in range(self.order - 1))
        )
        eq = sp.Eq(y_next_sym, expr)

        try:
            sol = sp.nsolve(eq.lhs - eq.rhs, y_next_sym, predictor)
            y_next = float(sol)
        except Exception as exc:
            raise ValueError("Adams-Moulton-Schritt konnte nicht gelöst werden.") from exc

        f_next = float(self.f(t_next, y_next))
        weighted_sum = float(coeffs[0]) * f_next + sum(float(coeffs[i + 1]) * hist_f[i] for i in range(self.order - 1))

        values = {
            "method": self.method_name,
            "order": self.order,
            "bootstrap": False,
            "h": h,
            "hist_t": hist_t,
            "hist_y": hist_y,
            "hist_f": hist_f,
            "coeffs": [float(c) for c in coeffs],
            "coeffs_fraction": coeffs,
            "predictor": predictor,
            "t_next": t_next,
            "f_next": f_next,
            "weighted_sum": weighted_sum,
        }
        return y_next, "", values

    # Berechnet den nächsten Schritt mit BDF-Verfahren
    def _step_bdf(self, t_n: float, y_n: float, h: float):
        alphas = BDF_ALPHAS[self.order]
        hist_y = [float(self.ys[-1 - i]) for i in range(self.order)]
        hist_t = [float(self.ts[-1 - i]) for i in range(self.order)]
        t_next = t_n + h
        predictor = y_n + h * float(self.f(t_n, y_n))

        y_next_sym = sp.Symbol("y_next")
        lhs = float(alphas[0]) * y_next_sym + sum(float(alphas[i + 1]) * hist_y[i] for i in range(self.order))
        rhs = h * self.f_expr.subs({self._t_sym: t_next, self._y_sym: y_next_sym})
        eq = sp.Eq(lhs, rhs)

        try:
            sol = sp.nsolve(eq.lhs - eq.rhs, y_next_sym, predictor)
            y_next = float(sol)
        except Exception as exc:
            raise ValueError("BDF-Schritt konnte nicht gelöst werden.") from exc

        f_next = float(self.f(t_next, y_next))

        values = {
            "method": self.method_name,
            "order": self.order,
            "bootstrap": False,
            "h": h,
            "hist_t": hist_t,
            "hist_y": hist_y,
            "alphas": [float(a) for a in alphas],
            "alphas_fraction": alphas,
            "predictor": predictor,
            "t_next": t_next,
            "f_next": f_next,
        }
        return y_next, "", values
