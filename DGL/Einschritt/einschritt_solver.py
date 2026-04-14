from __future__ import annotations
from dataclasses import dataclass, field
import sympy as sp

METHOD_EE = "Explizites Euler-Verfahren"
METHOD_HE = "Heun-Verfahren"
METHOD_ME = "Modifiziertes Euler-Verfahren"
METHOD_RK = "Runge-Kutta-Verfahren (4)"
METHOD_IE = "Implizites Euler-Verfahren"

METHODS = [METHOD_EE, METHOD_HE, METHOD_ME, METHOD_RK, METHOD_IE]




@dataclass
class StepResult:
    n: int
    t_n: float
    y_n: float
    t_next: float
    y_next: float
    text: str = ""
    values: dict = field(default_factory=dict)


class EinschrittSolver:
    def __init__(self, method_name: str, f, f_expr, a: float, b: float, y0: float, h: float):
        self.method_name = method_name
        self.f = f
        self.f_expr = f_expr

        self.a = float(a)
        self.b = float(b)
        self.y0 = float(y0)
        self.h = float(h)

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

    def current_step_size(self) -> float:
        return min(self.h, self.b - self.ts[-1])

    #Gibt und teilt die verfügbaren Verfahren
    def step(self) -> StepResult:
        if self.is_finished:
            raise RuntimeError("Intervall bereits vollständig berechnet.")

        t_n = float(self.ts[-1])
        y_n = float(self.ys[-1])
        h_n = float(self.current_step_size())
        t_next = t_n + h_n

        if self.method_name == METHOD_EE:
            y_next, text, values = self._step_explicit_euler(t_n, y_n, h_n)

        elif self.method_name == METHOD_HE:
            y_next, text, values = self._step_heun(t_n, y_n, h_n)

        elif self.method_name == METHOD_ME:
            y_next, text, values = self._step_modified_euler(t_n, y_n, h_n)

        elif self.method_name == METHOD_RK:
            y_next, text, values = self._step_rk4(t_n, y_n, h_n)

        elif self.method_name == METHOD_IE:
            y_next, text, values = self._step_implicit_euler(t_n, y_n, h_n)

        else:
            raise ValueError(f"Unbekanntes Verfahren: {self.method_name}")

        self.ts.append(float(t_next))
        self.ys.append(float(y_next))

        result = StepResult(
            n=self.n,
            t_n=t_n,
            y_n=y_n,
            t_next=t_next,
            y_next=float(y_next),
            text=text,
            values=values,
        )
        self.n += 1
        return result

    #bereitet die Daten für die Anzeige auf für explicit und berechnet die Werte für die Anzeige
    def _step_explicit_euler(self, t_n: float, y_n: float, h: float):
        f_n = float(self.f(t_n, y_n))
        y_next = y_n + h * f_n

        text = (
            f"Schritt {self.n + 1}\n"
            f"t_{self.n} = {t_n:.8g}\n"
            f"y_{self.n} = {y_n:.8g}\n"
            f"h = {h:.8g}\n\n"
            f"Formel:\n"
            f"y_{self.n+1} = y_{self.n} + h·f(t_{self.n}, y_{self.n})\n\n"
            f"Einsetzen:\n"
            f"f({t_n:.8g}, {y_n:.8g}) = {f_n:.8g}\n"
            f"y_{self.n+1} = {y_n:.8g} + {h:.8g} · {f_n:.8g}\n"
            f"y_{self.n+1} = {y_next:.8g}\n"
            f"t_{self.n+1} = {t_n + h:.8g}\n"
        )

        values = {
            "method": METHOD_EE,
            "h": h,
            "yprime_n": f_n,
        }
        return y_next, text, values

    ##bereitet die Daten für die Anzeige auf für Heun und berechnet die Werte für die Anzeige
    def _step_heun(self, t_n: float, y_n: float, h: float):
        k1 = float(self.f(t_n, y_n))
        predictor = y_n + h * k1
        k2 = float(self.f(t_n + h, predictor))
        y_next = y_n + h * 0.5 * (k1 + k2)

        text = (
            f"Schritt {self.n + 1}\n"
            f"t_{self.n} = {t_n:.8g}\n"
            f"y_{self.n} = {y_n:.8g}\n"
            f"h = {h:.8g}\n\n"
            f"k₁ = f(t_{self.n}, y_{self.n}) = f({t_n:.8g}, {y_n:.8g}) = {k1:.8g}\n"
            f"y* = y_{self.n} + h·k₁ = {y_n:.8g} + {h:.8g}·{k1:.8g} = {predictor:.8g}\n"
            f"k₂ = f(t_{self.n}+h, y*) = f({t_n+h:.8g}, {predictor:.8g}) = {k2:.8g}\n\n"
            f"y_{self.n+1} = y_{self.n} + h/2 · (k₁ + k₂)\n"
            f"y_{self.n+1} = {y_n:.8g} + {h:.8g}/2 · ({k1:.8g} + {k2:.8g})\n"
            f"y_{self.n+1} = {y_next:.8g}\n"
            f"t_{self.n+1} = {t_n + h:.8g}\n"
        )

        values = {
            "method": METHOD_HE,
            "h": h,
            "k1": k1,
            "y_star": predictor,
            "k2": k2,
        }
        return y_next, text, values

    #bereitet die Daten für die Anzeige auf für modifizierter Euler und berechnet die Werte für die Anzeige
    def _step_modified_euler(self, t_n: float, y_n: float, h: float):
        k1 = float(self.f(t_n, y_n))
        t_mid = t_n + h / 2.0
        y_mid = y_n + h / 2.0 * k1
        f_mid = float(self.f(t_mid, y_mid))
        y_next = y_n + h * f_mid

        text = (
            f"Schritt {self.n + 1}\n"
            f"t_{self.n} = {t_n:.8g}\n"
            f"y_{self.n} = {y_n:.8g}\n"
            f"h = {h:.8g}\n\n"
            f"k₁ = f(t_{self.n}, y_{self.n}) = f({t_n:.8g}, {y_n:.8g}) = {k1:.8g}\n"
            f"t_m = t_{self.n} + h/2 = {t_mid:.8g}\n"
            f"y_m = y_{self.n} + h/2·k₁ = {y_mid:.8g}\n"
            f"f(t_m, y_m) = f({t_mid:.8g}, {y_mid:.8g}) = {f_mid:.8g}\n\n"
            f"y_{self.n+1} = y_{self.n} + h·f(t_m, y_m)\n"
            f"y_{self.n+1} = {y_n:.8g} + {h:.8g} · {f_mid:.8g}\n"
            f"y_{self.n+1} = {y_next:.8g}\n"
            f"t_{self.n+1} = {t_n + h:.8g}\n"
        )

        values = {
            "method": METHOD_ME,
            "h": h,
            "k1": k1,
            "t_mid": t_mid,
            "y_mid": y_mid,
            "yprime_mid": f_mid,
        }
        return y_next, text, values

    #bereitet die Daten für die Anzeige auf für RK4 und berechnet die Werte für die Anzeige
    def _step_rk4(self, t_n: float, y_n: float, h: float):
        k1 = float(self.f(t_n, y_n))
        k2 = float(self.f(t_n + h / 2.0, y_n + h / 2.0 * k1))
        k3 = float(self.f(t_n + h / 2.0, y_n + h / 2.0 * k2))
        k4 = float(self.f(t_n + h, y_n + h * k3))

        y_next = y_n + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        text = (
            f"Schritt {self.n + 1}\n"
            f"t_{self.n} = {t_n:.8g}\n"
            f"y_{self.n} = {y_n:.8g}\n"
            f"h = {h:.8g}\n\n"
            f"k₁ = f({t_n:.8g}, {y_n:.8g}) = {k1:.8g}\n"
            f"k₂ = f({t_n+h/2.0:.8g}, {y_n + h/2.0*k1:.8g}) = {k2:.8g}\n"
            f"k₃ = f({t_n+h/2.0:.8g}, {y_n + h/2.0*k2:.8g}) = {k3:.8g}\n"
            f"k₄ = f({t_n+h:.8g}, {y_n + h*k3:.8g}) = {k4:.8g}\n\n"
            f"y_{self.n+1} = y_{self.n} + h/6 · (k₁ + 2k₂ + 2k₃ + k₄)\n"
            f"y_{self.n+1} = {y_n:.8g} + {h:.8g}/6 · ({k1:.8g} + 2·{k2:.8g} + 2·{k3:.8g} + {k4:.8g})\n"
            f"y_{self.n+1} = {y_next:.8g}\n"
            f"t_{self.n+1} = {t_n + h:.8g}\n"
        )

        values = {
            "method": METHOD_RK,
            "h": h,
            "k1": k1,
            "k2": k2,
            "k3": k3,
            "k4": k4,
        }
        return y_next, text, values

    #bereitet die Daten für die Anzeige auf für impliziter Euler und berechnet die Werte für die Anzeige
    def _step_implicit_euler(self, t_n: float, y_n: float, h: float):
        y_next_sym = sp.Symbol("y_next")
        t_next = t_n + h

        eq = sp.Eq(
            y_next_sym,
            y_n + h * self.f_expr.subs({
                self._t_sym: t_next,
                self._y_sym: y_next_sym
            })
        )

        start_guess = y_n + h * float(self.f(t_n, y_n))

        try:
            sol = sp.nsolve(eq.lhs - eq.rhs, y_next_sym, start_guess)
            y_next = float(sol)
        except Exception as exc:
            raise ValueError(
                "Impliziter Schritt konnte nicht gelöst werden. "
                "Versuche andere Startwerte oder eine einfachere DGL."
            ) from exc

        text = (
            f"Schritt {self.n + 1}\n"
            f"t_{self.n} = {t_n:.8g}\n"
            f"y_{self.n} = {y_n:.8g}\n"
            f"h = {h:.8g}\n\n"
            f"Implizite Formel:\n"
            f"y_{self.n+1} = y_{self.n} + h·f(t_{self.n+1}, y_{self.n+1})\n\n"
            f"Mit t_{self.n+1} = {t_next:.8g} ergibt sich die Gleichung:\n"
            f"{sp.sstr(eq.lhs)} = {sp.sstr(eq.rhs)}\n\n"
            f"Numerisch gelöst mit Startwert {start_guess:.8g}:\n"
            f"y_{self.n+1} = {y_next:.8g}\n"
            f"t_{self.n+1} = {t_next:.8g}\n"
        )

        values = {
            "method": METHOD_IE,
            "h": h,
            "start_guess": start_guess,
            "equation": sp.sstr(eq.lhs - eq.rhs),
        }
        return y_next, text, values