from __future__ import annotations
from newton_solver import NewtonSolver

#GUI-Logik für Newton-Verfahren
class NewtonMethod:
    title = "Newton-Verfahren"
    plotter_kind = "newton"

    def __init__(self):
        self.solver = NewtonSolver()

    
    def input_fields(self):
        return [
            ("fx", "f(x) =", "", 36),
            ("x0", "x0 =", "", 12),
            ("tol", "Tol =", "0", 12),
        ]

    def table_columns(self):
        return ["k", "xₖ", "f(xₖ)", "f'(xₖ)", "xₖ₊₁"]


    def on_start(self, values, parsers, plotter):
        fx = values["fx"].strip()
        if not fx:
            raise ValueError("Bitte f(x) eingeben.")
        expr, f, dexpr, df = parsers["parse_function_with_derivative"](fx)

        # eingelesene Zahlen anpassen
        try:
            x0 = float(values["x0"].replace(",", "."))
            tol = float(values["tol"].replace(",", "."))
        except Exception:
            raise ValueError("x0 und Tol muessen Zahlen sein.")

        self.solver.start(f, df, x0, tol)

        # Plot vorbereiten
        y0 = float(self.solver.f(self.solver.x))
        dy0 = float(self.solver.df(self.solver.x))
        plotter.set_function(self.solver.f)
        pad = 5.0
        plotter.set_view(self.solver.x - pad, self.solver.x + pad)
        plotter.set_state(self.solver.x, y0, dy0, None)
        plotter.refresh()

        return "Start ok.", None, False 

    
    def on_step(self, plotter):
        if self.solver.f is None or self.solver.df is None:
            return "Nicht gestartet.", None, True

        xk = float(self.solver.x)
        yk = float(self.solver.f(xk))
        dyk = float(self.solver.df(xk))

        status, row, done = self.solver.step()

        if self.solver.f is None or self.solver.df is None:
            return status, row, done

        xnext = None
        if row is not None and len(row) >= 5 and row[4] != "":
            xnext = float(row[4])

        plotter.set_state(xk, yk, dyk, xnext)
        plotter.refresh()

        return status, row, done


def main():

    from ns_base_gui import GenericMethodGUI
    app = GenericMethodGUI(NewtonMethod())
    app.mainloop()


if __name__ == "__main__":
    main()