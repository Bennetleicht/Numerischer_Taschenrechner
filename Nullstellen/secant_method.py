from __future__ import annotations
from secant_solver import SecantSolver

#Guilogik für Sekantenverfahren
class SecantMethod:
    title = "Sekantenverfahren"
    plotter_kind = "secant"

    def __init__(self):
        self.solver = SecantSolver()


    def input_fields(self):
        return [
            ("fx", "f(x) =", "", 36),
            ("x0", "x0 =", "", 10),
            ("x1", "x1 =", "", 10),
            ("tol", "Tol =", "0", 10),
        ]

    def table_columns(self):
        return ["k", "xₖ₋₁", "xₖ", "f(xₖ₋₁)", "f(xₖ)", "xₖ₊₁"]


    def on_start(self, values, parsers, plotter):
        fx = values["fx"].strip()
        if not fx:
            raise ValueError("Bitte f(x) eingeben.")
        expr, f = parsers["parse_function"](fx)

        # eingelesene Zahlen anpassen
        try:
            x0 = float(values["x0"].replace(",", "."))
            x1 = float(values["x1"].replace(",", "."))
            tol = float(values["tol"].replace(",", "."))
        except Exception:
            raise ValueError("x0, x1, Tol muessen Zahlen sein.")

        self.solver.start(f, x0, x1, tol)

        # Plot vorbereiten
        plotter.set_function(self.solver.f)
        xmin = min(self.solver.x_prev, self.solver.x_cur)
        xmax = max(self.solver.x_prev, self.solver.x_cur)
        width = xmax - xmin
        pad = max(5.0, 3.0 * width)
        plotter.set_view(xmin - pad, xmax + pad)
        plotter.set_state(self.solver.x_prev, self.solver.x_cur, None)
        plotter.refresh()

        return "Start ok.", None, False

    def on_step(self, plotter):
        status, row, done = self.solver.step()

        if self.solver.f is None:
            return status, row, done

        plot_x = None
        if row is not None and len(row) >= 6 and row[5] != "":
            plot_x = float(row[5])

        plotter.set_state(self.solver.x_prev, self.solver.x_cur, plot_x)
        plotter.refresh()

        return status, row, done



def main():
    from ns_base_gui import GenericMethodGUI
    app = GenericMethodGUI(SecantMethod())
    app.mainloop()


if __name__ == "__main__":
    main()