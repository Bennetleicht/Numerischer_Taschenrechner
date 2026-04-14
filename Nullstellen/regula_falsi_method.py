from __future__ import annotations
from regula_falsi_solver import RegulaFalsiSolver

#Gui Logik für Regula Falsi Methode
class RegulaFalsiMethod:
    title = "Regula Falsi"
    plotter_kind = "regula"

    def __init__(self):
        self.solver = RegulaFalsiSolver()

    
    def input_fields(self):
        return [
            ("fx", "f(x) =", "", 36),
            ("a", "a =", "", 10),
            ("b", "b =", "", 10),
            ("tol", "Tol =", "0", 10),
        ]

    def table_columns(self):
        return ["k", "a", "b", "f(a)", "f(b)", "x", "f(x)"]


    def on_start(self, values, parsers, plotter):
        fx = values["fx"].strip()
        if not fx:
            raise ValueError("Bitte f(x) eingeben.")
        expr, f = parsers["parse_function"](fx)

        try:
            a = float(values["a"].replace(",", "."))
            b = float(values["b"].replace(",", "."))
            tol = float(values["tol"].replace(",", "."))
        except Exception:
            raise ValueError("a, b, Tol muessen Zahlen sein.")

        self.solver.start(f, a, b, tol)

        # Plot vorbereiten
        plotter.set_function(self.solver.f)
        width = self.solver.b - self.solver.a
        pad = max(5.0, 3.0 * width)
        plotter.set_view(self.solver.a - pad, self.solver.b + pad)
        plotter.refresh()

        return "Start ok.", None, False


    def on_step(self, plotter):
        status, row, done = self.solver.step()

        if self.solver.f is None:
            return status, row, done

        plot_x = self.solver.last_x
        if row is not None and len(row) >= 6 and row[5] != "":
            plot_x = float(row[5])

        plotter.set_state(self.solver.a, self.solver.b, plot_x)
        plotter.refresh()

        return status, row, done



def main():
    from ns_base_gui import GenericMethodGUI
    app = GenericMethodGUI(RegulaFalsiMethod())
    app.mainloop()


if __name__ == "__main__":
    main()