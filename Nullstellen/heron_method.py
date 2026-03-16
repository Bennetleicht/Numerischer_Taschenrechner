from __future__ import annotations
from heron_solver import HeronSolver

#Gui Logik für Heron-Verfahren
class HeronMethod:
    title = "Heron-Verfahren"
    plotter_kind = "heron"

    def __init__(self):
        self.solver = HeronSolver()

    def input_fields(self):
        return [
            ("S", "S =", "", 12),
            ("x0", "x0 =", "", 12),
            ("tol", "Tol =", "0", 12),
        ]

    def table_columns(self):
        return ["k", "xₖ", "xₖ₊₁", "|Δx|"]

    def on_start(self, values, parsers, plotter):
        try:
            S = float(values["S"].replace(",", "."))
            x0 = float(values["x0"].replace(",", "."))
            tol = float(values["tol"].replace(",", "."))
        except Exception:
            raise ValueError("S, x0, Tol muessen Zahlen sein.")

        self.solver.start(S, x0, tol)

        # Plot vorbereiten
        plotter.set_S(self.solver.S)

        return "Start ok.", None, False


    def on_step(self, plotter):
        status, row, done = self.solver.step()
        return status, row, done


def main():
    from gui_app import GenericMethodGUI
    app = GenericMethodGUI(HeronMethod())
    app.mainloop()


if __name__ == "__main__":
    main()