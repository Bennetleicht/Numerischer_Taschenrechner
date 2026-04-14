from __future__ import annotations
from bisection_solver import BisectionSolver

# GUI-Logik für das Bisektionsverfahren
class BisectionMethod:
    title = "Bisektionsverfahren"
    plotter_kind = "ab"

    def __init__(self):  
        self.solver = BisectionSolver()

    # Eingabefelder für die GUI
    def input_fields(self):
        return [
            ("fx", "f(x) =", "", 36),
            ("a", "a =", "", 10),
            ("b", "b =", "", 10),
            ("tol", "Tol =", "0", 10),
        ]
    # Spaltennamen für die Tabelle
    def table_columns(self):
        return ["k", "a", "b", "m", "f(m)"]

    # Logik für Start-Button
    def on_start(self, values, parsers, plotter):
        fx = values["fx"].strip()
        if not fx:
            raise ValueError("Bitte f(x) eingeben.")
        expr, f = parsers["parse_function"](fx)

        # eingelesene Zahlen anpassen
        try:
            a = float(values["a"].replace(",", "."))
            b = float(values["b"].replace(",", "."))
            tol = float(values["tol"].replace(",", "."))
        except Exception:
            raise ValueError("a, b, Tol muessen Zahlen sein.")

        self.solver.start(f, a, b, tol)

        # Plot vorbereiten
        plotter.set_function(f) 
        width = b - a   
        pad = max(5.0, 3.0 * width) #Abstand zum Rand der Plotansicht 
        plotter.set_view(a - pad, b + pad) #Plotansicht so setzen, dass a und b gut sichtbar sind
        plotter.refresh() 

        return "Start ok.", None, False # Statusmeldung, Zeile für Tabelle (None = keine), done=False (noch nicht fertig)

    # Logik für Step-Button
    def on_step(self, plotter):
        status, row, done = self.solver.step()

        if self.solver.f is None:
            return status, row, done

        plotter.set_ab(self.solver.a, self.solver.b)
        plotter.refresh()

        return status, row, done


# startet die Gui
def main():
    from ns_base_gui import GenericMethodGUI
    app = GenericMethodGUI(BisectionMethod())
    app.mainloop()


if __name__ == "__main__":
    main()