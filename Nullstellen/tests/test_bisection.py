import pytest
from bisection_solver import BisectionSolver


def test_start_valid():
    # Testet den normalen Start des Solvers mit gültigen Parametern
    solver = BisectionSolver()

    # Start mit f(x) = x² - 4, Intervall [0,3], Toleranz 0.001
    solver.start(lambda x: x**2 - 4, 0.0, 3.0, 0.001)

    # Prüft ob die Funktion gespeichert wurde
    assert solver.f is not None

    # Prüft ob das Startintervall korrekt gespeichert wurde
    assert solver.a == 0.0
    assert solver.b == 3.0

    # Prüft ob die Toleranz korrekt gesetzt wurde
    assert solver.tol == 0.001

    # Prüft ob der Iterationszähler beim Start auf 0 gesetzt ist
    assert solver.k == 0


def test_start_invalid_interval():
    # Testet, ob ein Fehler geworfen wird wenn a >= b
    solver = BisectionSolver()

    # Erwartet ValueError weil das Intervall falsch herum ist
    with pytest.raises(ValueError, match="a < b"):
        solver.start(lambda x: x**2 - 4, 3.0, 0.0, 0.001)


def test_start_negative_tol():
    # Testet ob eine negative Toleranz abgefangen wird
    solver = BisectionSolver()

    # Negative Toleranz soll einen ValueError auslösen
    with pytest.raises(ValueError, match="Toleranz"):
        solver.start(lambda x: x**2 - 4, 0.0, 3.0, -1.0)


def test_start_requires_sign_change():
    # Testet ob ein Vorzeichenwechsel im Intervall verlangt wird
    solver = BisectionSolver()

    # f(x) = x² + 1 hat KEINE Nullstelle → kein Vorzeichenwechsel
    # deshalb muss der Solver einen Fehler werfen
    with pytest.raises(ValueError, match="Vorzeichenwechsel"):
        solver.start(lambda x: x**2 + 1, 0.0, 3.0, 0.001)


def test_start_rejects_nan_inf():
    # Testet ob ungültige Funktionswerte (Inf oder NaN) erkannt werden
    solver = BisectionSolver()

    # Funktion liefert für x=0 unendlichen Wert
    # Das muss vom Solver abgefangen werden
    with pytest.raises(ValueError, match="NaN/Inf"):
        solver.start(lambda x: float("inf") if x == 0.0 else x, 0.0, 1.0, 0.001)


def test_step_without_start():
    # Testet was passiert wenn step() aufgerufen wird ohne vorher start()
    solver = BisectionSolver()

    status, row, done = solver.step()

    # Der Solver sollte melden dass er noch nicht gestartet wurde
    assert status == "Nicht gestartet."

    # Es gibt noch keine Tabellenzeile
    assert row is None

    # Algorithmus ist damit beendet
    assert done is True


def test_step_left_boundary_root():
    # Testet den Spezialfall: linke Intervallgrenze ist bereits Nullstelle
    solver = BisectionSolver()

    # f(x) = x - 1 → Nullstelle bei x = 1
    solver.start(lambda x: x - 1, 1.0, 3.0, 0.001)

    status, row, done = solver.step()

    # Solver soll erkennen dass a bereits Nullstelle ist
    assert "a ist Nullstelle" in status

    # Algorithmus beendet sich sofort
    assert done is True

    # Tabellenzeile muss die richtige Ausgabe enthalten
    assert row == (1, "1", "3", "1", "0")


def test_step_right_boundary_root():
    # Testet den Spezialfall: rechte Intervallgrenze ist Nullstelle
    solver = BisectionSolver()

    # f(x) = x - 3 → Nullstelle bei x = 3
    solver.start(lambda x: x - 3, 1.0, 3.0, 0.001)

    status, row, done = solver.step()

    # Solver erkennt Nullstelle an b
    assert "b ist Nullstelle" in status

    # Algorithmus beendet sich sofort
    assert done is True

    # Tabellenzeile prüfen
    assert row == (1, "1", "3", "3", "0")


def test_step_normal_case():
    # Testet eine normale Bisektionsiteration
    solver = BisectionSolver()

    # Intervall enthält Nullstelle bei x=2
    solver.start(lambda x: x**2 - 4, 0.0, 3.0, 0.0)

    status, row, done = solver.step()

    # Status muss Iteration anzeigen
    assert "Iteration 1" in status

    # Algorithmus läuft weiter
    assert done is False

    # Tabellenzeile prüfen
    assert row == (1, "0", "3", "1.5", "-1.75")

    # Prüfen ob Intervall korrekt halbiert wurde
    assert solver.a == pytest.approx(1.5)
    assert solver.b == pytest.approx(3.0)


def test_step_stops_on_tol():
    # Testet ob der Solver stoppt wenn Intervallbreite <= Toleranz
    solver = BisectionSolver()

    # Intervallbreite = 0.5, Toleranz = 0.5
    solver.start(lambda x: x**2 - 4, 1.6, 2.1, 0.5)

    status, row, done = solver.step()

    # Solver muss Toleranzabbruch melden
    assert "Intervallbreite <= Tol" in status

    # Algorithmus beendet sich
    assert done is True

    # Tabellenzeile prüfen
    assert row == (1, "1.6", "2.1", "1.85", "-0.5775")


def test_step_max_iter():
    # Testet Abbruch wenn maximale Iterationen erreicht sind
    solver = BisectionSolver()

    solver.start(lambda x: x**2 - 4, 0.0, 3.0, 0.0)

    # Iterationszähler künstlich auf Maximum setzen
    solver.k = solver.max_iter

    status, row, done = solver.step()

    # Solver muss Hard-Limit melden
    assert "Hard-Limit" in status

    # Keine Tabellenzeile mehr
    assert row is None

    # Algorithmus beendet
    assert done is True


def test_multiple_steps():
    # Testet mehrere Iterationen hintereinander
    solver = BisectionSolver()

    solver.start(lambda x: x**2 - 4, 0.0, 3.0, 0.0)

    # Drei Iterationen durchführen
    status1, row1, done1 = solver.step()
    status2, row2, done2 = solver.step()
    status3, row3, done3 = solver.step()

    # Algorithmus darf noch nicht beendet sein
    assert done1 is False
    assert done2 is False
    assert done3 is False

    # Prüft ob das Intervall korrekt immer weiter eingeengt wird
    assert solver.a == pytest.approx(1.875)
    assert solver.b == pytest.approx(2.25)