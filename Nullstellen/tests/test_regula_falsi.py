import pytest
from regula_falsi_solver import RegulaFalsiSolver


def test_start_valid():
    # Testet den normalen Start des Solvers mit gültigen Parametern
    solver = RegulaFalsiSolver()

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

    # Prüft ob last_x beim Start noch leer ist
    assert solver.last_x is None


def test_start_invalid_interval():
    # Testet, ob ein Fehler geworfen wird wenn a >= b
    solver = RegulaFalsiSolver()

    # Erwartet ValueError weil das Intervall falsch herum ist
    with pytest.raises(ValueError, match="a < b"):
        solver.start(lambda x: x**2 - 4, 3.0, 0.0, 0.001)


def test_start_negative_tol():
    # Testet ob eine negative Toleranz abgefangen wird
    solver = RegulaFalsiSolver()

    # Negative Toleranz soll einen ValueError auslösen
    with pytest.raises(ValueError, match="Toleranz"):
        solver.start(lambda x: x**2 - 4, 0.0, 3.0, -1.0)


def test_start_requires_sign_change():
    # Testet ob ein Vorzeichenwechsel im Intervall verlangt wird
    solver = RegulaFalsiSolver()

    # f(x) = x² + 1 hat keine Nullstelle, also kein Vorzeichenwechsel
    with pytest.raises(ValueError, match="Vorzeichenwechsel"):
        solver.start(lambda x: x**2 + 1, 0.0, 3.0, 0.001)


def test_start_rejects_nan_inf():
    # Testet ob ungültige Funktionswerte (Inf oder NaN) erkannt werden
    solver = RegulaFalsiSolver()

    with pytest.raises(ValueError, match="NaN/Inf"):
        solver.start(lambda x: float("inf") if x == 0.0 else x, 0.0, 1.0, 0.001)


def test_step_without_start():
    # Testet was passiert wenn step() aufgerufen wird ohne vorher start()
    solver = RegulaFalsiSolver()

    status, row, done = solver.step()

    # Der Solver sollte melden dass er noch nicht gestartet wurde
    assert status == "Nicht gestartet."

    # Es gibt noch keine Tabellenzeile
    assert row is None

    # Algorithmus ist damit beendet
    assert done is True


def test_step_left_boundary_root():
    # Testet den Spezialfall: linke Intervallgrenze ist bereits Nullstelle
    solver = RegulaFalsiSolver()

    # f(x) = x - 1 -> Nullstelle bei x = 1
    solver.start(lambda x: x - 1, 1.0, 3.0, 0.001)

    status, row, done = solver.step()

    # Solver soll erkennen dass a bereits Nullstelle ist
    assert "a ist Nullstelle" in status

    # Algorithmus beendet sich sofort
    assert done is True

    # Tabellenzeile prüfen
    assert row == (1, "1", "3", "0", "2", "1", "0")


def test_step_right_boundary_root():
    # Testet den Spezialfall: rechte Intervallgrenze ist bereits Nullstelle
    solver = RegulaFalsiSolver()

    # f(x) = x - 3 -> Nullstelle bei x = 3
    solver.start(lambda x: x - 3, 1.0, 3.0, 0.001)

    status, row, done = solver.step()

    # Solver soll erkennen dass b bereits Nullstelle ist
    assert "b ist Nullstelle" in status

    # Algorithmus beendet sich sofort
    assert done is True

    # Tabellenzeile prüfen
    assert row == (1, "1", "3", "-2", "0", "3", "0")


def test_step_division_by_zero():
    # Testet den Sonderfall f(b) - f(a) = 0
    solver = RegulaFalsiSolver()

    # Start mit gültiger Funktion
    solver.start(lambda x: x**2 - 4, 0.0, 3.0, 0.001)

    # Danach künstlich Zustand so setzen, dass fa = fb
    solver.a = -3.0
    solver.b = 3.0

    status, row, done = solver.step()

    # Solver muss Abbruch wegen Division durch 0 melden
    assert "Division durch 0" in status

    # Algorithmus beendet sich
    assert done is True

    # Tabellenzeile prüfen
    assert row == (1, "-3", "3", "5", "5", "", "")


def test_step_normal_case():
    # Testet eine normale Regula-Falsi-Iteration
    solver = RegulaFalsiSolver()

    solver.start(lambda x: x**2 - 4, 0.0, 3.0, 0.0)

    status, row, done = solver.step()

    # Für a=0, b=3, fa=-4, fb=5 gilt:
    # x_new = (a*fb - b*fa) / (fb-fa) = (0*5 - 3*(-4)) / (5-(-4)) = 12/9 = 1.33333333333
    # f(x_new) = 1.33333333333² - 4 = -2.22222222222
    assert "Iteration 1" in status

    # Algorithmus läuft weiter
    assert done is False

    # Tabellenzeile prüfen
    assert row == (1, "0", "3", "-4", "5", "1.333333333", "-2.222222222")

    # Prüft ob neues Intervall korrekt gesetzt wurde
    assert solver.a == pytest.approx(1.3333333333333333)
    assert solver.b == pytest.approx(3.0)

    # Prüft ob letzter x-Wert gespeichert wurde
    assert solver.last_x == pytest.approx(1.3333333333333333)


def test_step_stops_on_tol():
    # Testet ob der Solver stoppt wenn |f(x)| <= Toleranz
    solver = RegulaFalsiSolver()

    # Lineare Funktion: der erste Regula-Falsi-Schritt trifft direkt die Nullstelle
    solver.start(lambda x: x - 2, 1.0, 3.0, 0.001)

    status, row, done = solver.step()

    # Solver muss Toleranzabbruch melden
    assert "|f(x)|<=Tol" in status

    # Algorithmus beendet sich
    assert done is True

    # Tabellenzeile prüfen
    assert row == (1, "1", "3", "-1", "1", "2", "0")


def test_step_max_iter():
    # Testet Abbruch wenn maximale Iterationen erreicht sind
    solver = RegulaFalsiSolver()

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
    # Testet mehrere Regula-Falsi-Iterationen hintereinander
    solver = RegulaFalsiSolver()

    solver.start(lambda x: x**2 - 4, 0.0, 3.0, 0.0)

    # Drei Iterationen durchführen
    status1, row1, done1 = solver.step()
    status2, row2, done2 = solver.step()
    status3, row3, done3 = solver.step()

    # Algorithmus darf noch nicht beendet sein
    assert done1 is False
    assert done2 is False
    assert done3 is False

    # Erwartete Werte:
    # x1 = 1.3333333333333333
    # x2 = 1.8461538461538463
    # x3 = 1.9682539682539684
    assert solver.a == pytest.approx(1.9682539682539684)
    assert solver.b == pytest.approx(3.0)
    assert solver.last_x == pytest.approx(1.9682539682539684)