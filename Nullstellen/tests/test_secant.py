import pytest
from secant_solver import SecantSolver


def test_start_valid():
    # Testet den normalen Start des Solvers mit gültigen Parametern
    solver = SecantSolver()

    # Start mit f(x) = x² - 4, Startwerten x0=1 und x1=3, Toleranz 0.001
    solver.start(lambda x: x**2 - 4, 1.0, 3.0, 0.001)

    # Prüft ob die Funktion gespeichert wurde
    assert solver.f is not None

    # Prüft ob die Startwerte korrekt gespeichert wurden
    assert solver.x_prev == 1.0
    assert solver.x_cur == 3.0

    # Prüft ob die Toleranz korrekt gesetzt wurde
    assert solver.tol == 0.001

    # Prüft ob der Iterationszähler beim Start auf 0 gesetzt ist
    assert solver.k == 0


def test_start_equal_points():
    # Testet ob gleiche Startwerte abgefangen werden
    solver = SecantSolver()

    # x0 und x1 dürfen nicht gleich sein
    with pytest.raises(ValueError, match="duerfen nicht gleich sein"):
        solver.start(lambda x: x**2 - 4, 2.0, 2.0, 0.001)


def test_start_negative_tol():
    # Testet ob eine negative Toleranz abgefangen wird
    solver = SecantSolver()

    # Negative Toleranz soll einen ValueError auslösen
    with pytest.raises(ValueError, match="Toleranz"):
        solver.start(lambda x: x**2 - 4, 1.0, 3.0, -1.0)


def test_start_rejects_nan_inf():
    # Testet ob ungültige Funktionswerte (Inf oder NaN) erkannt werden
    solver = SecantSolver()

    with pytest.raises(ValueError, match="NaN/Inf"):
        solver.start(lambda x: float("inf") if x == 1.0 else x, 1.0, 3.0, 0.001)


def test_step_without_start():
    # Testet was passiert wenn step() aufgerufen wird ohne vorher start()
    solver = SecantSolver()

    status, row, done = solver.step()

    # Der Solver sollte melden dass er noch nicht gestartet wurde
    assert status == "Nicht gestartet."

    # Es gibt noch keine Tabellenzeile
    assert row is None

    # Algorithmus ist damit beendet
    assert done is True


def test_step_division_by_zero():
    # Testet den Sonderfall f(xk) - f(xk-1) = 0
    solver = SecantSolver()

    # f(1) = -3 und f(-1) = -3, also gleicher Funktionswert
    solver.start(lambda x: x**2 - 4, -1.0, 1.0, 0.001)

    status, row, done = solver.step()

    # Solver muss Abbruch wegen Division durch 0 melden
    assert "Division durch 0" in status

    # Algorithmus beendet sich
    assert done is True

    # Tabellenzeile prüfen
    assert row == (1, "-1", "1", "-3", "-3", "")


def test_step_normal_case():
    # Testet eine normale Sekanteniteration
    solver = SecantSolver()

    solver.start(lambda x: x**2 - 4, 1.0, 3.0, 0.0)

    status, row, done = solver.step()

    # Für x0=1, x1=3:
    # f(1)=-3, f(3)=5
    # x2 = 3 - 5*(3-1)/(5-(-3)) = 3 - 10/8 = 1.75
    assert "Iteration 1" in status

    # Algorithmus läuft weiter
    assert done is False

    # Tabellenzeile prüfen
    assert row == (1, "1", "3", "-3", "5", "1.75")

    # Prüft ob die Werte korrekt weitergeschoben wurden
    assert solver.x_prev == pytest.approx(3.0)
    assert solver.x_cur == pytest.approx(1.75)


def test_step_stops_on_tol():
    # Testet ob der Solver stoppt wenn |x_{k+1} - x_k| <= Tol
    solver = SecantSolver()

    # Lineare Funktion: der erste Sekantenschritt trifft direkt die Nullstelle
    solver.start(lambda x: x - 2, 1.0, 3.0, 1.0)

    status, row, done = solver.step()

    # Solver muss Toleranzabbruch melden
    assert "|Δx|<=Tol" in status

    # Algorithmus beendet sich
    assert done is True

    # Tabellenzeile prüfen
    assert row == (1, "1", "3", "-1", "1", "2")


def test_step_max_iter():
    # Testet Abbruch wenn maximale Iterationen erreicht sind
    solver = SecantSolver()

    solver.start(lambda x: x**2 - 4, 1.0, 3.0, 0.0)

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
    # Testet mehrere Sekanteniterationen hintereinander
    solver = SecantSolver()

    solver.start(lambda x: x**2 - 4, 1.0, 3.0, 0.0)

    # Drei Iterationen durchführen
    status1, row1, done1 = solver.step()
    status2, row2, done2 = solver.step()
    status3, row3, done3 = solver.step()

    # Algorithmus darf noch nicht beendet sein
    assert done1 is False
    assert done2 is False
    assert done3 is False

    # Erwartete Werte:
    # x2 = 1.75
    # x3 = 1.9473684210526316
    # x4 = 2.00355871886121
    assert solver.x_prev == pytest.approx(1.9473684210526316)
    assert solver.x_cur == pytest.approx(2.00355871886121)