import pytest
from heron_solver import HeronSolver


def test_start_valid():
    # Testet den normalen Start des Solvers mit gültigen Parametern
    solver = HeronSolver()

    # Start mit S = 9, x0 = 3, Toleranz 0.001
    solver.start(9.0, 3.0, 0.001)

    # Prüft ob S korrekt gespeichert wurde
    assert solver.S == 9.0

    # Prüft ob der Startwert korrekt gespeichert wurde
    assert solver.x == 3.0

    # Prüft ob die Toleranz korrekt gesetzt wurde
    assert solver.tol == 0.001

    # Prüft ob der Iterationszähler beim Start auf 0 gesetzt ist
    assert solver.k == 0


def test_start_negative_S():
    # Testet ob negatives S abgefangen wird
    solver = HeronSolver()

    # Negative Zahl unter der Wurzel ist hier nicht erlaubt
    with pytest.raises(ValueError, match="S muss >= 0"):
        solver.start(-1.0, 1.0, 0.001)


def test_start_negative_tol():
    # Testet ob eine negative Toleranz abgefangen wird
    solver = HeronSolver()

    # Negative Toleranz soll einen ValueError auslösen
    with pytest.raises(ValueError, match="Toleranz"):
        solver.start(9.0, 3.0, -1.0)


def test_start_zero_x0_for_nonzero_S():
    # Testet ob x0 = 0 bei S != 0 abgefangen wird
    solver = HeronSolver()

    # Wegen Division S/x0 darf x0 hier nicht 0 sein
    with pytest.raises(ValueError, match="x0 darf nicht 0 sein"):
        solver.start(9.0, 0.0, 0.001)


def test_step_without_start():
    # Testet was passiert wenn step() aufgerufen wird ohne vorher start()
    solver = HeronSolver()

    status, row, done = solver.step()

    # Der Solver sollte melden dass er noch nicht gestartet wurde
    assert status == "Nicht gestartet."

    # Es gibt noch keine Tabellenzeile
    assert row is None

    # Algorithmus ist damit beendet
    assert done is True


def test_step_S_zero():
    # Testet den Sonderfall sqrt(0) = 0
    solver = HeronSolver()

    solver.start(0.0, 5.0, 0.001)

    status, row, done = solver.step()

    # Solver muss direkt 0 als Ergebnis liefern
    assert "sqrt(S)=0" in status

    # Algorithmus beendet sich sofort
    assert done is True

    # Tabellenzeile prüfen
    assert row == (1, "5", "0", "5.000e+00")

    # x muss jetzt 0 sein
    assert solver.x == 0.0


def test_step_normal_case():
    # Testet eine normale Heron-Iteration
    solver = HeronSolver()

    # S = 9, x0 = 5
    solver.start(9.0, 5.0, 0.0)

    status, row, done = solver.step()

    # x1 = 0.5 * (5 + 9/5) = 3.4
    assert "Iteration 1" in status

    # Algorithmus läuft weiter
    assert done is False

    # Tabellenzeile prüfen
    assert row == (1, "5", "3.4", "1.600e+00")

    # Neuer x-Wert muss gespeichert sein
    assert solver.x == pytest.approx(3.4)


def test_step_stops_on_tol():
    # Testet ob der Solver stoppt wenn |x_{k+1} - x_k| <= Tol
    solver = HeronSolver()

    # Bei x0 = 3 und S = 9 ist man schon exakt auf der Wurzel
    solver.start(9.0, 3.0, 0.0)

    status, row, done = solver.step()

    # x1 = 3, also Δx = 0
    assert "|Δx|<=Tol" in status

    # Algorithmus beendet sich
    assert done is True

    # Tabellenzeile prüfen
    assert row == (1, "3", "3", "0.000e+00")


def test_step_max_iter():
    # Testet Abbruch wenn maximale Iterationen erreicht sind
    solver = HeronSolver()

    solver.start(9.0, 5.0, 0.0)

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
    # Testet mehrere Heron-Iterationen hintereinander
    solver = HeronSolver()

    solver.start(9.0, 5.0, 0.0)

    # Drei Iterationen durchführen
    status1, row1, done1 = solver.step()
    status2, row2, done2 = solver.step()
    status3, row3, done3 = solver.step()

    # Algorithmus darf noch nicht beendet sein
    assert done1 is False
    assert done2 is False
    assert done3 is False

    # Erwartete Werte:
    # x1 = 3.4
    # x2 = 3.023529411764706
    # x3 = 3.00009155413138
    assert solver.x == pytest.approx(3.00009155413138)