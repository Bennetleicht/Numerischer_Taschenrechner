import pytest
from newton_solver import NewtonSolver


def test_start_valid():
    # Testet den normalen Start des Solvers mit gültigen Parametern
    solver = NewtonSolver()

    # Start mit f(x) = x² - 4, f'(x) = 2x, x0 = 3, Toleranz 0.001
    solver.start(lambda x: x**2 - 4, lambda x: 2*x, 3.0, 0.001)

    # Prüft ob Funktion und Ableitung gespeichert wurden
    assert solver.f is not None
    assert solver.df is not None

    # Prüft ob der Startwert korrekt gespeichert wurde
    assert solver.x == 3.0

    # Prüft ob die Toleranz korrekt gesetzt wurde
    assert solver.tol == 0.001

    # Prüft ob der Iterationszähler beim Start auf 0 gesetzt ist
    assert solver.k == 0


def test_start_negative_tol():
    # Testet ob eine negative Toleranz abgefangen wird
    solver = NewtonSolver()

    # Negative Toleranz soll einen ValueError auslösen
    with pytest.raises(ValueError, match="Toleranz"):
        solver.start(lambda x: x**2 - 4, lambda x: 2*x, 3.0, -1.0)


def test_start_rejects_nan_inf_in_f():
    # Testet ob ungültige Funktionswerte beim Start erkannt werden
    solver = NewtonSolver()

    # f(x0) liefert unendlichen Wert
    with pytest.raises(ValueError, match="NaN/Inf"):
        solver.start(lambda x: float("inf"), lambda x: 2*x, 3.0, 0.001)


def test_start_rejects_nan_inf_in_df():
    # Testet ob ungültige Ableitungswerte beim Start erkannt werden
    solver = NewtonSolver()

    # f'(x0) liefert unendlichen Wert
    with pytest.raises(ValueError, match="NaN/Inf"):
        solver.start(lambda x: x**2 - 4, lambda x: float("inf"), 3.0, 0.001)


def test_step_without_start():
    # Testet was passiert wenn step() aufgerufen wird ohne vorher start()
    solver = NewtonSolver()

    status, row, done = solver.step()

    # Der Solver sollte melden dass er noch nicht gestartet wurde
    assert status == "Nicht gestartet."

    # Es gibt noch keine Tabellenzeile
    assert row is None

    # Algorithmus ist damit beendet
    assert done is True


def test_step_nan_inf_during_iteration():
    # Testet Abbruch wenn während der Iteration NaN/Inf in f oder f' auftreten
    solver = NewtonSolver()

    solver.start(lambda x: x, lambda x: 1.0, 1.0, 0.001)

    # Danach künstlich kaputte Funktion einsetzen
    solver.f = lambda x: float("inf")

    status, row, done = solver.step()

    # Solver muss NaN/Inf-Abbruch melden
    assert "NaN/Inf" in status

    # Algorithmus beendet sich
    assert done is True

    # Tabellenzeile prüfen
    assert row == (1, "1", "inf", "1", "")


def test_step_derivative_too_small():
    # Testet Abbruch wenn die Ableitung praktisch 0 ist
    solver = NewtonSolver()

    solver.start(lambda x: x**2 + 1, lambda x: 0.0, 0.0, 0.001)

    status, row, done = solver.step()

    # Solver muss Abbruch wegen zu kleiner Ableitung melden
    assert "f'(xk) ~ 0" in status

    # Algorithmus beendet sich
    assert done is True

    # Tabellenzeile prüfen
    assert row == (1, "0", "1", "0", "")


def test_step_normal_case():
    # Testet eine normale Newton-Iteration
    solver = NewtonSolver()

    # f(x) = x² - 4, f'(x) = 2x, Start bei x0 = 3
    solver.start(lambda x: x**2 - 4, lambda x: 2*x, 3.0, 0.0)

    status, row, done = solver.step()

    # Newton-Schritt:
    # x1 = 3 - (9-4)/6 = 3 - 5/6 = 2.16666666667
    assert "Iteration 1" in status

    # Algorithmus läuft weiter
    assert done is False

    # Tabellenzeile prüfen
    assert row == (1, "3", "5", "6", "2.16666666667")

    # Prüfen ob neuer x-Wert korrekt gespeichert wurde
    assert solver.x == pytest.approx(2.1666666666666665)


def test_step_stops_on_tol():
    # Testet ob der Solver stoppt wenn |x_{k+1} - x_k| <= Tol
    solver = NewtonSolver()

    # Bei x0=2.1 ist der Newton-Schritt schon sehr klein
    solver.start(lambda x: x**2 - 4, lambda x: 2*x, 2.1, 0.2)

    status, row, done = solver.step()

    # Solver muss Toleranzabbruch melden
    assert "|Δx|<=Tol" in status

    # Algorithmus beendet sich
    assert done is True

    # x1 = 2.1 - (2.1² - 4)/4.2 = 2.00238095238
    assert row == (1, "2.1", "0.41", "4.2", "2.00238095238")


def test_step_max_iter():
    # Testet Abbruch wenn maximale Iterationen erreicht sind
    solver = NewtonSolver()

    solver.start(lambda x: x**2 - 4, lambda x: 2*x, 3.0, 0.0)

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
    # Testet mehrere Newton-Iterationen hintereinander
    solver = NewtonSolver()

    solver.start(lambda x: x**2 - 4, lambda x: 2*x, 3.0, 0.0)

    # Drei Iterationen durchführen
    status1, row1, done1 = solver.step()
    status2, row2, done2 = solver.step()
    status3, row3, done3 = solver.step()

    # Algorithmus darf noch nicht beendet sein
    assert done1 is False
    assert done2 is False
    assert done3 is False

    # Erwarteter Wert nach drei Newton-Schritten ab x0=3:
    # x1 = 2.1666666666666665
    # x2 = 2.0064102564102564
    # x3 = 2.0000102400262145
    assert solver.x == pytest.approx(2.0000102400262145)