import pytest

from jacobi_solver import JacobiSolver


# Fuehrt den Solver so lange aus bis er fertig ist
# und sammelt alle erzeugten Schritte.
def run_until_done(solver: JacobiSolver, max_steps: int = 100):
    steps = []

    for _ in range(max_steps):
        step = solver.step()
        steps.append(step)

        # Abbruch sobald der Solver fertig ist
        if step.kind == "done":
            break

    return steps


def test_start_valid():
    # Testet den normalen Start des Solvers mit gueltigen Daten
    solver = JacobiSolver()

    A = [
        [4.0, 1.0],
        [2.0, 3.0],
    ]
    b = [1.0, 2.0]
    x0 = [0.0, 0.0]

    solver.start(A, b, x0, tol=1e-6, safety_limit=50)

    # Prueft ob Dimension und Daten korrekt gespeichert wurden
    assert solver.n == 2
    assert solver.A == A
    assert solver.b == b
    assert solver.x == x0

    # Prueft ob Parameter korrekt gesetzt wurden
    assert solver.tol == 1e-6
    assert solver.safety_limit == 50

    # Prueft ob Solver korrekt initialisiert wurde
    assert solver.iteration == 0
    assert solver.done is False
    assert solver.started is True


def test_start_empty_input():
    # Testet ob leere Eingaben abgefangen werden
    solver = JacobiSolver()

    with pytest.raises(ValueError, match="duerfen nicht leer sein"):
        solver.start([], [], [])


def test_start_non_square_matrix():
    # Testet ob nicht-quadratische Matrizen abgefangen werden
    solver = JacobiSolver()

    A = [
        [1.0, 2.0],
        [3.0],
    ]
    b = [1.0, 2.0]
    x0 = [0.0, 0.0]

    with pytest.raises(ValueError, match="quadratisch"):
        solver.start(A, b, x0)


def test_start_wrong_b_length():
    # Testet ob falsche Laenge von b erkannt wird
    solver = JacobiSolver()

    A = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    b = [1.0]
    x0 = [0.0, 0.0]

    with pytest.raises(ValueError, match="Laenge"):
        solver.start(A, b, x0)


def test_start_wrong_x0_length():
    # Testet ob falsche Laenge von x0 erkannt wird
    solver = JacobiSolver()

    A = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    b = [1.0, 2.0]
    x0 = [0.0]

    with pytest.raises(ValueError, match="Laenge"):
        solver.start(A, b, x0)


def test_start_zero_diagonal_raises():
    # Testet ob ein Nullelement auf der Diagonale abgefangen wird
    solver = JacobiSolver()

    A = [
        [0.0, 1.0],
        [2.0, 3.0],
    ]
    b = [1.0, 2.0]
    x0 = [0.0, 0.0]

    with pytest.raises(ValueError, match="Diagonalelement"):
        solver.start(A, b, x0)


def test_step_without_start():
    # Testet Verhalten wenn step() vor start() aufgerufen wird
    solver = JacobiSolver()

    step = solver.step()

    # Solver sollte direkt done melden
    assert step.kind == "done"
    assert "Fertig" in step.message


def test_first_iteration_values_and_row_details():
    # Testet die erste komplette Jacobi-Iteration inkl. Detaildaten
    solver = JacobiSolver()

    A = [
        [4.0, 1.0],
        [2.0, 3.0],
    ]
    b = [1.0, 2.0]
    x0 = [0.0, 0.0]

    solver.start(A, b, x0)
    step = solver.step()

    # Es muss genau eine Iteration berechnet werden
    assert step.kind == "iter"
    assert step.iteration == 1

    # Erwartete Werte bei Jacobi mit altem Vektor [0, 0]:
    # x1 = (1 - 1*0) / 4 = 0.25
    # x2 = (2 - 2*0) / 3 = 2/3
    assert step.x_new[0] == pytest.approx(0.25)
    assert step.x_new[1] == pytest.approx(2.0 / 3.0)
    assert step.max_diff == pytest.approx(2.0 / 3.0)

    # Es muessen Detaildaten fuer beide Zeilen vorhanden sein
    assert len(step.row_details) == 2

    row0 = step.row_details[0]
    assert row0.row_index == 0
    assert row0.new_value == pytest.approx(0.25)
    assert row0.old_x == [0.0, 0.0]
    assert row0.terms[0] == (1, 1.0, 0.0, 0.0)

    row1 = step.row_details[1]
    assert row1.row_index == 1
    assert row1.new_value == pytest.approx(2.0 / 3.0)
    assert row1.old_x == [0.0, 0.0]
    assert row1.terms[0] == (0, 2.0, 0.0, 0.0)


def test_solver_converges_to_correct_solution():
    # Testet ob der Solver gegen die richtige Loesung konvergiert
    solver = JacobiSolver()

    A = [
        [4.0, 1.0],
        [2.0, 3.0],
    ]
    b = [1.0, 2.0]
    x0 = [0.0, 0.0]

    solver.start(A, b, x0, tol=1e-8, safety_limit=100)
    steps = run_until_done(solver)

    # Solver muss regulaer ueber die Toleranz beenden
    assert steps[-1].kind == "done"
    assert "Toleranz" in steps[-1].message

    # Exakte Loesung des Systems:
    # 4x1 + x2 = 1
    # 2x1 + 3x2 = 2
    # => x = [0.1, 0.6]
    assert solver.x[0] == pytest.approx(0.1, abs=1e-6)
    assert solver.x[1] == pytest.approx(0.6, abs=1e-6)


def test_safety_limit_aborts():
    # Testet Abbruch ueber das Sicherheitslimit
    solver = JacobiSolver()

    A = [
        [4.0, 1.0],
        [2.0, 3.0],
    ]
    b = [1.0, 2.0]
    x0 = [0.0, 0.0]

    solver.start(A, b, x0, tol=0.0, safety_limit=1)

    first = solver.step()
    second = solver.step()

    # Erste Iteration laeuft noch normal
    assert first.kind == "iter"

    # Danach muss wegen Sicherheitslimit abgebrochen werden
    assert second.kind == "done"
    assert "Sicherheitslimit" in second.message


def test_step_after_finish_stays_done():
    # Testet Verhalten wenn step() nach Ende erneut aufgerufen wird
    solver = JacobiSolver()

    A = [
        [4.0, 1.0],
        [2.0, 3.0],
    ]
    b = [1.0, 2.0]
    x0 = [0.0, 0.0]

    solver.start(A, b, x0, tol=1e-8, safety_limit=100)
    run_until_done(solver)

    step = solver.step()

    # Solver bleibt im Zustand done
    assert step.kind == "done"


def test_snapshot_returns_current_state():
    # Testet ob snapshot() den aktuellen Zustand korrekt als Kopie liefert
    solver = JacobiSolver()

    A = [
        [4.0, 1.0],
        [2.0, 3.0],
    ]
    b = [1.0, 2.0]
    x0 = [0.0, 0.0]

    solver.start(A, b, x0, tol=1e-5, safety_limit=12)
    solver.step()

    snap = solver.snapshot()

    assert snap["A"] == A
    assert snap["b"] == b
    assert snap["x"] == solver.x
    assert snap["iteration"] == 1
    assert snap["tol"] == 1e-5
    assert snap["safety_limit"] == 12
    assert snap["done"] is False
    assert snap["started"] is True

    # Snapshot muss eine Kopie sein, keine Referenz auf interne Daten
    snap["A"][0][0] = 999.0
    snap["b"][0] = 999.0
    snap["x"][0] = 999.0

    assert solver.A[0][0] == 4.0
    assert solver.b[0] == 1.0
    assert solver.x[0] != 999.0
