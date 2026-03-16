import pytest
from inkrementell.lr_solver import LRSolver


# Führt den Solver bis zum Abschluss aus
def run_until_done(solver: LRSolver, max_steps: int = 100):
    steps = []

    for _ in range(max_steps):
        step = solver.next_step()
        steps.append(step)

        # Abbruch wenn Solver fertig ist
        if step.kind == "done":
            break

    return steps


def test_solver_initializes_correctly():
    # Testet korrekte Initialisierung des Solvers

    A = [
        [2.0, 1.0],
        [4.0, 3.0],
    ]
    b = [1.0, 2.0]

    solver = LRSolver(A, b)

    # Prüft Dimension
    assert solver.n == 2

    # Prüft ob A korrekt als R gespeichert wurde
    assert solver.R == A

    # Prüft ob b korrekt gespeichert wurde
    assert solver.b == b

    # Prüft Initialisierung von L
    assert solver.L == [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    # Prüft Initialisierung von y und x
    assert solver.y == [0.0, 0.0]
    assert solver.x == [0.0, 0.0]

    # Prüft Startphase
    assert solver.phase == "decomp"

    # Solver darf noch nicht fertig sein
    assert solver.done is False


def test_simple_system_solution():
    # Testet Lösung eines einfachen linearen Gleichungssystems

    A = [
        [2.0, 1.0],
        [4.0, 3.0],
    ]
    b = [1.0, 2.0]

    solver = LRSolver(A, b)

    steps = run_until_done(solver)

    # Prüft ob Solver korrekt beendet wurde
    assert steps[-1].kind == "done"

    # Prüft ob die Lösung korrekt berechnet wurde
    x = solver.get_solution()
    assert x[0] == pytest.approx(0.5)
    assert x[1] == pytest.approx(0.0)


def test_first_step_is_lr_update():
    # Testet ob der erste Schritt ein LR-Zerlegungsschritt ist

    A = [
        [2.0, 1.0],
        [4.0, 3.0],
    ]
    b = [1.0, 2.0]

    solver = LRSolver(A, b)

    step = solver.next_step()

    assert step.kind == "lr_update"
    assert step.pivot == (0, 0)
    assert step.target_row == 1

    # Prüft ob die Pivotinformation im Text steht
    assert "Pivot" in step.message


def test_forward_and_back_sub_steps_occur():
    # Testet ob Vorwärts- und Rückwärtseinsetzen ausgeführt werden

    A = [
        [2.0, 1.0],
        [4.0, 3.0],
    ]
    b = [1.0, 2.0]

    solver = LRSolver(A, b)

    steps = run_until_done(solver)

    kinds = [s.kind for s in steps]

    assert "forward_sub" in kinds
    assert "back_sub" in kinds


def test_zero_pivot_aborts_in_decomposition():
    # Testet Abbruch wenn Pivot in R gleich 0 ist

    A = [
        [0.0, 1.0],
        [2.0, 3.0],
    ]
    b = [1.0, 2.0]

    solver = LRSolver(A, b)

    step = solver.next_step()

    assert step.kind == "done"
    assert "Pivot in R ist 0" in step.message


def test_non_square_matrix_raises():
    # Testet ob nicht-quadratische Matrix abgefangen wird

    A = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]
    b = [1.0, 2.0]

    with pytest.raises(ValueError):
        LRSolver(A, b)


def test_wrong_b_length_raises():
    # Testet ob falsche Länge von b erkannt wird

    A = [
        [1.0, 2.0],
        [3.0, 4.0],
    ]
    b = [1.0]

    with pytest.raises(ValueError):
        LRSolver(A, b)


def test_snapshot_contains_expected_keys():
    # Testet ob snapshot die erwarteten Daten enthält

    A = [
        [2.0, 1.0],
        [4.0, 3.0],
    ]
    b = [1.0, 2.0]

    solver = LRSolver(A, b)

    snap = solver.snapshot()

    # Prüft ob alle erwarteten Schlüssel vorhanden sind
    assert set(snap.keys()) == {"L", "R", "b", "y", "x", "phase"}