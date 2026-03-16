import pytest

from inkrementell.gauss_solver import GaussEliminationSolver


# Führt den Solver so lange aus bis er fertig ist
def run_until_done(solver: GaussEliminationSolver, max_steps: int = 100):
    steps = []
    for _ in range(max_steps):
        step = solver.next_step()
        steps.append(step)

        # Abbruch wenn Solver fertig ist
        if step.kind == "done":
            break

    return steps


def test_simple_system_solution_without_pivot_swap():
    # Testet ein einfaches LGS ohne Pivot-Tausch
    # Erwartete Lösung: x1 = 0.5, x2 = 0

    A = [
        [2.0, 1.0],
        [4.0, 3.0],
    ]
    b = [1.0, 2.0]

    solver = GaussEliminationSolver(A, b, pivot_mode="col")

    steps = run_until_done(solver)

    # Prüft ob der Solver korrekt beendet wurde
    assert steps[-1].kind == "done"

    # Prüft ob die berechnete Lösung korrekt ist
    x = solver.get_solution_original_order()
    assert x[0] == pytest.approx(0.5)
    assert x[1] == pytest.approx(0.0)


def test_column_pivot_swaps_rows():
    # Testet Spaltenpivotisierung
    # Erwartet einen Zeilentausch im ersten Schritt

    A = [
        [0.0, 1.0],
        [2.0, 3.0],
    ]
    b = [1.0, 5.0]

    solver = GaussEliminationSolver(A, b, pivot_mode="col")

    # Erster Schritt sollte ein Swap sein
    first = solver.next_step()

    assert first.kind == "swap"
    assert "Spaltenpivot" in first.message

    steps = run_until_done(solver)

    # Prüft ob der Solver korrekt beendet wurde
    assert steps[-1].kind == "done"

    # Prüft ob die Lösung korrekt ist
    x = solver.get_solution_original_order()
    assert x[0] == pytest.approx(1.0)
    assert x[1] == pytest.approx(1.0)


def test_row_pivot_swaps_columns_and_restores_original_variable_order():
    # Testet Zeilenpivotisierung (Spaltentausch)
    # Danach muss die Lösung wieder in Originalreihenfolge zurücksortiert werden

    A = [
        [1.0, 10.0],
        [2.0, 3.0],
    ]
    b = [21.0, 8.0]

    solver = GaussEliminationSolver(A, b, pivot_mode="row")

    # Prüft ob Spalten getauscht wurden
    first = solver.next_step()

    assert first.kind == "swap"
    assert "Zeilenpivot" in first.message

    steps = run_until_done(solver)

    # Prüft ob Solver korrekt beendet wurde
    assert steps[-1].kind == "done"

    # Prüft ob die Lösung korrekt berechnet wurde
    x = solver.get_solution_original_order()
    assert x[0] == pytest.approx(1.0)
    assert x[1] == pytest.approx(2.0)


def test_total_pivot_handles_row_and_column_swaps():
    # Testet Totalpivotisierung (Zeilen- und Spaltentausch)

    A = [
        [0.0, 2.0],
        [5.0, 1.0],
    ]
    b = [4.0, 7.0]

    solver = GaussEliminationSolver(A, b, pivot_mode="total")

    first = solver.next_step()

    # Prüft ob ein Pivot-Tausch durchgeführt wurde
    assert first.kind == "swap"
    assert "Totalpivot" in first.message

    steps = run_until_done(solver)

    assert steps[-1].kind == "done"

    # Prüft ob die Lösung korrekt ist
    x = solver.get_solution_original_order()
    assert x[0] == pytest.approx(1.0)
    assert x[1] == pytest.approx(2.0)


def test_custom_pivot_without_selection_aborts():
    # Testet ob ein fehlender benutzerdefinierter Pivot abgefangen wird

    A = [
        [1.0, 2.0],
        [3.0, 4.0],
    ]
    b = [5.0, 6.0]

    solver = GaussEliminationSolver(A, b, pivot_mode="custom", custom_pivot=None)

    step = solver.next_step()

    # Solver muss sofort abbrechen
    assert step.kind == "done"
    assert "kein Pivot" in step.message


def test_singular_matrix_aborts_on_zero_pivot():
    # Testet Abbruch bei singulärer Matrix (Pivot = 0)

    A = [
        [0.0, 1.0],
        [0.0, 2.0],
    ]
    b = [1.0, 2.0]

    solver = GaussEliminationSolver(A, b, pivot_mode="col")

    step = solver.next_step()

    assert step.kind == "done"
    assert "Pivot ist 0" in step.message


def test_invalid_non_square_matrix_raises():
    # Testet ob eine nicht-quadratische Matrix abgefangen wird

    A = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]
    b = [1.0, 2.0]

    with pytest.raises(ValueError):
        GaussEliminationSolver(A, b)


def test_invalid_b_length_raises():
    # Testet ob falsche Länge von b abgefangen wird

    A = [
        [1.0, 2.0],
        [3.0, 4.0],
    ]
    b = [1.0]

    with pytest.raises(ValueError):
        GaussEliminationSolver(A, b)


def test_backsub_returns_done_when_called_after_finish():
    # Testet Verhalten wenn next_step nach Ende erneut aufgerufen wird

    A = [
        [2.0, 1.0],
        [4.0, 3.0],
    ]
    b = [1.0, 2.0]

    solver = GaussEliminationSolver(A, b, pivot_mode="col")

    run_until_done(solver)

    step = solver.next_step()

    # Solver bleibt im Zustand "done"
    assert step.kind == "done"
    assert "Fertig" in step.message