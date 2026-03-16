import pytest
from inkrementell.cholesky_solver import CholeskySolver


def test_start_valid():
    # Testet den normalen Start des Solvers mit gültigen Daten
    solver = CholeskySolver()

    A = [
        [4.0, 2.0],
        [2.0, 3.0],
    ]
    b = [6.0, 5.0]

    solver.start(A, b)

    # Prüft ob Dimension korrekt erkannt wurde
    assert solver.n == 2

    # Prüft ob A korrekt gespeichert wurde
    assert solver.A == A

    # Prüft ob b korrekt gespeichert wurde
    assert solver.b == b

    # Prüft ob L korrekt mit Nullen initialisiert wurde
    assert solver.L == [
        [0.0, 0.0],
        [0.0, 0.0],
    ]

    # Prüft ob y und x korrekt mit Nullen initialisiert wurden
    assert solver.y == [0.0, 0.0]
    assert solver.x == [0.0, 0.0]

    # Prüft ob die Phase korrekt auf Zerlegung gesetzt wurde
    assert solver.phase == "decomp"

    # Prüft ob der Solver nicht fertig ist
    assert solver.done is False


def test_start_empty_input():
    # Testet ob leere Eingaben abgefangen werden
    solver = CholeskySolver()

    with pytest.raises(ValueError, match="duerfen nicht leer sein"):
        solver.start([], [])


def test_start_non_square_matrix():
    # Testet ob nicht-quadratische Matrizen abgefangen werden
    solver = CholeskySolver()

    A = [
        [1.0, 2.0],
        [3.0],
    ]
    b = [1.0, 2.0]

    with pytest.raises(ValueError, match="quadratisch"):
        solver.start(A, b)


def test_start_wrong_b_length():
    # Testet ob falsche Länge von b erkannt wird
    solver = CholeskySolver()

    A = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    b = [1.0]

    with pytest.raises(ValueError, match="Laenge"):
        solver.start(A, b)


def test_step_without_start():
    # Testet Verhalten wenn step() vor start() aufgerufen wird
    solver = CholeskySolver()

    step = solver.step()

    # Solver sollte direkt done melden
    assert step.kind == "done"
    assert "Fertig" in step.message


def test_first_diagonal_update():
    # Testet den ersten Cholesky-Schritt L[1,1]
    solver = CholeskySolver()

    A = [
        [4.0, 2.0],
        [2.0, 3.0],
    ]
    b = [6.0, 5.0]

    solver.start(A, b)
    step = solver.step()

    # Erster Schritt muss ein Cholesky-Update sein
    assert step.kind == "chol_update"

    # Es muss L[0][0] = sqrt(4) = 2 werden
    assert solver.L[0][0] == pytest.approx(2.0)

    # Geänderte Zellen prüfen
    assert ("L", 0, 0) in step.changed
    assert ("Lt", 0, 0) in step.changed


def test_second_cholesky_update():
    # Testet den zweiten Schritt L[2,1]
    solver = CholeskySolver()

    A = [
        [4.0, 2.0],
        [2.0, 3.0],
    ]
    b = [6.0, 5.0]

    solver.start(A, b)

    solver.step()  # L[1,1]
    step = solver.step()  # L[2,1]

    # Zweiter Schritt muss wieder Zerlegung sein
    assert step.kind == "chol_update"

    # L[1][0] = 2 / 2 = 1
    assert solver.L[1][0] == pytest.approx(1.0)

    assert ("L", 1, 0) in step.changed
    assert ("Lt", 0, 1) in step.changed


def test_full_decomposition():
    # Testet komplette Cholesky-Zerlegung einer 2x2-Matrix
    solver = CholeskySolver()

    A = [
        [4.0, 2.0],
        [2.0, 3.0],
    ]
    b = [6.0, 5.0]

    solver.start(A, b)

    solver.step()  # L[0,0]
    solver.step()  # L[1,0]
    solver.step()  # L[1,1]

    # Erwartete Zerlegung:
    # L = [[2,0],[1,sqrt(2)]]
    assert solver.L[0][0] == pytest.approx(2.0)
    assert solver.L[1][0] == pytest.approx(1.0)
    assert solver.L[1][1] == pytest.approx(2**0.5)


def test_forward_substitution():
    # Testet Vorwärtseinsetzen nach abgeschlossener Zerlegung
    solver = CholeskySolver()

    A = [
        [4.0, 2.0],
        [2.0, 3.0],
    ]
    b = [6.0, 5.0]

    solver.start(A, b)

    # Zerlegung vollständig
    solver.step()
    solver.step()
    solver.step()

    # Jetzt beginnt forward phase
    step = solver.step()

    assert step.kind == "forward_sub"

    # y1 = 6 / 2 = 3
    assert solver.y[0] == pytest.approx(3.0)
    assert ("y", 0, 0) in step.changed


def test_backward_substitution():
    # Testet komplettes Lösen des Systems
    solver = CholeskySolver()

    A = [
        [4.0, 2.0],
        [2.0, 3.0],
    ]
    b = [6.0, 5.0]

    solver.start(A, b)

    # Alle Schritte bis zum Ende
    while True:
        step = solver.step()
        if step.kind == "done" and "abgeschlossen" in step.message:
            break

    # Exakte Lösung des Systems:
    # 4x1 + 2x2 = 6
    # 2x1 + 3x2 = 5
    # => x = [1, 1]
    assert solver.x[0] == pytest.approx(1.0)
    assert solver.x[1] == pytest.approx(1.0)


def test_matrix_not_positive_definite():
    # Testet Abbruch bei nicht positiv definiter Matrix
    solver = CholeskySolver()

    A = [
        [1.0, 2.0],
        [2.0, 1.0],
    ]
    b = [1.0, 1.0]

    solver.start(A, b)

    # Erster Schritt geht noch, zweiter oder dritter muss abbrechen
    step1 = solver.step()
    step2 = solver.step()
    step3 = solver.step()

    # Spätestens hier muss Abbruch kommen
    assert step3.kind == "done"
    assert "nicht positiv definit" in step3.message


def test_snapshot():
    # Testet ob snapshot den aktuellen Zustand korrekt zurückgibt
    solver = CholeskySolver()

    A = [
        [4.0, 2.0],
        [2.0, 3.0],
    ]
    b = [6.0, 5.0]

    solver.start(A, b)
    solver.step()  # L[0,0]

    snap = solver.snapshot()

    assert snap["A"] == A
    assert snap["b"] == b
    assert snap["L"][0][0] == pytest.approx(2.0)
    assert snap["phase"] == "decomp"