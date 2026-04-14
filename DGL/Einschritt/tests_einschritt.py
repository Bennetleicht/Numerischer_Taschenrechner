import math
import pytest
import sympy as sp

from .einschritt_solver import (
    EinschrittSolver,
    METHOD_EE,
    METHOD_HE,
    METHOD_ME,
    METHOD_RK,
    METHOD_IE,
)

#Objekte
@pytest.fixture
def t():
    return sp.Symbol("t")


@pytest.fixture
def y():
    return sp.Symbol("y")


@pytest.fixture
def f():
    return lambda t, y: y


@pytest.fixture
def f_expr(y):
    return y

#Hilfsfunktion zum Erstellen von Solver-Objekten mit Standardparametern
def make_solver(method, f, f_expr, a=0.0, b=0.1, y0=1.0, h=0.1):
    return EinschrittSolver(method, f, f_expr, a, b, y0, h)

#testet eingaben
def test_init_rejects_nonpositive_h(f, f_expr):
    with pytest.raises(ValueError, match="Schrittweite h muss positiv sein"):
        EinschrittSolver(METHOD_EE, f, f_expr, 0.0, 1.0, 1.0, 0.0)

#testet Eingaben
def test_init_rejects_b_not_greater_than_a(f, f_expr):
    with pytest.raises(ValueError, match="b muss größer als a sein"):
        EinschrittSolver(METHOD_EE, f, f_expr, 1.0, 1.0, 1.0, 0.1)

#testet den Anfangszustand des Solvers
def test_initial_state(f, f_expr):
    solver = make_solver(METHOD_EE, f, f_expr, a=0.0, b=1.0, y0=2.0, h=0.2)

    assert solver.ts == [0.0]
    assert solver.ys == [2.0]
    assert solver.n == 0
    assert solver.is_finished is False
    assert solver.current_step_size() == pytest.approx(0.2)

#testet die Berechnung der Schrittgröße am Ende des Intervalls
def test_current_step_size_uses_remaining_interval(f, f_expr):
    solver = make_solver(METHOD_EE, f, f_expr, a=0.0, b=0.25, y0=1.0, h=0.1)

    r1 = solver.step()
    r2 = solver.step()
    r3 = solver.step()

    assert r1.t_next == pytest.approx(0.1)
    assert r2.t_next == pytest.approx(0.2)
    assert r3.t_next == pytest.approx(0.25)
    assert r3.values["h"] == pytest.approx(0.05)
    assert solver.is_finished is True

#testet, dass nach Abschluss des Intervalls kein weiterer Schritt möglich ist
def test_step_after_finish_raises(f, f_expr):
    solver = make_solver(METHOD_EE, f, f_expr, a=0.0, b=0.1, y0=1.0, h=0.1)

    solver.step()

    with pytest.raises(RuntimeError, match="Intervall bereits vollständig berechnet"):
        solver.step()

#testet, dass ein unbekanntes Verfahren zu einem Fehler führt
def test_unknown_method_raises(f, f_expr):
    solver = make_solver("Irgendwas kaputtes", f, f_expr)

    with pytest.raises(ValueError, match="Unbekanntes Verfahren"):
        solver.step()

#testet die Berechnung eines Schritts mit dem expliziten Euler-Verfahren
def test_explicit_euler_one_step(f, f_expr):
    solver = make_solver(METHOD_EE, f, f_expr)
    result = solver.step()

    expected = 1.0 + 0.1 * 1.0  # y_n + h * f(t_n, y_n) = 1 + 0.1*1

    assert result.n == 0
    assert result.t_n == pytest.approx(0.0)
    assert result.y_n == pytest.approx(1.0)
    assert result.t_next == pytest.approx(0.1)
    assert result.y_next == pytest.approx(expected)

    assert result.values["method"] == METHOD_EE
    assert result.values["h"] == pytest.approx(0.1)
    assert result.values["yprime_n"] == pytest.approx(1.0)

    assert solver.ts == [0.0, 0.1]
    assert solver.ys == [1.0, expected]
    assert solver.n == 1

#testet die Berechnung eines Schritts mit dem Heun-Verfahren
def test_heun_one_step(f, f_expr):
    solver = make_solver(METHOD_HE, f, f_expr)
    result = solver.step()

    k1 = 1.0
    y_star = 1.0 + 0.1 * k1
    k2 = y_star
    expected = 1.0 + 0.1 * 0.5 * (k1 + k2)

    assert result.y_next == pytest.approx(expected)
    assert result.values["method"] == METHOD_HE
    assert result.values["k1"] == pytest.approx(k1)
    assert result.values["y_star"] == pytest.approx(y_star)
    assert result.values["k2"] == pytest.approx(k2)

#testet die Berechnung eines Schritts mit dem modifizierten Euler-Verfahren
def test_modified_euler_one_step(f, f_expr):
    solver = make_solver(METHOD_ME, f, f_expr)
    result = solver.step()

    k1 = 1.0
    t_mid = 0.05
    y_mid = 1.0 + 0.05 * k1
    f_mid = y_mid
    expected = 1.0 + 0.1 * f_mid

    assert result.y_next == pytest.approx(expected)
    assert result.values["method"] == METHOD_ME
    assert result.values["k1"] == pytest.approx(k1)
    assert result.values["t_mid"] == pytest.approx(t_mid)
    assert result.values["y_mid"] == pytest.approx(y_mid)
    assert result.values["yprime_mid"] == pytest.approx(f_mid)

#testet die Berechnung eines Schritts mit dem RK4-Verfahren
def test_rk4_one_step(f, f_expr):
    solver = make_solver(METHOD_RK, f, f_expr)
    result = solver.step()

    k1 = 1.0
    k2 = 1.0 + 0.05 * k1
    k3 = 1.0 + 0.05 * k2
    k4 = 1.0 + 0.1 * k3
    expected = 1.0 + (0.1 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    assert result.y_next == pytest.approx(expected)
    assert result.values["method"] == METHOD_RK
    assert result.values["k1"] == pytest.approx(k1)
    assert result.values["k2"] == pytest.approx(k2)
    assert result.values["k3"] == pytest.approx(k3)
    assert result.values["k4"] == pytest.approx(k4)

#testet die Berechnung eines Schritts mit dem impliziten Euler-Verfahren
def test_implicit_euler_one_step_linear_case(f, f_expr):
    solver = make_solver(METHOD_IE, f, f_expr)
    result = solver.step()

    # y' = y  =>  y_{n+1} = y_n + h*y_{n+1}
    # => y_{n+1} = y_n / (1-h)
    expected = 1.0 / (1.0 - 0.1)

    assert result.y_next == pytest.approx(expected)
    assert result.values["method"] == METHOD_IE
    assert result.values["h"] == pytest.approx(0.1)
    assert result.values["start_guess"] == pytest.approx(1.1)
    assert "y_next" in result.values["equation"]
