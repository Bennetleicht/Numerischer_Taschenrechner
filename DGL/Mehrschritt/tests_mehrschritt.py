import pytest
import sympy as sp

from .mehrschritt_solver import (
    MehrschrittSolver,
    StepResult,
    METHOD_AB,
    METHOD_AM,
    METHOD_BDF,
    ALLOWED_ORDERS,
    AB_COEFFS,
    AM_COEFFS,
    BDF_ALPHAS,
)


# Objekte / Hilfsfunktionen


@pytest.fixture
def linear_problem():
    # y' = y, y(0)=1
    t, y = sp.symbols("t y")
    f_expr = y

    def f(t_val, y_val):
        return y_val

    return f, f_expr


@pytest.fixture
def constant_problem():
    # y' = 2, y(0)=1
    t, y = sp.symbols("t y")
    f_expr = sp.Integer(2)

    def f(t_val, y_val):
        return 2.0

    return f, f_expr


def make_solver(method, order, f, f_expr, a=0.0, b=1.0, y0=1.0, h=0.1):
    return MehrschrittSolver(
        method_name=method,
        order=order,
        f=f,
        f_expr=f_expr,
        a=a,
        b=b,
        y0=y0,
        h=h,
    )


# Grundzustand


def test_initial_state(linear_problem):
    f, f_expr = linear_problem
    solver = make_solver(METHOD_AB, 1, f, f_expr)

    assert solver.n == 0
    assert solver.ts == [0.0]
    assert solver.ys == [1.0]
    assert solver.is_finished is False

#wird häufiger ausgeführt
@pytest.mark.parametrize("order", ALLOWED_ORDERS)
def test_required_history_equals_order(linear_problem, order):
    f, f_expr = linear_problem
    solver = make_solver(METHOD_AB, order, f, f_expr)

    assert solver._required_history() == order


def test_is_finished_false_initially(linear_problem):
    f, f_expr = linear_problem
    solver = make_solver(METHOD_AB, 1, f, f_expr, a=0.0, b=1.0, h=0.25)

    assert solver.is_finished is False


def test_is_finished_true_at_right_boundary(linear_problem):
    f, f_expr = linear_problem
    solver = make_solver(METHOD_AB, 1, f, f_expr, a=0.0, b=1.0, h=0.25)

    solver.ts = [0.0, 0.25, 0.5, 0.75, 1.0]
    solver.ys = [1, 1, 1, 1, 1]

    assert solver.is_finished is True


# current_step_size

def test_current_step_size_regular(linear_problem):
    f, f_expr = linear_problem
    solver = make_solver(METHOD_AB, 1, f, f_expr, h=0.2)

    assert solver.current_step_size() == pytest.approx(0.2)


def test_current_step_size_shortened_at_end(linear_problem):
    f, f_expr = linear_problem
    solver = make_solver(METHOD_AB, 1, f, f_expr, b=1.0, h=0.3)

    solver.ts = [0.0, 0.3, 0.6, 0.9]
    solver.ys = [1, 1, 1, 1]

    assert solver.current_step_size() == pytest.approx(0.1)


# StepResult / allgemeine Konsistenz

def test_step_returns_stepresult(linear_problem):
    f, f_expr = linear_problem
    solver = make_solver(METHOD_AB, 1, f, f_expr)

    result = solver.step()

    assert isinstance(result, StepResult)


def test_step_updates_internal_state(linear_problem):
    f, f_expr = linear_problem
    solver = make_solver(METHOD_AB, 1, f, f_expr, h=0.1)

    result = solver.step()

    assert solver.n == 1
    assert len(solver.ts) == 2
    assert len(solver.ys) == 2
    assert solver.ts[-1] == pytest.approx(result.t_next)
    assert solver.ys[-1] == pytest.approx(result.y_next)


def test_step_result_time_fields_are_consistent(linear_problem):
    f, f_expr = linear_problem
    solver = make_solver(METHOD_AB, 1, f, f_expr, h=0.25)

    result = solver.step()

    assert result.n == 0
    assert result.t_n == pytest.approx(0.0)
    assert result.t_next == pytest.approx(0.25)
    assert result.y_n == pytest.approx(1.0)


def test_step_after_finished_raises(linear_problem):
    f, f_expr = linear_problem
    solver = make_solver(METHOD_AB, 1, f, f_expr, a=0.0, b=0.1, h=0.1)

    solver.step()

    with pytest.raises(RuntimeError, match="Intervall bereits vollständig berechnet"):
        solver.step()


# Bootstrap

@pytest.mark.parametrize("method", [METHOD_AB, METHOD_AM, METHOD_BDF])
def test_bootstrap_used_for_order_2(method, linear_problem):
    f, f_expr = linear_problem
    solver = make_solver(method, 2, f, f_expr, h=0.1)

    result = solver.step()

    assert result.values["bootstrap"] is True
    assert "k1" in result.values
    assert "k2" in result.values
    assert "k3" in result.values
    assert "k4" in result.values


@pytest.mark.parametrize("method", [METHOD_AB, METHOD_AM, METHOD_BDF])
def test_no_bootstrap_for_order_1(method, linear_problem):
    f, f_expr = linear_problem
    solver = make_solver(method, 1, f, f_expr, h=0.1)

    result = solver.step()

    assert result.values["bootstrap"] is False


def test_bootstrap_needed_counter_for_order_4(linear_problem):
    f, f_expr = linear_problem
    solver = make_solver(METHOD_AB, 4, f, f_expr, h=0.1)

    result = solver.step()

    assert result.values["bootstrap_index"] == 0
    assert result.values["bootstrap_needed"] == 3


def test_order_3_uses_two_bootstrap_steps_then_real_method(linear_problem):
    f, f_expr = linear_problem
    solver = make_solver(METHOD_AB, 3, f, f_expr, h=0.1)

    r1 = solver.step()
    r2 = solver.step()
    r3 = solver.step()

    assert r1.values["bootstrap"] is True
    assert r2.values["bootstrap"] is True
    assert r3.values["bootstrap"] is False


# AB-Tests

def test_ab1_equals_explicit_euler_for_y_equals_y(linear_problem):
    f, f_expr = linear_problem
    solver = make_solver(METHOD_AB, 1, f, f_expr, y0=1.0, h=0.1)

    result = solver.step()

    expected = 1.0 + 0.1 * 1.0
    assert result.y_next == pytest.approx(expected)


def test_ab1_constant_rhs_is_exact(constant_problem):
    f, f_expr = constant_problem
    solver = make_solver(METHOD_AB, 1, f, f_expr, y0=1.0, h=0.25)

    result = solver.step()

    # y' = 2 => y(t) = 1 + 2t
    expected = 1.0 + 2.0 * 0.25
    assert result.y_next == pytest.approx(expected)


def test_ab2_second_step_uses_history(linear_problem):
    f, f_expr = linear_problem
    solver = make_solver(METHOD_AB, 2, f, f_expr, h=0.1)

    solver.step()  # bootstrap
    result = solver.step()

    assert result.values["bootstrap"] is False
    assert len(result.values["hist_t"]) == 2
    assert len(result.values["hist_y"]) == 2
    assert len(result.values["hist_f"]) == 2
    assert len(result.values["coeffs"]) == 2


def test_ab2_formula_matches_saved_values(linear_problem):
    f, f_expr = linear_problem
    solver = make_solver(METHOD_AB, 2, f, f_expr, h=0.1)

    solver.step()  # bootstrap
    result = solver.step()

    hist_y = result.values["hist_y"]
    hist_f = result.values["hist_f"]
    coeffs = AB_COEFFS[2]
    h = result.values["h"]

    expected = hist_y[0] + h * (
        float(coeffs[0]) * hist_f[0] +
        float(coeffs[1]) * hist_f[1]
    )

    assert result.y_next == pytest.approx(expected)


# AM-Tests

def test_am1_equals_backward_euler_for_y_equals_y(linear_problem):
    f, f_expr = linear_problem
    solver = make_solver(METHOD_AM, 1, f, f_expr, y0=1.0, h=0.1)

    result = solver.step()

    expected = 1.0 / (1.0 - 0.1)
    assert result.y_next == pytest.approx(expected)


def test_am1_constant_rhs_is_exact(constant_problem):
    f, f_expr = constant_problem
    solver = make_solver(METHOD_AM, 1, f, f_expr, y0=1.0, h=0.2)

    result = solver.step()

    expected = 1.0 + 2.0 * 0.2
    assert result.y_next == pytest.approx(expected)


def test_am2_after_bootstrap_has_predictor_and_f_next(linear_problem):
    f, f_expr = linear_problem
    solver = make_solver(METHOD_AM, 2, f, f_expr, h=0.1)

    solver.step()  # bootstrap
    result = solver.step()

    assert result.values["bootstrap"] is False
    assert "predictor" in result.values
    assert "f_next" in result.values
    assert "t_next" in result.values
    assert len(result.values["coeffs"]) == 2
    assert len(result.values["hist_f"]) == 1


def test_am2_formula_matches_saved_values(linear_problem):
    f, f_expr = linear_problem
    solver = make_solver(METHOD_AM, 2, f, f_expr, h=0.1)

    solver.step()  # bootstrap
    result = solver.step()

    coeffs = AM_COEFFS[2]
    y_n = result.y_n
    h = result.values["h"]
    f_next = result.values["f_next"]
    hist_f = result.values["hist_f"]

    expected = y_n + h * (
        float(coeffs[0]) * f_next +
        float(coeffs[1]) * hist_f[0]
    )

    assert result.y_next == pytest.approx(expected)


# BDF-Tests

def test_bdf1_equals_backward_euler_for_y_equals_y(linear_problem):
    f, f_expr = linear_problem
    solver = make_solver(METHOD_BDF, 1, f, f_expr, y0=1.0, h=0.1)

    result = solver.step()

    expected = 1.0 / (1.0 - 0.1)
    assert result.y_next == pytest.approx(expected)


def test_bdf1_constant_rhs_is_exact(constant_problem):
    f, f_expr = constant_problem
    solver = make_solver(METHOD_BDF, 1, f, f_expr, y0=1.0, h=0.25)

    result = solver.step()

    expected = 1.0 + 2.0 * 0.25
    assert result.y_next == pytest.approx(expected)


def test_bdf2_after_bootstrap_has_predictor_and_f_next(linear_problem):
    f, f_expr = linear_problem
    solver = make_solver(METHOD_BDF, 2, f, f_expr, h=0.1)

    solver.step()  # bootstrap
    result = solver.step()

    assert result.values["bootstrap"] is False
    assert "predictor" in result.values
    assert "f_next" in result.values
    assert "t_next" in result.values
    assert len(result.values["alphas"]) == 3
    assert len(result.values["hist_y"]) == 2


def test_bdf2_formula_matches_saved_values(linear_problem):
    f, f_expr = linear_problem
    solver = make_solver(METHOD_BDF, 2, f, f_expr, h=0.1)

    solver.step()  # bootstrap
    result = solver.step()

    alphas = BDF_ALPHAS[2]
    y_next = result.y_next
    hist_y = result.values["hist_y"]
    h = result.values["h"]
    f_next = result.values["f_next"]

    lhs = float(alphas[0]) * y_next + float(alphas[1]) * hist_y[0] + float(alphas[2]) * hist_y[1]
    rhs = h * f_next

    assert lhs == pytest.approx(rhs)


# Letzter verkürzter Schritt

@pytest.mark.parametrize("method", [METHOD_AB, METHOD_AM, METHOD_BDF])
def test_last_step_lands_exactly_on_b_order_1(method, linear_problem):
    f, f_expr = linear_problem
    solver = make_solver(method, 1, f, f_expr, a=0.0, b=1.0, y0=1.0, h=0.3)

    while not solver.is_finished:
        solver.step()

    assert solver.ts[-1] == pytest.approx(1.0)


def test_last_result_has_shortened_t_next(linear_problem):
    f, f_expr = linear_problem
    solver = make_solver(METHOD_AB, 1, f, f_expr, a=0.0, b=1.0, y0=1.0, h=0.3)

    r1 = solver.step()
    r2 = solver.step()
    r3 = solver.step()
    r4 = solver.step()

    assert r1.t_next == pytest.approx(0.3)
    assert r2.t_next == pytest.approx(0.6)
    assert r3.t_next == pytest.approx(0.9)
    assert r4.t_next == pytest.approx(1.0)


# Mehrere Schritte / Monotonie bei y'=y

@pytest.mark.parametrize("method", [METHOD_AB, METHOD_AM, METHOD_BDF])
def test_solution_grows_for_y_equals_y(method, linear_problem):
    f, f_expr = linear_problem
    solver = make_solver(method, 1, f, f_expr, y0=1.0, h=0.1)

    prev = solver.ys[-1]
    for _ in range(5):
        solver.step()
        assert solver.ys[-1] > prev
        prev = solver.ys[-1]


def test_n_matches_number_of_steps(linear_problem):
    f, f_expr = linear_problem
    solver = make_solver(METHOD_AB, 1, f, f_expr, a=0.0, b=0.5, y0=1.0, h=0.1)

    count = 0
    while not solver.is_finished:
        solver.step()
        count += 1

    assert solver.n == count


# Tabellenkonsistenz

@pytest.mark.parametrize("order", ALLOWED_ORDERS)
def test_ab_coeff_lengths(order):
    assert len(AB_COEFFS[order]) == order


@pytest.mark.parametrize("order", ALLOWED_ORDERS)
def test_am_coeff_lengths(order):
    assert len(AM_COEFFS[order]) == order


@pytest.mark.parametrize("order", ALLOWED_ORDERS)
def test_bdf_alpha_lengths(order):
    assert len(BDF_ALPHAS[order]) == order + 1


@pytest.mark.parametrize("order", ALLOWED_ORDERS)
def test_coeff_entries_are_numeric_like(order):
    for c in AB_COEFFS[order]:
        assert float(c) == pytest.approx(float(c))
    for c in AM_COEFFS[order]:
        assert float(c) == pytest.approx(float(c))
    for a in BDF_ALPHAS[order]:
        assert float(a) == pytest.approx(float(a))