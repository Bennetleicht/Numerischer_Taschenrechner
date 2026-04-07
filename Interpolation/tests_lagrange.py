import pytest
import numpy as np
from lagrange_solver import LagrangeSolver



# Objekte

@pytest.fixture
def solver():
    return LagrangeSolver()

@pytest.fixture
def linear_solver(solver):
    """f(x) = 2x + 1  →  Stützpunkte (0,1) und (1,3)"""
    solver.start([0.0, 1.0], [1.0, 3.0])
    return solver

@pytest.fixture
def quadratic_solver(solver):
    """f(x) = x²  →  Stützpunkte (-1,1), (0,0), (1,1)"""
    solver.start([-1.0, 0.0, 1.0], [1.0, 0.0, 1.0])
    return solver

@pytest.fixture
def cubic_solver(solver):
    """f(x) = x³  →  4 Stützpunkte"""
    xs = [-2.0, -1.0, 1.0, 2.0]
    solver.start(xs, [x**3 for x in xs])
    return solver

# validate_input

class TestValidateInput:

    def test_none_x_raises(self, solver):
        with pytest.raises(ValueError, match="müssen existieren"):
            solver.validate_input(None, [1.0])

    def test_none_f_raises(self, solver):
        with pytest.raises(ValueError, match="müssen existieren"):
            solver.validate_input([1.0], None)

    def test_length_mismatch_raises(self, solver):
        with pytest.raises(ValueError, match="gleich viele Punkte"):
            solver.validate_input([0.0, 1.0], [1.0])

    def test_too_few_points_raises(self, solver):
        with pytest.raises(ValueError, match="Mindestens 2"):
            solver.validate_input([0.0], [1.0])

    def test_duplicate_x_raises(self, solver):
        with pytest.raises(ValueError, match="paarweise verschieden"):
            solver.validate_input([1.0, 1.0], [2.0, 3.0])

    def test_valid_input_passes(self, solver):
        solver.validate_input([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])  # kein Fehler erwartet


# start

class TestStart:

    def test_stores_x_and_f(self, solver):
        solver.start([0.0, 1.0], [5.0, 7.0])
        assert solver.x == [0.0, 1.0]
        assert solver.f == [5.0, 7.0]

    def test_copies_input_lists(self, solver):
        x = [0.0, 1.0]
        f = [1.0, 2.0]
        solver.start(x, f)
        x.append(99.0)
        assert len(solver.x) == 2  
    def test_invalid_input_propagates(self, solver):
        with pytest.raises(ValueError):
            solver.start([1.0], [1.0])


# evaluate (scalar)

class TestEvaluate:

    def test_linear_at_nodes(self, linear_solver):
        assert linear_solver.evaluate(0.0) == pytest.approx(1.0)
        assert linear_solver.evaluate(1.0) == pytest.approx(3.0)

    def test_linear_interpolation(self, linear_solver):
        assert linear_solver.evaluate(0.5) == pytest.approx(2.0)

    def test_quadratic_at_nodes(self, quadratic_solver):
        for xi, fi in zip([-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]):
            assert quadratic_solver.evaluate(xi) == pytest.approx(fi)

    def test_quadratic_between_nodes(self, quadratic_solver):
        assert quadratic_solver.evaluate(0.5) == pytest.approx(0.25)

    def test_cubic_at_nodes(self, cubic_solver):
        for x in [-2.0, -1.0, 1.0, 2.0]:
            assert cubic_solver.evaluate(x) == pytest.approx(x**3, abs=1e-10)

    def test_cubic_between_nodes(self, cubic_solver):
        assert cubic_solver.evaluate(0.0) == pytest.approx(0.0, abs=1e-10)

    def test_extrapolation_linear(self, linear_solver):
        # 2*2 + 1 = 5
        assert linear_solver.evaluate(2.0) == pytest.approx(5.0)


# evaluate_array

class TestEvaluateArray:

    def test_returns_ndarray(self, linear_solver):
        result = linear_solver.evaluate_array([0.0, 0.5, 1.0])
        assert isinstance(result, np.ndarray)

    def test_matches_scalar_evaluate(self, quadratic_solver):
        t_arr = np.linspace(-1.0, 1.0, 20)
        arr_result = quadratic_solver.evaluate_array(t_arr)
        scalar_result = np.array([quadratic_solver.evaluate(t) for t in t_arr])
        np.testing.assert_allclose(arr_result, scalar_result, atol=1e-12)

    def test_single_element_array(self, linear_solver):
        result = linear_solver.evaluate_array([0.5])
        assert result[0] == pytest.approx(2.0)

    def test_accepts_list_input(self, linear_solver):
        result = linear_solver.evaluate_array([0.0, 1.0])
        np.testing.assert_allclose(result, [1.0, 3.0])

    def test_empty_array(self, linear_solver):
        result = linear_solver.evaluate_array([])
        assert result.shape == (0,)


# Polynome richtig geparsed

class TestPolynomialCoeffs:

    def test_linear_coefficients(self, linear_solver):
        """f(x) = 2x + 1  →  Koeffizienten [1, 2]"""
        coeffs = linear_solver.polynomial_coeffs()
        assert coeffs[0] == pytest.approx(1.0, abs=1e-10) # Konstante
        assert coeffs[1] == pytest.approx(2.0, abs=1e-10) # lineare Term

    def test_quadratic_coefficients(self, quadratic_solver):
        """f(x) = x²  →  Koeffizienten [0, 0, 1]"""
        coeffs = quadratic_solver.polynomial_coeffs()
        assert coeffs[0] == pytest.approx(0.0, abs=1e-10)  # Konstante
        assert coeffs[1] == pytest.approx(0.0, abs=1e-10)  # lineare Term
        assert coeffs[2] == pytest.approx(1.0, abs=1e-10)  # quadratischer Term

