import pytest
import numpy as np
from polynom_solver import VandermondeSolver


# Objekte

@pytest.fixture
def solver():
    return VandermondeSolver()


@pytest.fixture
def linear_solver(solver):
    """f(x) = 2x + 1  →  Stützpunkte (0,1) und (1,3)"""
    solver.start([0.0, 1.0], [1.0, 3.0])
    solver.solve()
    return solver


@pytest.fixture
def quadratic_solver(solver):
    """f(x) = x²  →  Stützpunkte (-1,1), (0,0), (1,1)"""
    solver.start([-1.0, 0.0, 1.0], [1.0, 0.0, 1.0])
    solver.solve()
    return solver


@pytest.fixture
def cubic_solver(solver):
    """f(x) = x³  →  4 Stützpunkte"""
    xs = [-2.0, -1.0, 1.0, 2.0]
    solver.start(xs, [x ** 3 for x in xs])
    solver.solve()
    return solver


# TestStart – Prüft Eingabevalidierung und korrekte Speicherung der Stützpunkte

class TestStart:

    def test_stores_x_and_f(self, solver):
        solver.start([0.0, 1.0], [1.0, 2.0])
        assert solver.x == [0.0, 1.0]
        assert solver.f == [1.0, 2.0]

    def test_resets_coeffs_and_V(self, linear_solver):
        linear_solver.start([0.0, 1.0], [1.0, 2.0])
        assert linear_solver.coeffs is None
        assert linear_solver.V is None

    def test_copies_lists(self, solver):
        x = [0.0, 1.0]
        f = [1.0, 2.0]
        solver.start(x, f)
        x.append(99.0)
        assert len(solver.x) == 2

    def test_length_mismatch_raises(self, solver):
        with pytest.raises(ValueError, match="gleich lang"):
            solver.start([0.0, 1.0], [1.0])

    def test_too_few_points_raises(self, solver):
        with pytest.raises(ValueError, match="Mindestens 2"):
            solver.start([0.0], [1.0])

    def test_duplicate_x_raises(self, solver):
        with pytest.raises(ValueError, match="paarweise verschieden"):
            solver.start([1.0, 1.0], [2.0, 3.0])


# TestBuildVandermonde – Prüft Aufbau und Struktur der Vandermonde-Matrix

class TestBuildVandermonde:

    def test_returns_ndarray(self, solver):
        solver.start([0.0, 1.0], [1.0, 2.0])
        V = solver.build_vandermonde()
        assert isinstance(V, np.ndarray)

    def test_shape(self, solver):
        solver.start([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
        V = solver.build_vandermonde()
        assert V.shape == (3, 3)

    def test_first_column_all_ones(self, solver):
        solver.start([-1.0, 0.0, 1.0], [1.0, 0.0, 1.0])
        V = solver.build_vandermonde()
        np.testing.assert_array_equal(V[:, 0], [1.0, 1.0, 1.0])

    def test_second_column_equals_x(self, solver):
        xs = [-1.0, 0.0, 2.0]
        solver.start(xs, [0.0, 0.0, 0.0])
        V = solver.build_vandermonde()
        np.testing.assert_array_equal(V[:, 1], xs)

    def test_stores_V_on_instance(self, solver):
        solver.start([0.0, 1.0], [1.0, 2.0])
        solver.build_vandermonde()
        assert solver.V is not None

    def test_known_2x2_matrix(self, solver):
        """V für x=[0,1] muss [[1,0],[1,1]] sein."""
        solver.start([0.0, 1.0], [1.0, 3.0])
        V = solver.build_vandermonde()
        expected = np.array([[1.0, 0.0], [1.0, 1.0]])
        np.testing.assert_allclose(V, expected)


# TestSolve – Prüft ob das LGS korrekt gelöst und gespeichert wird

class TestSolve:

    def test_returns_ndarray(self, solver):
        solver.start([0.0, 1.0], [1.0, 3.0])
        coeffs = solver.solve()
        assert isinstance(coeffs, np.ndarray)

    def test_stores_coeffs(self, solver):
        solver.start([0.0, 1.0], [1.0, 3.0])
        solver.solve()
        assert solver.coeffs is not None

    def test_builds_vandermonde_automatically(self, solver):
        solver.start([0.0, 1.0], [1.0, 3.0])
        assert solver.V is None
        solver.solve()
        assert solver.V is not None

    def test_linear_coefficients(self, linear_solver):
        """f(x) = 2x + 1  →  a_0=1, a_1=2"""
        assert linear_solver.coeffs[0] == pytest.approx(1.0, abs=1e-10)
        assert linear_solver.coeffs[1] == pytest.approx(2.0, abs=1e-10)

    def test_quadratic_coefficients(self, quadratic_solver):
        """f(x) = x²  →  a_0=0, a_1=0, a_2=1"""
        np.testing.assert_allclose(quadratic_solver.coeffs, [0.0, 0.0, 1.0], atol=1e-10)

    def test_coeffs_length(self, cubic_solver):
        assert len(cubic_solver.coeffs) == 4


# TestEvaluate – Prüft skalare Auswertung des interpolierten Polynoms

class TestEvaluate:

    def test_raises_without_solve(self, solver):
        solver.start([0.0, 1.0], [1.0, 3.0])
        with pytest.raises(RuntimeError, match="Noch nicht gelöst"):
            solver.evaluate(0.5)

    def test_linear_at_nodes(self, linear_solver):
        assert linear_solver.evaluate(0.0) == pytest.approx(1.0)
        assert linear_solver.evaluate(1.0) == pytest.approx(3.0)

    def test_linear_between_nodes(self, linear_solver):
        assert linear_solver.evaluate(0.5) == pytest.approx(2.0)

    def test_quadratic_at_nodes(self, quadratic_solver):
        for xi, fi in zip([-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]):
            assert quadratic_solver.evaluate(xi) == pytest.approx(fi, abs=1e-10)

    def test_quadratic_between_nodes(self, quadratic_solver):
        assert quadratic_solver.evaluate(0.5) == pytest.approx(0.25, abs=1e-10)

    def test_extrapolation(self, linear_solver):
        assert linear_solver.evaluate(2.0) == pytest.approx(5.0)

    def test_returns_float(self, linear_solver):
        assert isinstance(linear_solver.evaluate(0.5), float)


# TestEvaluateArray – Prüft vektorisierte Auswertung über numpy-Arrays

class TestEvaluateArray:

    def test_raises_without_solve(self, solver):
        solver.start([0.0, 1.0], [1.0, 3.0])
        with pytest.raises(RuntimeError, match="Noch nicht gelöst"):
            solver.evaluate_array([0.5])

    def test_returns_ndarray(self, linear_solver):
        assert isinstance(linear_solver.evaluate_array([0.0, 0.5, 1.0]), np.ndarray)

    def test_matches_scalar_evaluate(self, quadratic_solver):
        t_arr = np.linspace(-1.0, 1.0, 20)
        arr = quadratic_solver.evaluate_array(t_arr)
        scalar = np.array([quadratic_solver.evaluate(t) for t in t_arr])
        np.testing.assert_allclose(arr, scalar, atol=1e-12)

    def test_accepts_list_input(self, linear_solver):
        result = linear_solver.evaluate_array([0.0, 1.0])
        np.testing.assert_allclose(result, [1.0, 3.0])

    def test_single_element(self, linear_solver):
        result = linear_solver.evaluate_array([0.5])
        assert result[0] == pytest.approx(2.0)

    def test_empty_array(self, linear_solver):
        result = linear_solver.evaluate_array([])
        assert result.shape == (0,)


# TestPolyString – Prüft die lesbare Textdarstellung des Polynoms

class TestPolyString:

    def test_before_solve_returns_placeholder(self, solver):
        solver.start([0.0, 1.0], [1.0, 2.0])
        assert solver.poly_string() == "—"

    def test_starts_with_p_of_x(self, linear_solver):
        assert linear_solver.poly_string().startswith("p(x) =")

    def test_linear_contains_x_term(self, linear_solver):
        s = linear_solver.poly_string()
        assert "·x" in s

    def test_quadratic_contains_x_squared(self, quadratic_solver):
        s = quadratic_solver.poly_string()
        assert "x^2" in s

    def test_zero_polynomial(self, solver):
        """Alle Koeffizienten null → p(x) = 0"""
        solver.start([0.0, 1.0], [0.0, 0.0])
        solver.solve()
        assert solver.poly_string() == "p(x) = 0"

    def test_returns_string(self, linear_solver):
        assert isinstance(linear_solver.poly_string(), str)
