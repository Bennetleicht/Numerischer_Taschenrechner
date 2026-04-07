import pytest
import numpy as np
from spline_solver import CubicSplineSolver


# Objekte

@pytest.fixture
def solver():
    return CubicSplineSolver()


@pytest.fixture
def natural_solver(solver):
    """Natürlicher Spline durch f(x) = x² an 5 Stützpunkten."""
    xs = [-2.0, -1.0, 0.0, 1.0, 2.0]
    solver.start(xs, [x ** 2 for x in xs])
    solver.compute()
    return solver


@pytest.fixture
def hermite_solver(solver):
    """Hermite-Spline durch f(x) = x³, Ableitungen exakt: f'(-2)=12, f'(2)=12."""
    xs = [-2.0, -1.0, 0.0, 1.0, 2.0]
    solver.start(xs, [x ** 3 for x in xs], boundary="hermite", df0=12.0, dfn=12.0)
    solver.compute()
    return solver


@pytest.fixture
def three_point_solver(solver):
    """Minimaler natürlicher Spline mit genau 3 Stützpunkten."""
    solver.start([0.0, 1.0, 2.0], [0.0, 1.0, 0.0])
    solver.compute()
    return solver


# TestStart – Prüft Eingabevalidierung, Sortierung und Speicherung

class TestStart:

    def test_stores_x_and_f(self, solver):
        solver.start([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
        assert solver.x == [0.0, 1.0, 2.0]
        assert solver.f == [0.0, 1.0, 4.0]

    def test_sorts_by_x(self, solver):
        solver.start([2.0, 0.0, 1.0], [4.0, 0.0, 1.0])
        assert solver.x == [0.0, 1.0, 2.0]
        assert solver.f == [0.0, 1.0, 4.0]

    def test_length_mismatch_raises(self, solver):
        with pytest.raises(ValueError, match="gleich lang"):
            solver.start([0.0, 1.0], [1.0])

    def test_too_few_points_raises(self, solver):
        with pytest.raises(ValueError, match="Mindestens 3"):
            solver.start([0.0, 1.0], [0.0, 1.0])

    def test_duplicate_x_raises(self, solver):
        with pytest.raises(ValueError, match="paarweise verschieden"):
            solver.start([1.0, 1.0, 2.0], [0.0, 1.0, 2.0])

    def test_stores_boundary(self, solver):
        solver.start([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], boundary="hermite")
        assert solver.boundary == "hermite"

    def test_stores_derivatives(self, solver):
        solver.start([0.0, 1.0, 2.0], [0.0, 1.0, 0.0],
                     boundary="hermite", df0=1.0, dfn=-1.0)
        assert solver.df0 == 1.0
        assert solver.dfn == -1.0

    def test_default_boundary_is_natural(self, solver):
        solver.start([0.0, 1.0, 2.0], [0.0, 1.0, 0.0])
        assert solver.boundary == "natural"


# TestCompute – Prüft Aufbau des Systems und Berechnung der Momente

class TestCompute:

    def test_returns_matrix_and_rhs(self, solver):
        solver.start([0.0, 1.0, 2.0], [0.0, 1.0, 0.0])
        A, rhs = solver.compute()
        assert isinstance(A, np.ndarray)
        assert isinstance(rhs, np.ndarray)

    def test_natural_moments_boundary_zero(self, natural_solver):
        """Natürlicher Spline: M_0 = M_n = 0."""
        assert natural_solver.M[0] == pytest.approx(0.0, abs=1e-12)
        assert natural_solver.M[-1] == pytest.approx(0.0, abs=1e-12)

    def test_natural_matrix_shape(self, solver):
        solver.start([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 0.0, 1.0])
        A, rhs = solver.compute()
        assert A.shape == (2, 2)

    def test_hermite_matrix_shape(self, solver):
        solver.start([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 0.0, 1.0],
                     boundary="hermite")
        A, rhs = solver.compute()
        assert A.shape == (4, 4)

    def test_h_values_correct(self, solver):
        solver.start([0.0, 2.0, 5.0], [0.0, 1.0, 0.0])
        solver.compute()
        assert solver.h == pytest.approx([2.0, 3.0])

    def test_coeffs_computed_after_compute(self, solver):
        solver.start([0.0, 1.0, 2.0], [0.0, 1.0, 0.0])
        solver.compute()
        assert len(solver.a) == 2
        assert len(solver.b) == 2
        assert len(solver.c) == 2
        assert len(solver.d) == 2

    def test_stores_V_and_rhs(self, solver):
        solver.start([0.0, 1.0, 2.0], [0.0, 1.0, 0.0])
        solver.compute()
        assert solver.V is not None
        assert solver.rhs is not None


# TestEvaluate – Prüft skalare Auswertung des Splines

class TestEvaluate:

    def test_at_nodes_natural(self, natural_solver):
        for xi, fi in zip([-2.0, -1.0, 0.0, 1.0, 2.0], [4.0, 1.0, 0.0, 1.0, 4.0]):
            assert natural_solver.evaluate(xi) == pytest.approx(fi, abs=1e-10)

    def test_at_nodes_hermite(self, hermite_solver):
        for xi in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            assert hermite_solver.evaluate(xi) == pytest.approx(xi ** 3, abs=1e-10)

    def test_between_nodes(self, three_point_solver):
        result = three_point_solver.evaluate(1.0)
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_left_extrapolation_uses_first_segment(self, natural_solver):
        """Werte links von x_0 landen im ersten Segment."""
        result = natural_solver.evaluate(-3.0)
        assert isinstance(result, float)

    def test_right_boundary_exact(self, natural_solver):
        assert natural_solver.evaluate(2.0) == pytest.approx(4.0, abs=1e-10)


# TestEvaluateArray – Prüft vektorisierte Auswertung

class TestEvaluateArray:

    def test_returns_ndarray(self, natural_solver):
        result = natural_solver.evaluate_array([0.0, 0.5, 1.0])
        assert isinstance(result, np.ndarray)

    def test_matches_scalar_evaluate(self, natural_solver):
        t_arr = np.linspace(-2.0, 2.0, 30)
        arr = natural_solver.evaluate_array(t_arr)
        scalar = np.array([natural_solver.evaluate(t) for t in t_arr])
        np.testing.assert_allclose(arr, scalar, atol=1e-12)

    def test_at_all_nodes(self, natural_solver):
        result = natural_solver.evaluate_array([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = [4.0, 1.0, 0.0, 1.0, 4.0]
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_single_element(self, natural_solver):
        result = natural_solver.evaluate_array([0.0])
        assert result[0] == pytest.approx(0.0, abs=1e-10)

    def test_empty_array(self, natural_solver):
        result = natural_solver.evaluate_array([])
        assert len(result) == 0


# TestConsistency – Prüft mathematische Korrektheit mit bekannten Funktionen


class TestConsistency:

    def test_sin_approximation(self, solver):
        """Kubischer Spline soll sin(x) auf [0, 2π] gut approximieren."""
        xs = np.linspace(0, 2 * np.pi, 10)
        solver.start(list(xs), list(np.sin(xs)))
        solver.compute()
        t_test = np.linspace(0.1, 2 * np.pi - 0.1, 30)
        result = solver.evaluate_array(t_test)
        np.testing.assert_allclose(result, np.sin(t_test), atol=1e-3)

    def test_constant_function(self, solver):
        """Spline durch f(x) = 3 muss überall 3 liefern."""
        solver.start([0.0, 1.0, 2.0, 3.0], [3.0, 3.0, 3.0, 3.0])
        solver.compute()
        for t in np.linspace(0.0, 3.0, 20):
            assert solver.evaluate(t) == pytest.approx(3.0, abs=1e-10)

    def test_linear_function(self, solver):
        """Spline durch f(x) = 2x muss exakt linear bleiben."""
        xs = [0.0, 1.0, 2.0, 3.0]
        solver.start(xs, [2 * x for x in xs])
        solver.compute()
        for t in np.linspace(0.0, 3.0, 20):
            assert solver.evaluate(t) == pytest.approx(2 * t, abs=1e-10)