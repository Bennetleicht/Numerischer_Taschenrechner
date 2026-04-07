import math
import pytest
import numpy as np

from bezier_solver import BezierSolver


# Objekte

@pytest.fixture
def solver():
    return BezierSolver()


@pytest.fixture
def quadratic(solver):
    """Quadratische Bézierkurve: drei Kontrollpunkte."""
    solver.start([(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)])
    return solver


@pytest.fixture
def cubic(solver):
    """Kubische Bézierkurve: vier Kontrollpunkte."""
    solver.start([(0.0, 0.0), (0.25, 1.0), (0.75, 1.0), (1.0, 0.0)])
    return solver


#  TestStart – Prüft Eingabevalidierung, Sortierung und Speicherung

class TestStart:
    def test_accepts_two_points(self, solver):
        solver.start([(0, 0), (1, 1)])
        assert len(solver.control_points) == 2

    def test_accepts_many_points(self, solver):
        pts = [(i, i * 2) for i in range(10)]
        solver.start(pts)
        assert solver.control_points == pts

    def test_raises_on_single_point(self, solver):
        with pytest.raises(ValueError):
            solver.start([(0, 0)])

    def test_raises_on_empty_list(self, solver):
        with pytest.raises(ValueError):
            solver.start([])

    def test_stores_copy(self, solver):
        pts = [(0.0, 0.0), (1.0, 1.0)]
        solver.start(pts)
        pts.append((2.0, 2.0))         
        assert len(solver.control_points) == 2  


#  Bernstein-Basis 

class TestBernsteinBasis:
    def test_partition_of_unity(self, solver):
        """Summe aller Basispolynome muss 1 ergeben (Zerlegung der Eins)."""
        n = 4
        for t in np.linspace(0, 1, 11):
            total = sum(solver.bernstein_basis(i, n, t) for i in range(n + 1))
            assert math.isclose(total, 1.0, abs_tol=1e-12), f"t={t}: Summe={total}"

    def test_non_negative(self, solver):
        """Basispolynome sind auf [0,1] nicht-negativ."""
        n = 3
        for t in np.linspace(0, 1, 21):
            for i in range(n + 1):
                assert solver.bernstein_basis(i, n, t) >= 0.0

    def test_boundary_t0(self, solver):
        """Bei t=0 ist nur b_{0,n}(0) = 1, alle anderen = 0."""
        n = 3
        assert math.isclose(solver.bernstein_basis(0, n, 0.0), 1.0)
        for i in range(1, n + 1):
            assert math.isclose(solver.bernstein_basis(i, n, 0.0), 0.0)

    def test_boundary_t1(self, solver):

        n = 3
        assert math.isclose(solver.bernstein_basis(n, n, 1.0), 1.0)
        for i in range(n):
            assert math.isclose(solver.bernstein_basis(i, n, 1.0), 0.0)

    def test_symmetry(self, solver):

        n = 5
        for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for i in range(n + 1):
                assert math.isclose(
                    solver.bernstein_basis(i, n, t),
                    solver.bernstein_basis(n - i, n, 1 - t),
                    abs_tol=1e-12,
                )

    def test_array_matches_scalar(self, solver):
        """bernstein_basis_array muss mit bernstein_basis übereinstimmen."""
        n, i = 3, 2
        t_arr = np.linspace(0, 1, 50)
        scalar_vals = np.array([solver.bernstein_basis(i, n, float(t)) for t in t_arr])
        array_vals = solver.bernstein_basis_array(i, n, t_arr)
        np.testing.assert_allclose(array_vals, scalar_vals, atol=1e-14)


#  Endpunkte 

class TestEndpoints:
    """Bézierkurven interpolieren immer den ersten und letzten Kontrollpunkt."""

    @pytest.mark.parametrize("method", ["evaluate_casteljau", "evaluate_bernstein"])
    def test_start_point(self, cubic, method):
        pt = getattr(cubic, method)(0.0)
        assert math.isclose(pt[0], 0.0, abs_tol=1e-12)
        assert math.isclose(pt[1], 0.0, abs_tol=1e-12)

    @pytest.mark.parametrize("method", ["evaluate_casteljau", "evaluate_bernstein"])
    def test_end_point(self, cubic, method):
        pt = getattr(cubic, method)(1.0)
        assert math.isclose(pt[0], 1.0, abs_tol=1e-12)
        assert math.isclose(pt[1], 0.0, abs_tol=1e-12)


# Äquivalenz: de Casteljau == Bernstein 

class TestEquivalence:
    """Beide Algorithmen müssen für alle t numerisch übereinstimmen."""

    @pytest.mark.parametrize("t", np.linspace(0, 1, 21).tolist())
    def test_quadratic_equivalence(self, quadratic, t):
        dc = quadratic.evaluate_casteljau(t)
        bs = quadratic.evaluate_bernstein(t)
        assert math.isclose(dc[0], bs[0], abs_tol=1e-10)
        assert math.isclose(dc[1], bs[1], abs_tol=1e-10)

    @pytest.mark.parametrize("t", np.linspace(0, 1, 21).tolist())
    def test_cubic_equivalence(self, cubic, t):
        dc = cubic.evaluate_casteljau(t)
        bs = cubic.evaluate_bernstein(t)
        assert math.isclose(dc[0], bs[0], abs_tol=1e-10)
        assert math.isclose(dc[1], bs[1], abs_tol=1e-10)

# de-Casteljau-Tableau 

class TestCasteljauTableau:
    def test_tableau_depth(self, cubic):
        """Tableau hat n+1 Stufen für n Grad (kubisch → 4 Stufen)."""
        tableau = cubic.de_casteljau_full(0.5)
        assert len(tableau) == 4

    def test_tableau_shrinks(self, cubic):
        """Jede Stufe hat einen Punkt weniger als die vorherige."""
        tableau = cubic.de_casteljau_full(0.5)
        for level, row in enumerate(tableau):
            assert len(row) == 4 - level

    def test_first_row_is_control_points(self, cubic):
        """Erste Tableau-Zeile = ursprüngliche Kontrollpunkte."""
        tableau = cubic.de_casteljau_full(0.5)
        assert tableau[0] == cubic.control_points

    def test_last_row_matches_evaluate(self, cubic):
        """Letzter Eintrag des Tableaus == evaluate_casteljau."""
        t = 0.3
        tableau = cubic.de_casteljau_full(t)
        direct = cubic.evaluate_casteljau(t)
        assert math.isclose(tableau[-1][0][0], direct[0], abs_tol=1e-14)
        assert math.isclose(tableau[-1][0][1], direct[1], abs_tol=1e-14)


#  Bernstein-Schritte 

class TestBernsteinSteps:
    def test_step_count(self, cubic):
        """bernstein_steps liefert n+1 Einträge (einen je Kontrollpunkt)."""
        steps = cubic.bernstein_steps(0.5)
        assert len(steps) == 4

    def test_final_partial_matches_evaluate(self, cubic):
        """Letzte Partialsumme == evaluate_bernstein."""
        t = 0.6
        steps = cubic.bernstein_steps(t)
        expected = cubic.evaluate_bernstein(t)
        assert math.isclose(steps[-1]["partial"][0], expected[0], abs_tol=1e-12)
        assert math.isclose(steps[-1]["partial"][1], expected[1], abs_tol=1e-12)

    def test_b_vals_sum_to_one(self, quadratic):
        """Summe der b_val-Felder = 1 (Partition der Eins)."""
        steps = quadratic.bernstein_steps(0.4)
        total = sum(s["b_val"] for s in steps)
        assert math.isclose(total, 1.0, abs_tol=1e-12)

    def test_contrib_matches_b_val_times_point(self, cubic):
        """contrib = b_val · P_i für jeden Schritt."""
        pts = cubic.control_points
        steps = cubic.bernstein_steps(0.7)
        for s in steps:
            i, b = s["i"], s["b_val"]
            assert math.isclose(s["contrib"][0], b * pts[i][0], abs_tol=1e-14)
            assert math.isclose(s["contrib"][1], b * pts[i][1], abs_tol=1e-14)
