from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pytest
import numpy as np
from gauss_legendre_solver import Gauss_Legendre_Solver, _GL_NODES, _GL_WEIGHTS


# Hilfsfunktionen 
def f_const(x):  return 2.0           # ∫₋₁¹ 2 dx = 4,  ∫₀¹ 2 dx = 2
def f_linear(x): return x             # ∫₀¹ x dx = 0.5
def f_quad(x):   return x**2          # ∫₀¹ x² dx = 1/3
def f_cubic(x):  return x**3          # ∫₀¹ x³ dx = 0.25
def f_x4(x):     return x**4          # ∫₁³ x⁴ dx = 48.4 (exakt)
def f_sin(x):    return np.sin(x)     # ∫₀π sin dx = 2
def f_exp(x):    return np.exp(x)     # ∫₀¹ eˣ dx = e-1 ≈ 1.71828
def f_neg(x):    return -x            # ∫₀¹ -x dx = -0.5


# Objekte  
@pytest.fixture
def solver():
    return Gauss_Legendre_Solver()


#  Tabellen-Tests: Stützstellen und Gewichte
class TestTabellen:
    """Prüft ob die gespeicherten Werte korrekt sind."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_anzahl_stuetzstellen(self, n):
        assert len(_GL_NODES[n]) == n

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_anzahl_gewichte(self, n):
        assert len(_GL_WEIGHTS[n]) == n

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_gewichtssumme_ist_zwei(self, n):
        # Summe aller Gewichte muss immer 2 sein (Länge des Intervalls [-1,1])
        assert pytest.approx(sum(_GL_WEIGHTS[n]), rel=1e-9) == 2.0

    #alle im Intervall [-1,1] liegenden Stützstellen
    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_stuetzstellen_in_minus1_bis_1(self, n):
        for x in _GL_NODES[n]:
            assert -1.0 <= x <= 1.0

    def test_n1_stuetzstelle(self):
        assert pytest.approx(_GL_NODES[1][0], abs=1e-12) == 0.0

    def test_n1_gewicht(self):
        assert pytest.approx(_GL_WEIGHTS[1][0], rel=1e-12) == 2.0

    def test_n2_stuetzstellen_symmetrisch(self):
        x1, x2 = _GL_NODES[2]
        assert pytest.approx(x1, rel=1e-9) == -1/np.sqrt(3)
        assert pytest.approx(x2, rel=1e-9) ==  1/np.sqrt(3)

    def test_n2_gewichte_gleich(self):
        assert pytest.approx(_GL_WEIGHTS[2][0], rel=1e-9) == 1.0
        assert pytest.approx(_GL_WEIGHTS[2][1], rel=1e-9) == 1.0

    def test_n3_mittlere_stuetzstelle_null(self):
        assert pytest.approx(_GL_NODES[3][1], abs=1e-12) == 0.0

    def test_n3_mittleres_gewicht(self):
        assert pytest.approx(_GL_WEIGHTS[3][1], rel=1e-9) == 8/9

    def test_n3_aeussere_gewichte(self):
        assert pytest.approx(_GL_WEIGHTS[3][0], rel=1e-9) == 5/9
        assert pytest.approx(_GL_WEIGHTS[3][2], rel=1e-9) == 5/9

    def test_n5_mittlere_stuetzstelle_null(self):
        assert pytest.approx(_GL_NODES[5][2], abs=1e-12) == 0.0

    def test_n5_mittleres_gewicht(self):
        assert pytest.approx(_GL_WEIGHTS[5][2], rel=1e-9) == 128/225

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_stuetzstellen_aufsteigend(self, n):
        nodes = _GL_NODES[n]
        for i in range(len(nodes) - 1):
            assert nodes[i] < nodes[i+1]


# Init-Tests
class TestInit:
    def test_alle_attribute_none(self, solver):
        assert solver.f       is None
        assert solver.a       is None
        assert solver.b       is None
        assert solver.n       is None
        assert solver.xi      is None
        assert solver.nodes   is None
        assert solver.weights is None
        assert solver.fxi     is None
        assert solver.I       is None


# Start-Tests
class TestStart:
    def test_setzt_f(self, solver):
        solver.start(f_linear, 0, 1, 2)
        assert solver.f is f_linear

    def test_setzt_a_b(self, solver):
        solver.start(f_linear, 0, 1, 2)
        assert solver.a == 0.0
        assert solver.b == 1.0

    def test_setzt_n(self, solver):
        solver.start(f_linear, 0, 1, 3)
        assert solver.n == 3

    def test_xi_korrekte_laenge(self, solver):
        for n in [1, 2, 3, 4, 5]:
            solver.start(f_linear, 0, 1, n)
            assert len(solver.xi) == n

    def test_nodes_korrekte_laenge(self, solver):
        for n in [1, 2, 3, 4, 5]:
            solver.start(f_linear, 0, 1, n)
            assert len(solver.nodes) == n

    def test_weights_korrekte_laenge(self, solver):
        for n in [1, 2, 3, 4, 5]:
            solver.start(f_linear, 0, 1, n)
            assert len(solver.weights) == n

    def test_transformation_nodes_in_intervall(self, solver):
        solver.start(f_linear, 2, 5, 3)
        for t in solver.nodes:
            assert 2.0 <= t <= 5.0

    def test_transformation_formel(self, solver):
        # tᵢ = (b-a)/2 · xᵢ + (a+b)/2
        a, b = 1.0, 3.0
        solver.start(f_linear, a, b, 2)
        h   = 0.5 * (b - a)
        mid = 0.5 * (a + b)
        for xi, ti in zip(solver.xi, solver.nodes):
            expected = h * xi + mid
            assert pytest.approx(ti, rel=1e-9) == expected

    def test_I_ist_none_nach_start(self, solver):
        solver.start(f_linear, 0, 1, 2)
        assert solver.I is None

    def test_fehler_a_gleich_b(self, solver):
        with pytest.raises(ValueError, match="a < b"):
            solver.start(f_linear, 1, 1, 2)

    def test_fehler_a_groesser_b(self, solver):
        with pytest.raises(ValueError, match="a < b"):
            solver.start(f_linear, 5, 1, 2)

    def test_fehler_n_zu_gross(self, solver):
        with pytest.raises(ValueError, match="n muss"):
            solver.start(f_linear, 0, 1, 6)

    def test_fehler_n_null(self, solver):
        with pytest.raises(ValueError, match="n muss"):
            solver.start(f_linear, 0, 1, 0)

    def test_fehler_n_negativ(self, solver):
        with pytest.raises(ValueError, match="n muss"):
            solver.start(f_linear, 0, 1, -1)

    def test_fehler_nan_fa(self, solver):
        with pytest.raises(ValueError, match="NaN"):
            solver.start(lambda x: float("nan"), 0, 1, 2)

    def test_fehler_inf_fb(self, solver):
        with pytest.raises(ValueError, match="NaN"):
            solver.start(lambda x: float("inf") if x == 1 else 0.0, 0, 1, 2)

    def test_negatives_intervall_erlaubt(self, solver):
        solver.start(f_linear, -3, -1, 2)
        assert solver.a == -3.0
        assert solver.b == -1.0

    def test_grosses_intervall(self, solver):
        solver.start(f_const, 0, 1000, 2)
        assert solver.b == 1000.0


# Step ohne Start
class TestStepOhneStart:
    def test_gibt_not_started_zurueck(self, solver):
        status, row, done = solver.step()
        assert done is True
        assert row  is None
        assert "Nicht gestartet" in status


# Step: Rückgabewerte
class TestStepRueckgabe:
    def test_done_ist_true(self, solver):
        solver.start(f_linear, 0, 1, 2)
        _, _, done = solver.step()
        assert done is True

    def test_row_ist_nicht_none(self, solver):
        solver.start(f_linear, 0, 1, 2)
        _, row, _ = solver.step()
        assert row is not None

    def test_row_hat_8_elemente(self, solver):
        solver.start(f_linear, 0, 1, 2)
        _, row, _ = solver.step()
        assert len(row) == 8

    def test_row_a_b_korrekt(self, solver):
        solver.start(f_linear, 2, 5, 2)
        _, row, _ = solver.step()
        a, b, n, xi, nodes, fxi, wi, I = row
        assert a == 2.0
        assert b == 5.0

    def test_row_n_korrekt(self, solver):
        solver.start(f_linear, 0, 1, 3)
        _, row, _ = solver.step()
        _, _, n, _, _, _, _, _ = row
        assert n == 3

    def test_status_enthaelt_integral(self, solver):
        solver.start(f_linear, 0, 1, 2)
        status, _, _ = solver.step()
        assert "Integral berechnet" in status

    def test_fxi_laenge_korrekt(self, solver):
        for n in [1, 2, 3, 4, 5]:
            solver.start(f_linear, 0, 1, n)
            _, row, _ = solver.step()
            _, _, _, _, _, fxi, _, _ = row
            assert len(fxi) == n

    def test_I_wird_gesetzt(self, solver):
        solver.start(f_linear, 0, 1, 2)
        solver.step()
        assert solver.I is not None
        assert np.isfinite(solver.I)


# Grenzfälle 
class TestGrenzfaelle:
    def test_sehr_kleines_intervall(self, solver):
        solver.start(f_quad, 0, 1e-8, 2)
        _, row, _ = solver.step()
        I = row[7]
        assert np.isfinite(I)

    def test_grosses_intervall(self, solver):
        # ∫₀¹⁰⁰⁰ 2 dx = 2000
        solver.start(f_const, 0, 1000, 2)
        _, row, _ = solver.step()
        I = row[7]
        assert pytest.approx(I, rel=1e-6) == 2000.0

    def test_negatives_intervall(self, solver):
        # ∫₋₁⁰ x² dx = 1/3
        solver.start(f_quad, -1, 0, 3)
        _, row, _ = solver.step()
        I = row[7]
        assert pytest.approx(I, rel=1e-6) == 1/3

    def test_symmetrisches_intervall_ungerade_funktion(self, solver):
        # ∫₋₁¹ x dx = 0
        solver.start(f_linear, -1, 1, 2)
        _, row, _ = solver.step()
        I = row[7]
        assert pytest.approx(I, abs=1e-10) == 0.0

    def test_alle_n_liefern_endliche_werte(self, solver):
        for n in [1, 2, 3, 4, 5]:
            solver.start(f_sin, 0, np.pi, n)
            _, row, _ = solver.step()
            I = row[7]
            assert np.isfinite(I)

    def test_mehrfacher_aufruf_start(self, solver):
        # start kann mehrfach aufgerufen werden
        solver.start(f_linear, 0, 1, 2)
        solver.step()
        solver.start(f_quad, 0, 1, 3)
        _, row, _ = solver.step()
        I = row[7]
        assert pytest.approx(I, rel=1e-6) == 1/3