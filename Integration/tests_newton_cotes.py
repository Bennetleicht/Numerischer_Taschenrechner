from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pytest
import numpy as np
from newton_cotes_solver import Newton_Cotes_Solver


# Hilfsfunktionen
def f_const(x):   return 2.0
def f_linear(x):  return x
def f_quad(x):    return x**2
def f_cubic(x):   return x**3
def f_sin(x):     return np.sin(x)
def f_neg(x):     return -x


# Objekte
@pytest.fixture
def solver_trapez():
    return Newton_Cotes_Solver("Trapezregel", "Einzelstreifen")

@pytest.fixture
def solver_simpson():
    return Newton_Cotes_Solver("Simpsonregel", "Einzelstreifen")

@pytest.fixture
def solver_38():
    return Newton_Cotes_Solver("3/8-Regel", "Einzelstreifen")

@pytest.fixture
def solver_milne():
    return Newton_Cotes_Solver("Milne-Regel", "Einzelstreifen")


# Init-Tests
class TestInit:
    def test_init_werte(self):
        s = Newton_Cotes_Solver("Trapezregel", "Einzelstreifen")
        assert s.f is None
        assert s.a is None
        assert s.b is None
        assert s.tol == 0.0
        assert s.m == 1
        assert s.verfahren == "Trapezregel"
        assert s.modus == "Einzelstreifen"

    def test_init_verschiedene_verfahren(self):
        for v in ["Trapezregel", "Simpsonregel", "3/8-Regel", "Milne-Regel"]:
            s = Newton_Cotes_Solver(v, "Einzelstreifen")
            assert s.verfahren == v

    def test_init_modus(self):
        s = Newton_Cotes_Solver("Trapezregel", "Doppelstreifen")
        assert s.modus == "Doppelstreifen"


# Start-Tests
class TestStart:
    def test_start_setzt_werte(self, solver_trapez):
        solver_trapez.start(f_linear, 0, 1, 0, m=4)
        assert solver_trapez.f is f_linear
        assert solver_trapez.a == 0.0
        assert solver_trapez.b == 1.0
        assert solver_trapez.tol == 0.0
        assert solver_trapez.m == 4

    def test_start_standard_m(self, solver_trapez):
        solver_trapez.start(f_linear, 0, 1, 0)
        assert solver_trapez.m == 1

    def test_start_toleranz(self, solver_trapez):
        solver_trapez.start(f_linear, 0, 1, 1e-6, m=2)
        assert solver_trapez.tol == 1e-6

    def test_start_fehler_a_groesser_b(self, solver_trapez):
        with pytest.raises(ValueError, match="a < b"):
            solver_trapez.start(f_linear, 5, 1, 0)

    def test_start_fehler_a_gleich_b(self, solver_trapez):
        with pytest.raises(ValueError, match="a < b"):
            solver_trapez.start(f_linear, 1, 1, 0)

    def test_start_fehler_negative_toleranz(self, solver_trapez):
        with pytest.raises(ValueError, match="Toleranz"):
            solver_trapez.start(f_linear, 0, 1, -1)

    def test_start_fehler_m_zu_klein(self, solver_trapez):
        with pytest.raises(ValueError, match="m muss >= 1"):
            solver_trapez.start(f_linear, 0, 1, 0, m=0)

    def test_start_fehler_nan_fa(self, solver_trapez):
        def f_nan(x): return float("nan")
        with pytest.raises(ValueError, match="NaN/Inf"):
            solver_trapez.start(f_nan, 0, 1, 0)

    def test_start_fehler_inf_fb(self, solver_trapez):
        def f_inf(x): return float("inf") if x == 1 else 0.0
        with pytest.raises(ValueError, match="NaN/Inf"):
            solver_trapez.start(f_inf, 0, 1, 0)

    def test_start_negative_intervall(self, solver_trapez):
        solver_trapez.start(f_linear, -5, -1, 0, m=3)
        assert solver_trapez.a == -5.0
        assert solver_trapez.b == -1.0

    def test_start_grosse_werte(self, solver_trapez):
        solver_trapez.start(f_const, 0, 1e6, 0, m=10)
        assert solver_trapez.b == 1e6


# Step ohne Start
class TestStepOhneStart:
    def test_step_ohne_start(self, solver_trapez):
        status, row, done = solver_trapez.step()
        assert done is True
        assert row is None
        assert "Nicht gestartet" in status


# Trapezregel
class TestTrapezregel:
    def test_konstante_funktion(self, solver_trapez):
        solver_trapez.start(f_const, 0, 1, 0, m=4)
        _, row, done = solver_trapez.step()
        assert done is True
        assert row is not None

        a, b, I, m, xs, ys = row
        assert pytest.approx(I, rel=1e-9) == 2.0
        assert m == 4
        assert len(xs) == 5
        assert len(ys) == 5

    def test_lineare_funktion(self, solver_trapez):
        solver_trapez.start(f_linear, 0, 1, 0, m=8)
        _, row, _ = solver_trapez.step()
        _, _, I, _, _, _ = row
        assert pytest.approx(I, rel=1e-9) == 0.5

    def test_quadratische_funktion_fehler_bei_einem_streifen(self, solver_trapez):
        solver_trapez.start(f_quad, 0, 1, 0, m=1)
        _, row, _ = solver_trapez.step()
        _, _, I, _, _, _ = row
        assert abs(I - 1/3) > 1e-6

    def test_negative_funktion(self, solver_trapez):
        solver_trapez.start(f_neg, 0, 1, 0, m=6)
        _, row, _ = solver_trapez.step()
        _, _, I, _, _, _ = row
        assert pytest.approx(I, rel=1e-9) == -0.5

    def test_row_enthaelt_a_b(self, solver_trapez):
        solver_trapez.start(f_linear, 2, 5, 0, m=3)
        _, row, _ = solver_trapez.step()
        a, b, I, m, xs, ys = row
        assert a == 2.0
        assert b == 5.0
        assert m == 3
        assert len(xs) == 4
        assert len(ys) == 4

    def test_done_ist_true(self, solver_trapez):
        solver_trapez.start(f_linear, 0, 1, 0, m=2)
        _, _, done = solver_trapez.step()
        assert done is True

    def test_status_string(self, solver_trapez):
        solver_trapez.start(f_linear, 0, 1, 0, m=2)
        status, _, _ = solver_trapez.step()
        assert "Integral berechnet" in status


# Simpsonregel
class TestSimpsonregel:
    def test_konstante(self, solver_simpson):
        solver_simpson.start(f_const, 0, 1, 0, m=2)
        _, row, _ = solver_simpson.step()
        _, _, I, m, xs, ys = row
        assert pytest.approx(I, rel=1e-9) == 2.0
        assert m == 2
        assert len(xs) == 3
        assert len(ys) == 3

    def test_linear(self, solver_simpson):
        solver_simpson.start(f_linear, 0, 1, 0, m=2)
        _, row, _ = solver_simpson.step()
        _, _, I, _, _, _ = row
        assert pytest.approx(I, rel=1e-9) == 0.5

    def test_quadratisch_exakt(self, solver_simpson):
        solver_simpson.start(f_quad, 0, 1, 0, m=2)
        _, row, _ = solver_simpson.step()
        _, _, I, _, _, _ = row
        assert pytest.approx(I, rel=1e-9) == 1/3

    def test_kubisch_exakt(self, solver_simpson):
        solver_simpson.start(f_cubic, 0, 1, 0, m=2)
        _, row, _ = solver_simpson.step()
        _, _, I, _, _, _ = row
        assert pytest.approx(I, rel=1e-9) == 0.25

    def test_sinus_einzelnes_simpson_panel(self, solver_simpson):
        solver_simpson.start(f_sin, 0, np.pi, 0, m=2)
        _, row, _ = solver_simpson.step()
        _, _, I, _, _, _ = row
        assert pytest.approx(I, rel=1e-9) == 2 * np.pi / 3

    def test_ungerades_m_wird_aufgerundet(self, solver_simpson):
        solver_simpson.start(f_linear, 0, 1, 0, m=3)
        _, row, _ = solver_simpson.step()
        _, _, I, m, xs, ys = row
        assert m == 4
        assert len(xs) == 5
        assert len(ys) == 5
        assert pytest.approx(I, rel=1e-9) == 0.5


# 3/8-Regel
class TestDreiAchtelRegel:
    def test_konstante(self, solver_38):
        solver_38.start(f_const, 0, 1, 0, m=7)
        _, row, _ = solver_38.step()
        _, _, I, m, xs, ys = row
        assert pytest.approx(I, rel=1e-9) == 2.0
        assert m == 7
        assert len(xs) == 4
        assert len(ys) == 4

    def test_linear(self, solver_38):
        solver_38.start(f_linear, 0, 1, 0, m=5)
        _, row, _ = solver_38.step()
        _, _, I, _, _, _ = row
        assert pytest.approx(I, rel=1e-9) == 0.5

    def test_kubisch_exakt(self, solver_38):
        solver_38.start(f_cubic, 0, 1, 0, m=9)
        _, row, _ = solver_38.step()
        _, _, I, _, _, _ = row
        assert pytest.approx(I, rel=1e-9) == 0.25

    def test_row_struktur(self, solver_38):
        solver_38.start(f_linear, 1, 3, 0, m=4)
        _, row, done = solver_38.step()
        a, b, I, m, xs, ys = row
        assert a == 1.0
        assert b == 3.0
        assert m == 4
        assert len(xs) == 4
        assert len(ys) == 4
        assert done is True


# Milne-Regel
class TestMilneRegel:
    def test_konstante(self, solver_milne):
        solver_milne.start(f_const, 0, 1, 0, m=8)
        _, row, _ = solver_milne.step()
        _, _, I, m, xs, ys = row
        assert pytest.approx(I, rel=1e-9) == 2.0
        assert m == 8
        assert len(xs) == 5
        assert len(ys) == 5

    def test_linear(self, solver_milne):
        solver_milne.start(f_linear, 0, 1, 0, m=3)
        _, row, _ = solver_milne.step()
        _, _, I, _, _, _ = row
        assert pytest.approx(I, rel=1e-9) == 0.5

    def test_kubisch_exakt(self, solver_milne):
        solver_milne.start(f_cubic, 0, 1, 0, m=6)
        _, row, _ = solver_milne.step()
        _, _, I, _, _, _ = row
        assert pytest.approx(I, rel=1e-9) == 0.25

    def test_sinus_genau(self, solver_milne):
        solver_milne.start(f_sin, 0, np.pi, 0, m=1)
        _, row, _ = solver_milne.step()
        _, _, I, _, _, _ = row
        assert pytest.approx(I, rel=1e-2) == 2.0


# Unbekanntes Verfahren
class TestUnbekanntesVerfahren:
    def test_unbekanntes_verfahren(self):
        s = Newton_Cotes_Solver("Unbekannt", "Einzelstreifen")
        s.start(f_linear, 0, 1, 0)
        with pytest.raises(ValueError, match="Unbekanntes Verfahren"):
            s.step()


# Vergleich der Verfahren
class TestVerfahrenVergleich:
    @pytest.mark.parametrize("v", ["Trapezregel", "Simpsonregel", "3/8-Regel", "Milne-Regel"])
    def test_alle_exakt_fuer_konstante(self, v):
        s = Newton_Cotes_Solver(v, "Einzelstreifen")
        s.start(f_const, 0, 1, 0, m=4)
        _, row, _ = s.step()
        _, _, I, _, _, _ = row
        assert pytest.approx(I, rel=1e-9) == 2.0

    @pytest.mark.parametrize("v", ["Trapezregel", "Simpsonregel", "3/8-Regel", "Milne-Regel"])
    def test_alle_exakt_fuer_linear(self, v):
        s = Newton_Cotes_Solver(v, "Einzelstreifen")
        s.start(f_linear, 0, 1, 0, m=4)
        _, row, _ = s.step()
        _, _, I, _, _, _ = row
        assert pytest.approx(I, rel=1e-9) == 0.5

    @pytest.mark.parametrize("v", ["Simpsonregel", "3/8-Regel", "Milne-Regel"])
    def test_hoehere_verfahren_exakt_fuer_kubisch(self, v):
        s = Newton_Cotes_Solver(v, "Einzelstreifen")
        s.start(f_cubic, 0, 1, 0, m=4)
        _, row, _ = s.step()
        _, _, I, _, _, _ = row
        assert pytest.approx(I, rel=1e-9) == 0.25


