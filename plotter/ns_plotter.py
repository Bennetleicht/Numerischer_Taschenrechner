from __future__ import annotations
import numpy as np

from base_plotter import DynamicFunctionPlot

#Hier Bisektion
class ABPlotter(DynamicFunctionPlot):
    def __init__(self, parent):
        self.a = None
        self.b = None
        super().__init__(parent)

    def _init_overlay(self):
        self.vline_a = self.ax.axvline(0, linestyle="--", linewidth=1)
        self.vline_b = self.ax.axvline(0, linestyle="--", linewidth=1)
        self.scatter_ab = self.ax.scatter([], [], s=60)
        self.ann_a = self.ax.annotate("a", xy=(0, 0), xytext=(6, 6), textcoords="offset points")
        self.ann_b = self.ax.annotate("b", xy=(0, 0), xytext=(6, 6), textcoords="offset points")
        self._overlay_artists = [self.vline_a, self.vline_b, self.scatter_ab, self.ann_a, self.ann_b]

    def set_ab(self, a, b):
        self.a = float(a)
        self.b = float(b)
        self.set_overlay_visible(True)
        self._update_overlay_geometry()
        self.canvas.draw_idle()

    def clear_ab(self):
        self.a = None
        self.b = None
        self.set_overlay_visible(False)
        self.canvas.draw_idle()

    def _update_overlay_geometry(self):
        if self.f is None or self.a is None or self.b is None:
            return
        a, b = self.a, self.b
        try:
            fa = float(self.f(a))
            fb = float(self.f(b))
        except Exception:
            fa, fb = np.nan, np.nan

        self.vline_a.set_xdata([a, a])
        self.vline_b.set_xdata([b, b])
        self.scatter_ab.set_offsets(np.array([[a, fa], [b, fb]], dtype=float))
        self.ann_a.xy = (a, fa)
        self.ann_b.xy = (b, fb)

#Hier Newton
class NewtonPlotter(DynamicFunctionPlot):
    def __init__(self, parent):
        self.xk = None
        self.yk = None
        self.dyk = None
        self.xnext = None
        super().__init__(parent)

    def _init_overlay(self):
        self.scatter_xk = self.ax.scatter([], [], s=60)
        self.ann_xk = self.ax.annotate("xk", xy=(0, 0), xytext=(6, 6), textcoords="offset points")
        (self.tangent_line,) = self.ax.plot([], [], linewidth=1.2)
        self.scatter_xn = self.ax.scatter([], [], s=60)
        self.ann_xn = self.ax.annotate("xk+1", xy=(0, 0), xytext=(6, 6), textcoords="offset points")
        self.vline_xn = self.ax.axvline(0, linestyle="--", linewidth=1)
        self._overlay_artists = [self.scatter_xk, self.ann_xk, self.tangent_line, self.scatter_xn, self.ann_xn, self.vline_xn]

    def clear_state(self):
        self.xk = self.yk = self.dyk = self.xnext = None
        self.set_overlay_visible(False)
        self.tangent_line.set_data([], [])
        self.canvas.draw_idle()

    def set_state(self, xk, yk, dyk, xnext=None):
        self.xk = float(xk)
        self.yk = float(yk)
        self.dyk = float(dyk)
        self.xnext = None if xnext is None else float(xnext)
        self.set_overlay_visible(True)
        self._update_overlay_geometry()
        self.canvas.draw_idle()

    def _update_overlay_geometry(self):
        if self.f is None or self.xk is None:
            return
        xk, yk, dyk = self.xk, self.yk, self.dyk
        self.scatter_xk.set_offsets(np.array([[xk, yk]], dtype=float))
        self.ann_xk.xy = (xk, yk)

        xlim = self.ax.get_xlim()
        if np.isfinite(yk) and np.isfinite(dyk):
            tx = np.linspace(xlim[0], xlim[1], 200)
            ty = yk + dyk * (tx - xk)
            self.tangent_line.set_data(tx, ty)
        else:
            self.tangent_line.set_data([], [])

        if self.xnext is not None and np.isfinite(self.xnext):
            xn = self.xnext
            self.scatter_xn.set_offsets(np.array([[xn, 0.0]], dtype=float))
            self.ann_xn.xy = (xn, 0.0)
            self.vline_xn.set_xdata([xn, xn])
            self.scatter_xn.set_visible(True)
            self.ann_xn.set_visible(True)
            self.vline_xn.set_visible(True)
        else:
            self.scatter_xn.set_visible(False)
            self.ann_xn.set_visible(False)
            self.vline_xn.set_visible(False)

#Hier Regula Falsi
class RegulaFalsiPlotter(ABPlotter):
    def __init__(self, parent):
        self.xnew = None
        super().__init__(parent)

    def _init_overlay(self):
        super()._init_overlay()
        (self.sec_line,) = self.ax.plot([], [], linestyle="--", linewidth=1.2)
        self.scatter_x = self.ax.scatter([], [], s=60)
        self.ann_x = self.ax.annotate("x", xy=(0, 0), xytext=(6, 6), textcoords="offset points")
        self.vline_x = self.ax.axvline(0, linestyle="--", linewidth=1)
        self._overlay_artists += [self.sec_line, self.scatter_x, self.ann_x, self.vline_x]

    def clear_state(self):
        self.xnew = None
        self.clear_ab()
        self.sec_line.set_data([], [])
        self.canvas.draw_idle()

    def set_state(self, a, b, xnew=None):
        self.set_ab(a, b)
        self.xnew = None if xnew is None else float(xnew)
        self._update_overlay_geometry()
        self.canvas.draw_idle()

    def _update_overlay_geometry(self):
        super()._update_overlay_geometry()
        if self.f is None or self.a is None or self.b is None:
            return
        a, b = self.a, self.b
        try:
            fa = float(self.f(a))
            fb = float(self.f(b))
        except Exception:
            fa, fb = np.nan, np.nan

        xlim = self.ax.get_xlim()
        if np.isfinite(fa) and np.isfinite(fb) and (b != a):
            sx = np.linspace(xlim[0], xlim[1], 200)
            slope = (fb - fa) / (b - a)
            sy = fb + slope * (sx - b)
            self.sec_line.set_data(sx, sy)
        else:
            self.sec_line.set_data([], [])

        if self.xnew is not None and np.isfinite(self.xnew):
            xn = self.xnew
            self.scatter_x.set_offsets(np.array([[xn, 0.0]], dtype=float))
            self.ann_x.xy = (xn, 0.0)
            self.vline_x.set_xdata([xn, xn])
            self.scatter_x.set_visible(True)
            self.ann_x.set_visible(True)
            self.vline_x.set_visible(True)
        else:
            self.scatter_x.set_visible(False)
            self.ann_x.set_visible(False)
            self.vline_x.set_visible(False)

#Hier Sekantenverfahren
class SecantPlotter(DynamicFunctionPlot):
    def __init__(self, parent):
        self.xprev = None
        self.xcur = None
        self.xnext = None
        super().__init__(parent)

    def _init_overlay(self):
        self.scatter_pts = self.ax.scatter([], [], s=60)
        self.ann_prev = self.ax.annotate("xk-1", xy=(0, 0), xytext=(6, 6), textcoords="offset points")
        self.ann_cur = self.ax.annotate("xk", xy=(0, 0), xytext=(6, 6), textcoords="offset points")
        (self.sec_line,) = self.ax.plot([], [], linestyle="--", linewidth=1.2)
        self.scatter_next = self.ax.scatter([], [], s=60)
        self.ann_next = self.ax.annotate("xk+1", xy=(0, 0), xytext=(6, 6), textcoords="offset points")
        self.vline_next = self.ax.axvline(0, linestyle="--", linewidth=1)
        self._overlay_artists = [
            self.scatter_pts, self.ann_prev, self.ann_cur, self.sec_line,
            self.scatter_next, self.ann_next, self.vline_next
        ]

    def clear_state(self):
        self.xprev = self.xcur = self.xnext = None
        self.set_overlay_visible(False)
        self.sec_line.set_data([], [])
        self.canvas.draw_idle()

    def set_state(self, xprev, xcur, xnext=None):
        self.xprev, self.xcur = float(xprev), float(xcur)
        self.xnext = None if xnext is None else float(xnext)
        self.set_overlay_visible(True)
        self._update_overlay_geometry()
        self.canvas.draw_idle()

    def _update_overlay_geometry(self):
        if self.f is None or self.xprev is None or self.xcur is None:
            return
        xp, xc = self.xprev, self.xcur
        try:
            fp = float(self.f(xp))
            fc = float(self.f(xc))
        except Exception:
            fp, fc = np.nan, np.nan

        self.scatter_pts.set_offsets(np.array([[xp, fp], [xc, fc]], dtype=float))
        self.ann_prev.xy = (xp, fp)
        self.ann_cur.xy = (xc, fc)

        xlim = self.ax.get_xlim()
        if np.isfinite(fp) and np.isfinite(fc) and (xc != xp):
            sx = np.linspace(xlim[0], xlim[1], 200)
            slope = (fc - fp) / (xc - xp)
            sy = fc + slope * (sx - xc)
            self.sec_line.set_data(sx, sy)
        else:
            self.sec_line.set_data([], [])

        if self.xnext is not None and np.isfinite(self.xnext):
            xn = self.xnext
            self.scatter_next.set_offsets(np.array([[xn, 0.0]], dtype=float))
            self.ann_next.xy = (xn, 0.0)
            self.vline_next.set_xdata([xn, xn])
            self.scatter_next.set_visible(True)
            self.ann_next.set_visible(True)
            self.vline_next.set_visible(True)
        else:
            self.scatter_next.set_visible(False)
            self.ann_next.set_visible(False)
            self.vline_next.set_visible(False)

#Hier Heron-Verfahren
class HeronPlotter(DynamicFunctionPlot):
    def __init__(self, parent):
        self.S = None
        super().__init__(parent, xlabel="x", ylabel="y")
        # y=x Linie zusätzlich
        (self.line_yx,) = self.ax.plot([], [], linewidth=1.2,
                                        linestyle="--", label="y=x")
        self.ax.legend()

    def _init_overlay(self):
        self.pt_r    = self.ax.scatter([], [], s=60)
        self.vline_r = self.ax.axvline(0, linestyle="--", linewidth=1)
        self.hline_r = self.ax.axhline(0, linestyle="--", linewidth=1)
        self._overlay_artists = [self.pt_r, self.vline_r, self.hline_r]

    def clear_state(self):
        self.S = None
        self.set_function(None)
        self.curve_line.set_label("g(x)")  # Label zurücksetzen
        self.line_yx.set_data([], [])
        self.set_overlay_visible(False)
        self.canvas.draw_idle()

    def set_S(self, S: float):
        self.S = float(S)
        r = float(np.sqrt(S)) if S >= 0 else 0.0
        pad = max(5.0, 5.0 * (abs(r) if r != 0 else 1.0))

        def g(x):
            return 0.5 * (x + self.S / x)

        self.set_function(g)
        self.curve_line.set_label("g(x)")
        self.set_view(-pad, pad, -pad, pad)
        self.set_overlay_visible(True)
        self.refresh()

    def _update_overlay_geometry(self):
        if self.S is None:
            return
        # y=x Linie mitziehen
        xlim = self.ax.get_xlim()
        xs = np.linspace(xlim[0], xlim[1], 200)
        self.line_yx.set_data(xs, xs)

        # Fixpunkt bei sqrt(S)
        r = float(np.sqrt(self.S))
        self.pt_r.set_offsets(np.array([[r, r]], dtype=float))
        self.vline_r.set_xdata([r, r])
        self.hline_r.set_ydata([r, r])

