from __future__ import annotations
import numpy as np

from base_plotter import DynamicFunctionPlot

#hier Newton-Cotes-Verfahren
class NewtonCotesPlotter(DynamicFunctionPlot):
    def __init__(self, parent):
        self.a = None
        self.b = None
        self.verfahren = None
        self.m = 1
        self.xs_nodes = None
        self.ys_nodes = None
        self._fills = []
        super().__init__(parent)

    def _init_overlay(self):
        self.vline_a, = self.ax.plot([], [], linestyle="--", linewidth=1, color="C0")
        self.vline_b, = self.ax.plot([], [], linestyle="--", linewidth=1, color="C1")

        self.scatter_ab = self.ax.scatter([], [], s=60, color="red", zorder=5)
        self.scatter_m = self.ax.scatter([], [], s=45, color="green", zorder=5)

        self.ann_a = self.ax.annotate("a", xy=(0, 0), xytext=(6, 6), textcoords="offset points")
        self.ann_b = self.ax.annotate("b", xy=(0, 0), xytext=(6, 6), textcoords="offset points")

        self.connect_line, = self.ax.plot([], [], linestyle="-", linewidth=1.5, color="C3")
        self._vlines_inner = []

        self._overlay_artists = [
            self.vline_a, self.vline_b,
            self.scatter_ab, self.scatter_m,
            self.ann_a, self.ann_b,
            self.connect_line,
        ]

    def _remove_fills(self):
        for p in self._fills:
            try:
                p.remove()
            except Exception:
                pass
        self._fills = []

    def _remove_inner_vlines(self):
        for ln in self._vlines_inner:
            try:
                ln.remove()
            except Exception:
                pass
        self._vlines_inner = []

    def set_a_b(self, a, b, verfahren=None, m=1, xs_nodes=None, ys_nodes=None):
        self.a = float(a)
        self.b = float(b)
        self.verfahren = verfahren
        self.m = int(m)
        self.xs_nodes = xs_nodes
        self.ys_nodes = ys_nodes
        self.set_overlay_visible(True)
        self._update_overlay_geometry()
        self.canvas.draw_idle()

    def clear_ab(self):
        self.a = None
        self.b = None
        self.verfahren = None
        self.xs_nodes = None
        self.ys_nodes = None
        self.m = 1
        self.set_overlay_visible(False)
        self._remove_fills()
        self._remove_inner_vlines()
        self.canvas.draw_idle()

    def _update_overlay_geometry(self):
        if self.f is None or self.a is None or self.b is None:
            return

        a, b, f, m = self.a, self.b, self.f, self.m
        v = self.verfahren

        if self.xs_nodes is not None and self.ys_nodes is not None:
            xs = np.asarray(self.xs_nodes, dtype=float)
            ys = np.asarray(self.ys_nodes, dtype=float)
        else:
            xs, ys = self._compute_nodes(a, b, f, v, m)

        fa = ys[0] if len(ys) > 0 else np.nan
        fb = ys[-1] if len(ys) > 0 else np.nan

        self.vline_a.set_data([a, a], [0, fa])
        self.vline_b.set_data([b, b], [0, fb])
        self.scatter_ab.set_offsets(np.array([[a, fa], [b, fb]], dtype=float))
        self.ann_a.xy = (a, fa)
        self.ann_b.xy = (b, fb)

        inner_xs = xs[1:-1]
        inner_ys = ys[1:-1]
        if len(inner_xs) > 0:
            self.scatter_m.set_offsets(np.column_stack([inner_xs, inner_ys]))
            self.scatter_m.set_visible(True)
        else:
            self.scatter_m.set_offsets(np.empty((0, 2)))
            self.scatter_m.set_visible(False)

        self._remove_fills()
        self._remove_inner_vlines()

        if v == "Trapezregel" and m > 1:
            self._draw_composite_trapez(xs, ys)
        elif v == "Simpsonregel" and m > 2:
            self._draw_composite_simpson(xs, ys, m)
        else:
            self.connect_line.set_data(xs, ys)
            if np.all(np.isfinite(ys)):
                fill = self.ax.fill(
                    list(xs) + [b, a],
                    list(ys) + [0, 0],
                    color="C3", alpha=0.18, zorder=1
                )
                self._fills.extend(fill)

    def _draw_composite_trapez(self, xs, ys):
        colors = ["#2563eb", "#16a34a", "#dc2626", "#d97706", "#7c3aed",
                  "#0891b2", "#be185d", "#65a30d", "#ea580c", "#4f46e5"]

        n = len(xs) - 1
        for i in range(n):
            xi, xi1 = xs[i], xs[i + 1]
            fxi, fxi1 = ys[i], ys[i + 1]
            col = colors[i % len(colors)]

            patch_x = [xi, xi, xi1, xi1]
            patch_y = [0.0, fxi, fxi1, 0.0]
            fill = self.ax.fill(patch_x, patch_y, color=col, alpha=0.20, zorder=1)
            self._fills.extend(fill)

            ln, = self.ax.plot([xi, xi1], [fxi, fxi1], color=col, linewidth=1.5, zorder=3)
            self._vlines_inner.append(ln)

            if i < n - 1:
                vl, = self.ax.plot([xi1, xi1], [0, fxi1],
                                   color="#6b7280", linestyle="--", linewidth=0.8, zorder=2)
                self._vlines_inner.append(vl)

        self.connect_line.set_data([], [])

    def _draw_composite_simpson(self, xs, ys, m):
        colors = ["#2563eb", "#16a34a", "#dc2626", "#d97706", "#7c3aed",
                  "#0891b2", "#be185d", "#65a30d", "#ea580c", "#4f46e5"]

        pairs = m // 2
        for i in range(pairs):
            idx0, idx1, idx2 = 2 * i, 2 * i + 1, 2 * i + 2
            x0, x1, x2 = xs[idx0], xs[idx1], xs[idx2]
            y0, y1, y2 = ys[idx0], ys[idx1], ys[idx2]
            col = colors[i % len(colors)]

            t = np.linspace(x0, x2, 120)
            L0 = (t - x1) * (t - x2) / ((x0 - x1) * (x0 - x2))
            L1 = (t - x0) * (t - x2) / ((x1 - x0) * (x1 - x2))
            L2 = (t - x0) * (t - x1) / ((x2 - x0) * (x2 - x1))
            p = y0 * L0 + y1 * L1 + y2 * L2

            fill = self.ax.fill_between(t, p, 0.0, color=col, alpha=0.22, zorder=1)
            self._fills.append(fill)

            ln, = self.ax.plot(t, p, color=col, linewidth=1.5, zorder=3)
            self._vlines_inner.append(ln)

            vl_m, = self.ax.plot([x1, x1], [0, y1], color=col, linestyle=":", linewidth=1.0, zorder=2)
            self._vlines_inner.append(vl_m)

            if i < pairs - 1:
                vl, = self.ax.plot([x2, x2], [0, y2],
                                   color="#6b7280", linestyle="--", linewidth=0.8, zorder=2)
                self._vlines_inner.append(vl)

        self.connect_line.set_data([], [])

    @staticmethod
    def _compute_nodes(a, b, f, verfahren, m):
        if verfahren in ("Trapezregel", "Simpsonregel"):
            _m = m if (verfahren == "Trapezregel" or m % 2 == 0) else m + 1
            xs = np.linspace(a, b, _m + 1)
        elif verfahren == "3/8-Regel":
            xs = np.array([a, a + (b - a) / 3, a + 2 * (b - a) / 3, b])
        elif verfahren == "Milne-Regel":
            xs = np.array([a, a + (b - a) / 4, a + (b - a) / 2, a + 3 * (b - a) / 4, b])
        else:
            xs = np.array([a, b])

        ys = np.array([float(f(x)) for x in xs])
        return xs, ys
    

#hier Gauss-Legendre-Verfahren
class GaussLegendrePlotter(DynamicFunctionPlot):
    def __init__(self, parent):
        self.a      = None
        self.b      = None
        self.nodes  = None   
        self.fxi    = None   
        self._fill  = None
        super().__init__(parent)

    def _init_overlay(self):
        # Grenzen a und b
        self.vline_a, = self.ax.plot([], [], linestyle="-",  linewidth=1.5, color="C1")
        self.vline_b, = self.ax.plot([], [], linestyle="-",  linewidth=1.5, color="C1")
        # Stützstellen-Punkte
        self.scatter_nodes = self.ax.scatter([], [], s=70, color="red", zorder=5)
        # senkrechte gestrichelte Linien an Stützstellen (max 5)
        self._vlines = []
        for _ in range(5):
            line, = self.ax.plot([], [], linestyle="--", linewidth=1, color="C2")
            self._vlines.append(line)
        self._overlay_artists = (
            [self.vline_a, self.vline_b, self.scatter_nodes] + self._vlines
        )

    def set_nodes(self, a, b, nodes, fxi):
        self.a     = float(a)
        self.b     = float(b)
        self.nodes = list(nodes)
        self.fxi   = list(fxi)
        self.set_overlay_visible(True)
        self._update_overlay_geometry()
        self.canvas.draw_idle()

    def clear_ab(self):
        self.a = self.b = self.nodes = self.fxi = None
        self._remove_fill()
        self.set_overlay_visible(False)
        self.canvas.draw_idle()

    # Reset
    def clear(self):
        self.f     = None
        self.a     = None
        self.b     = None
        self.nodes = None
        self.fxi   = None
        self._fill = None

        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("f(x)")
        self.ax.axhline(0, linewidth=1)
        self.ax.grid(True, linewidth=0.6, alpha=0.6)
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-2, 2)

        
        (self.curve_line,) = self.ax.plot([], [], linewidth=1.8)

        # Overlay neu anlegen
        self._init_overlay()
        self.set_overlay_visible(False)

        self.canvas.draw_idle()

    # interne Hilfsmethoden 
    def _remove_fill(self):
        if self._fill is not None:
            try:
                self._fill.remove()
            except Exception:
                pass
            self._fill = None

    def _update_overlay_geometry(self):
        if self.a is None or self.nodes is None or self.f is None:
            return

        a, b   = self.a, self.b
        nodes  = self.nodes
        fxi    = self.fxi

        # Grenzen
        try:
            fa = float(self.f(a))
            fb = float(self.f(b))
        except Exception:
            fa, fb = 0.0, 0.0
        self.vline_a.set_data([a, a], [0, fa])
        self.vline_b.set_data([b, b], [0, fb])

        # Stützstellen-Punkte
        import numpy as np
        self.scatter_nodes.set_offsets(np.column_stack([nodes, fxi]))

        # senkrechte Linien an Stützstellen
        for i, line in enumerate(self._vlines):
            if i < len(nodes):
                line.set_data([nodes[i], nodes[i]], [0, fxi[i]])
                line.set_visible(True)
            else:
                line.set_visible(False)

        # Fläche unter Kurve zwischen a und b
        self._remove_fill()
        xs_fill = np.linspace(a, b, 400)
        ys_fill = self._safe_eval(xs_fill)
        self._fill = self.ax.fill_between(
            xs_fill, ys_fill, 0,
            alpha=0.18, color="C3", zorder=1
        )
