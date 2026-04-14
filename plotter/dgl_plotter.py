from __future__ import annotations
import numpy as np

from base_plotter import DynamicFunctionPlot

class DiscretePlot(DynamicFunctionPlot):
    def _resample_now(self):
        """Überschreibt die Generic-Logik: keine Funktionsabtastung nötig."""
        pass

    def _on_scroll_zoom(self, event):

        mode = self._scroll_mode(event)
        if mode is None:
            return

        base = 1.15
        if event.button == "up":
            scale = 1 / base
        elif event.button == "down":
            scale = base
        else:
            return

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        if cur_xlim[1] == cur_xlim[0] or cur_ylim[1] == cur_ylim[0]:
            return

        xcenter = event.xdata if event.xdata is not None else 0.5 * (cur_xlim[0] + cur_xlim[1])
        ycenter = event.ydata if event.ydata is not None else 0.5 * (cur_ylim[0] + cur_ylim[1])

        if mode in ("both", "xaxis"):
            new_w = (cur_xlim[1] - cur_xlim[0]) * scale
            relx = (xcenter - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
            self.ax.set_xlim(xcenter - new_w * relx, xcenter + new_w * (1 - relx))

        if mode in ("both", "yaxis"):
            new_h = (cur_ylim[1] - cur_ylim[0]) * scale
            rely = (ycenter - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0])
            self.ax.set_ylim(ycenter - new_h * rely, ycenter + new_h * (1 - rely))

        self.canvas.draw_idle()

    def _on_mouse_press(self, event):
        """Pan auch ohne gesetzte Funktion erlauben."""
        if event.button != 1 or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        self._panning = True
        self._pan_press = (event.xdata, event.ydata, self.ax.get_xlim(), self.ax.get_ylim())

    def _on_mouse_release(self, event):
        if event.button != 1:
            return
        if not self._panning:
            return

        self._panning = False
        self._pan_press = None

    def _on_mouse_move(self, event):
        if not self._panning:
            return
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        xpress, ypress, (xmin0, xmax0), (ymin0, ymax0) = self._pan_press
        dx = event.xdata - xpress
        dy = event.ydata - ypress

        new_xlim = (xmin0 - dx, xmax0 - dx)
        new_ylim = (ymin0 - dy, ymax0 - dy)

        self._pending_limits = (new_xlim, new_ylim)
        if self._pan_after_id is None:
            self._pan_after_id = self.widget().after(self._pan_delay_ms, self._on_pan_timer)    

#Hier DGL
class DGL_Plotter(DiscretePlot):

    def __init__(self, parent, xlabel: str = "t", ylabel: str = "y(t)"):
        super().__init__(parent, xlabel=xlabel, ylabel=ylabel)

        self.f = None
        self.curve_line.set_data([], [])
        self.curve_line.set_linewidth(1.8)
        self.curve_line.set_marker("o")
        self.curve_line.set_markersize(5)
        self.curve_line.set_label("Näherung")

        self.ax.legend(loc="best", fontsize=9)
        self.canvas.draw_idle()

    def set_points(self, ts, ys):
        """Setzt die aktuell bekannten Stützstellen."""
        ts = np.asarray(ts, dtype=float)
        ys = np.asarray(ys, dtype=float)
        self.curve_line.set_data(ts, ys)
        self.fit_view()
        self.canvas.draw_idle()

    def append_point(self, t: float, y: float):
        """Hängt einen weiteren Punkt an die vorhandene Lösung an."""
        old_x = np.asarray(self.curve_line.get_xdata(), dtype=float)
        old_y = np.asarray(self.curve_line.get_ydata(), dtype=float)

        new_x = np.append(old_x, float(t))
        new_y = np.append(old_y, float(y))

        self.curve_line.set_data(new_x, new_y)
        self.fit_view()
        self.canvas.draw_idle()

    def clear(self):
        """Entfernt alle Punkte und setzt den Plot zurück."""
        self.curve_line.set_data([], [])
        self.ax.set_title("")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(-1, 1)
        self.canvas.draw_idle()

    def set_title(self, title: str):
        """Setzt den Plot-Titel."""
        self.ax.set_title(title, fontsize=10)
        self.canvas.draw_idle()

    def fit_view(self, margin: float = 0.1):
        """Passt die Achsen an alle aktuell vorhandenen Punkte an."""
        ts = np.asarray(self.curve_line.get_xdata(), dtype=float)
        ys = np.asarray(self.curve_line.get_ydata(), dtype=float)

        if ts.size == 0 or ys.size == 0:
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(-1, 1)
            return

        finite_mask = np.isfinite(ts) & np.isfinite(ys)
        ts = ts[finite_mask]
        ys = ys[finite_mask]

        if ts.size == 0 or ys.size == 0:
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(-1, 1)
            return

        xmin = float(np.min(ts))
        xmax = float(np.max(ts))
        ymin = float(np.min(ys))
        ymax = float(np.max(ys))

        if np.isclose(xmin, xmax):
            xmin -= 1.0
            xmax += 1.0
        if np.isclose(ymin, ymax):
            ymin -= 1.0
            ymax += 1.0

        dx = (xmax - xmin) * margin
        dy = (ymax - ymin) * margin

        self.ax.set_xlim(xmin - dx, xmax + dx)
        self.ax.set_ylim(ymin - dy, ymax + dy)

    def refresh(self):
        """Beim Einschrittplot nur neu zeichnen, keine Funktionsabtastung."""
        self.canvas.draw_idle()

    def update_solution(self, ts, ys):
        """Kompatibilitätsmethode für die GUI."""
        self.set_points(ts, ys)
        
    def _init_overlay(self):
        pass

class ComparisonPlot(DiscretePlot):
    def __init__(self, parent, xlabel: str = "t", ylabel: str = "y(t)"):
        super().__init__(parent, xlabel=xlabel, ylabel=ylabel)

        # Die einzelne Kurve aus der Basisklasse wird hier nicht benutzt
        self.curve_line.set_data([], [])
        self.curve_line.set_visible(False)

        self.solution_lines = []
        self.max_curves = 5

        self.ax.legend(loc="best", fontsize=9)
        self.canvas.draw_idle()

    def add_solution(self, ts, ys, label: str):
        ts = np.asarray(ts, dtype=float)
        ys = np.asarray(ys, dtype=float)

        if ts.size == 0 or ys.size == 0:
            return False

        if len(self.solution_lines) >= self.max_curves:
            return False

        (line,) = self.ax.plot(ts, ys, marker="o", linewidth=1.8, markersize=4, label=label)
        self.solution_lines.append(line)
        self.ax.legend(loc="best", fontsize=9)
        self.fit_view()
        self.canvas.draw_idle()
        return True

    def clear_all(self):
        for line in self.solution_lines:
            try:
                line.remove()
            except Exception:
                pass

        self.solution_lines.clear()
        leg = self.ax.get_legend()
        if leg is not None:
            leg.remove()

        self.ax.set_title("")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(-1, 1)
        self.canvas.draw_idle()

    def fit_view(self, margin: float = 0.1):
        if not self.solution_lines:
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(-1, 1)
            return

        all_x = []
        all_y = []

        for line in self.solution_lines:
            xs = np.asarray(line.get_xdata(), dtype=float)
            ys = np.asarray(line.get_ydata(), dtype=float)

            mask = np.isfinite(xs) & np.isfinite(ys)
            xs = xs[mask]
            ys = ys[mask]

            if xs.size > 0 and ys.size > 0:
                all_x.append(xs)
                all_y.append(ys)

        if not all_x or not all_y:
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(-1, 1)
            return

        xs = np.concatenate(all_x)
        ys = np.concatenate(all_y)

        xmin = float(np.min(xs))
        xmax = float(np.max(xs))
        ymin = float(np.min(ys))
        ymax = float(np.max(ys))

        if np.isclose(xmin, xmax):
            xmin -= 1.0
            xmax += 1.0
        if np.isclose(ymin, ymax):
            ymin -= 1.0
            ymax += 1.0

        dx = (xmax - xmin) * margin
        dy = (ymax - ymin) * margin

        self.ax.set_xlim(xmin - dx, xmax + dx)
        self.ax.set_ylim(ymin - dy, ymax + dy)

    def refresh(self):
        self.canvas.draw_idle()

    def _init_overlay(self):
        pass
