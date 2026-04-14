from __future__ import annotations
import numpy as np

from base_plotter import DynamicFunctionPlot

# für interpolationsverfahren
class InterpolationPlotter(DynamicFunctionPlot):
    def __init__(self, parent):
        self.points_x = []
        self.points_y = []
        self.curve_x = np.array([], dtype=float)
        self.curve_y = np.array([], dtype=float)
        self.eval_x = None
        self.eval_y = None
        self.eval_label = None
        self.title_text = ""
        self.curve_label = None
        super().__init__(parent, xlabel="x", ylabel="y")

    def _init_overlay(self):
        self.scatter_pts = self.ax.scatter([], [], s=60, color="red", zorder=5)
        self.scatter_eval = self.ax.scatter([], [], s=80, marker="D", color="green", zorder=6)
        self.vline_eval, = self.ax.plot([], [], linestyle=":", linewidth=1, color="green", alpha=0.7)
        self._overlay_artists = [self.scatter_pts, self.scatter_eval, self.vline_eval]

    def clear_plot(self):
        self.points_x = []
        self.points_y = []
        self.curve_x = np.array([], dtype=float)
        self.curve_y = np.array([], dtype=float)
        self.eval_x = None
        self.eval_y = None
        self.eval_label = None
        self.title_text = ""
        self.curve_label = None
        self.f = None

        self.ax.clear()
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.axhline(0, linewidth=1, color="#374151")
        self.ax.grid(True, linewidth=0.6, alpha=0.6)

        (self.curve_line,) = self.ax.plot([], [], linewidth=1.8, color="#2563eb", zorder=3)
        self._init_overlay()
        self.set_overlay_visible(False)

        self.canvas.draw_idle()

    def set_title(self, title: str):
        self.title_text = title or ""

    # zeichnet punkt im plot
    def set_points(self, xs, ys):
        self.points_x = list(xs)
        self.points_y = list(ys)
        self._update_overlay_geometry()
        self.scatter_pts.set_visible(bool(self.points_x))
        self.canvas.draw_idle()

    # zeichnet funktion im plot
    def set_curve(self, xs, ys, label=None):
        self.curve_x = np.array(xs, dtype=float)
        self.curve_y = np.array(ys, dtype=float)
        self.curve_label = label
        self.curve_line.set_data(self.curve_x, self.curve_y)
        self.curve_line.set_visible(True)
        self.curve_line.set_label(label or "_nolegend_")
        self.canvas.draw_idle()

    def _resample_now(self):
        if self.f is not None:
            xmin, xmax = self.ax.get_xlim()
            xs = np.linspace(xmin, xmax, self._desired_samples())
            ys = self._safe_eval(xs)
            self.curve_x = np.array(xs, dtype=float)
            self.curve_y = np.array(ys, dtype=float)
            self.curve_line.set_data(xs, ys)
            return
        if self.curve_x.size > 0:
            self.curve_line.set_data(self.curve_x, self.curve_y)
            return
        self.curve_line.set_data([], [])

    # für bezier interaktiven modus
    def _interactive_enabled(self):
        has_points = bool(self.points_x) and bool(self.points_y)
        has_curve = self.curve_x.size > 0
        has_function = self.f is not None
        has_legend_only = self.ax.get_legend() is not None
        return has_function or has_curve or has_points or has_legend_only

    # scrollen
    def _on_scroll_zoom(self, event):
        if not self._interactive_enabled():
            return

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

        self._update_overlay_geometry()
        self.canvas.draw_idle()
        if self.f is not None:
            self._schedule_resample()

    def _on_mouse_press(self, event):
        if not self._interactive_enabled():
            return
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
        if self.f is not None:
            self._schedule_resample()

    def set_eval_point(self, x=None, y=None, label=None):
        self.eval_x = None if x is None else float(x)
        self.eval_y = None if y is None else float(y)
        self.eval_label = label
        self._update_overlay_geometry()
        self.canvas.draw_idle()

    def clear_eval_point(self):
        self.eval_x = None
        self.eval_y = None
        self.eval_label = None
        self._update_overlay_geometry()
        self.canvas.draw_idle()

    def auto_view(self, pad_ratio=0.15, min_pad=0.5):
        xs = []
        ys = []

        if self.curve_x.size > 0:
            xs.extend(self.curve_x.tolist())
        if self.curve_y.size > 0:
            ys.extend(self.curve_y[np.isfinite(self.curve_y)].tolist())

        xs.extend([x for x in self.points_x if np.isfinite(x)])
        ys.extend([y for y in self.points_y if np.isfinite(y)])

        if self.eval_x is not None and np.isfinite(self.eval_x):
            xs.append(self.eval_x)
        if self.eval_y is not None and np.isfinite(self.eval_y):
            ys.append(self.eval_y)

        if not xs:
            self.ax.set_xlim(-1, 1)
            self.ax.set_ylim(-1, 1)
            return

        xmin, xmax = min(xs), max(xs)
        ymin = min(ys) if ys else -1.0
        ymax = max(ys) if ys else 1.0

        if np.isclose(xmin, xmax):
            xmin -= 1.0
            xmax += 1.0
        if np.isclose(ymin, ymax):
            ymin -= 1.0
            ymax += 1.0

        xpad = max(min_pad, (xmax - xmin) * pad_ratio)
        ypad = max(min_pad, (ymax - ymin) * pad_ratio)

        self.ax.set_xlim(xmin - xpad, xmax + xpad)
        self.ax.set_ylim(ymin - ypad, ymax + ypad)

    def refresh(self):
        if self.title_text:
            self.ax.set_title(self.title_text, fontsize=11)
        if self.curve_label:
            self.curve_line.set_label(self.curve_label)
        self._resample_now()
        self._update_overlay_geometry()
        self.canvas.draw_idle()

    def _update_overlay_geometry(self):
        if self.points_x and self.points_y and len(self.points_x) == len(self.points_y):
            pts = np.column_stack([self.points_x, self.points_y]).astype(float)
            self.scatter_pts.set_offsets(pts)
            self.scatter_pts.set_visible(True)
        else:
            self.scatter_pts.set_offsets(np.empty((0, 2)))
            self.scatter_pts.set_visible(False)

        if self.eval_x is not None and self.eval_y is not None:
            self.scatter_eval.set_offsets(np.array([[self.eval_x, self.eval_y]], dtype=float))
            self.scatter_eval.set_visible(True)
            self.vline_eval.set_data([self.eval_x, self.eval_x], [0, self.eval_y])
            self.vline_eval.set_visible(True)
        else:
            self.scatter_eval.set_offsets(np.empty((0, 2)))
            self.scatter_eval.set_visible(False)
            self.vline_eval.set_data([], [])
            self.vline_eval.set_visible(False)
  