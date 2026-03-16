from __future__ import annotations
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

#Plotter mit Matplotlib, der dynamisch Funktionen darstellen kann und Zoom & Pan unterstützt.
class DynamicFunctionPlot:
    def __init__(self, parent, xlabel="x", ylabel="f(x)"):
        self.fig = Figure(figsize=(6.5, 4.8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.axhline(0, linewidth=1)
        self.ax.grid(True, linewidth=0.6, alpha=0.6)

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)

        self.f = None  

        (self.curve_line,) = self.ax.plot([], [], linewidth=1.8)

        self._min_samples = 800
        self._max_samples = 7000
        self._samples_per_pixel = 1.0

        # pan 
        self._panning = False
        self._pan_press = None
        self._pending_limits = None
        self._pan_after_id = None
        self._pan_delay_ms = 25  # ~40 fps

        # zoom 
        self._resample_after_id = None
        self._resample_delay_ms = 120

        # overlays
        self._overlay_artists = []
        self._init_overlay()
        self.set_overlay_visible(False)

        # connect events
        self.canvas.mpl_connect("scroll_event", self._on_scroll_zoom)
        self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

    def widget(self):
        return self.canvas.get_tk_widget()

    def set_function(self, f):
        self.f = f

    def set_view(self, xmin, xmax, ymin=None, ymax=None):
        """Setzt die Ansichtsgrenzen.

        Falls ymin/ymax fehlen, passt y automatisch an die aktuelle Funktion 
        im Bereich [xmin,xmax] an - mit Rand. Sonst sehen viele Plots 'falsch' aus, 
        weil die Kurve außerhalb des Standard-y-Bereichs liegt.
        """
        xmin = float(xmin)
        xmax = float(xmax)
        self.ax.set_xlim(xmin, xmax)

        if ymin is not None and ymax is not None:
            self.ax.set_ylim(float(ymin), float(ymax))
            return

        if self.f is None:
            return
        xs = np.linspace(xmin, xmax, min(1500, self._desired_samples()))
        ys = self._safe_eval(xs)
        finite = ys[np.isfinite(ys)]
        if finite.size == 0:
            self.ax.set_ylim(-1.0, 1.0)
            return
        ymin2 = float(np.nanmin(finite))
        ymax2 = float(np.nanmax(finite))
        if np.isclose(ymin2, ymax2):
            ymin2 -= 1.0
            ymax2 += 1.0
        margin = 0.15 * (ymax2 - ymin2)
        self.ax.set_ylim(ymin2 - margin, ymax2 + margin)

    def refresh(self):
        self._resample_now()
        self._update_overlay_geometry()
        self.canvas.draw_idle()

    
    def _desired_samples(self) -> int:
        w, _ = self.canvas.get_width_height()
        n = int(self._samples_per_pixel * max(1, w))
        return int(max(self._min_samples, min(self._max_samples, n)))

    # Bei manchen Funktionen kann f(xs) fehlschlagen, 
    # obwohl f(float(x)) für einzelne Werte funktioniert. Daher versuchen wir beides.
    def _safe_eval(self, xs: np.ndarray) -> np.ndarray:
        try:
            ys = self.f(xs)
        except Exception:
            ys = np.array([float(self.f(float(x))) for x in xs], dtype=float)
        ys = np.array(ys, dtype=float)
        ys[~np.isfinite(ys)] = np.nan
        return ys

    #Neuberechnung nach Zoom/Pan oder Funktionswechsel
    def _resample_now(self):
        if self.f is None:
            self.curve_line.set_data([], [])
            return
        xmin, xmax = self.ax.get_xlim()
        xs = np.linspace(xmin, xmax, self._desired_samples())
        ys = self._safe_eval(xs)
        self.curve_line.set_data(xs, ys)

    # Planung einer verzögerten Neuberechnung, um nicht bei jedem kleinen Zoom/Pan sofort neu zu zeichnen, sondern erst nach einer kurzen Pause.
    #Vielleicht rausnehmen??? Dann ruckelt es nicht so beim Pan???
    def _schedule_resample(self):
        if self._resample_after_id is not None:
            try:
                self.widget().after_cancel(self._resample_after_id)
            except Exception:
                pass
            self._resample_after_id = None
        self._resample_after_id = self.widget().after(self._resample_delay_ms, self._on_resample_timer)

    # Timer-Callback für die verzögerte Neuberechnung nach Zoom/Pan.
    def _on_resample_timer(self):
        self._resample_after_id = None
        self._resample_now()
        self._update_overlay_geometry()
        self.canvas.draw_idle()

    # zusätzliche grafische Elemente (z.B. Linien, Punkte), die über der Funktionskurve liegen.

    def set_overlay_visible(self, visible: bool):
        for a in self._overlay_artists:
            try:
                a.set_visible(bool(visible))
            except Exception:
                pass

    def _update_overlay_geometry(self):
        pass

    # Zoom mit Scrollen: Je nachdem, ob der Cursor über der Plotfläche, der x-Achse oder der y-Achse ist, 
    # wird entsprechend in beide Richtungen oder nur in einer Richtung gezoomt. Das Zoom-Verhalten orientiert sich an Simulink.
    def _on_scroll_zoom(self, event):
        if self.f is None:
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
        # Simulink-like:
        # - scroll in plot area => zoom x+y
        # - scroll over x-axis => zoom x only
        # - scroll over y-axis => zoom y only
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
        self._schedule_resample()

    # Bestimmt den Scroll-Modus (Zoom-Richtung) basierend auf der Position des Cursors: über der Plotfläche, der x-Achse oder der y-Achse.
    def _scroll_mode(self, event):
        
        if event.inaxes == self.ax:
            return "both"
        if event.x is None or event.y is None:
            return None

        bbox = self.ax.get_window_extent()
        left, right = bbox.x0, bbox.x1
        bottom, top = bbox.y0, bbox.y1
        x, y = event.x, event.y

        # x-axis label/tick area
        if (left <= x <= right) and (bottom - 60 <= y < bottom):
            return "xaxis"

        # y-axis label/tick area
        if (bottom <= y <= top) and (left - 80 <= x < left):
            return "yaxis"

        return None
    
    def _on_mouse_press(self, event):
        if self.f is None:
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
        self._schedule_resample()

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

    def _on_pan_timer(self):
        self._pan_after_id = None
        if self._pending_limits is None:
            return
        (xlim, ylim) = self._pending_limits
        self._pending_limits = None
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self._update_overlay_geometry()
        self.canvas.draw_idle()
        if self._pending_limits is not None:
            self._pan_after_id = self.widget().after(self._pan_delay_ms, self._on_pan_timer)


#Spezialisierte Plotter für die verschiedenen Verfahren, die zusätzliche grafische Elemente (z.B. Linien, Punkte) über der Funktionskurve darstellen,
#um die Schritte der Verfahren zu visualisieren. Alle erben von DynamicFunctionPlot und erweitern die Overlay-Funktionalität.
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