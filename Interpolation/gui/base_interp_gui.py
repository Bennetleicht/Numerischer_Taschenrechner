
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional

from PIL import ImageTk
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Nullstellen')))
from Nullstellen.plotters_mpl import InterpolationPlotter

from base_gui.gui_utils import ScrollableFrame, maximize_window

# Render-Funktionen zentral in base_gui.latex_renderer 
# Werden hier re-exportiert, damit bestehende Importe aus gui.base_interp_gui
# weiterhin funktionieren, ohne Änderungen in allen Unterklassen.
from base_gui.latex_renderer import (          
    render_formula_block,
    render_formula,
    render_matrix_img,
)



# Basis GUI-Klass
class BaseInterpGUI(tk.Tk):
    """
    Gemeinsame Basis für LagrangeGUI, PolynomGUI, SplineGUI, BezierGUI.

    Geteilt (identisch in allen 4 GUIs):
      - _init_style()
      - _build_ui()        Kopfzeile + Layout + Plot-Frame
      - _draw_empty_plot()
      - _add_step_card()   LaTeX-Karte mit optionalem Matrix-Bild
      - _on_step_btn()
      Boilerplate in _do_step / _do_start via Hilfsmethoden

    Unterklassen überschreiben:
      - _do_start()        Pflicht
      - _do_step()         Pflicht
      - _on_reset()        Pflicht (rufe _on_reset_base() für gemeinsamen Teil)
      - _update_plot()     Optional
      - _render_header()   Optional (wird nach Start/Reset aufgerufen)
      - _extra_header_widgets(top, col) - int   Optional (zusätzliche Header-Widgets)
    """

    def __init__(self):
        super().__init__()
        # method muss vor _build_ui existieren (wird von Unterklasse gesetzt)
        self.started    = False
        self.step_count = 0
        self._imgs: list = []         
        self._point_entries: list[tuple[tk.Entry, tk.Entry]] = []
        self._point_count_var = tk.StringVar(value="4")
        self._init_style()
        self._build_ui()
        maximize_window(self)
        self.title(self.method.title)


    # Style (identisch in allen 4 GUIs) 
    def _init_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        self.bg_app  = "#f6f7fb"
        self.bg_card = "#ffffff"
        self.bg_head = "#f3f4f6"
        self.c_hi    = "#fef9c3"
        self.c_done  = "#dcfce7"
        style.configure("App.TFrame",   background=self.bg_app)
        style.configure("Title.TLabel", background=self.bg_app,
                        font=("Segoe UI", 13, "bold"))
        style.configure("Hint.TLabel",  background=self.bg_app,
                        font=("Segoe UI", 9), foreground="#4b5563")
        style.configure("TButton",      font=("Segoe UI", 10))

    # UI-Grundstruktur identisch
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.configure(bg=self.bg_app)

        #  Kopfzeile 
        top = ttk.Frame(self, padding=(10, 8), style="App.TFrame")
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text=self.method.title,
                  style="Title.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

        # Eingabefelder aus method.input_fields()
        self._input_vars: dict[str, tk.StringVar] = {}
        self._input_widgets: dict[str, ttk.Entry] = {}
        row1 = ttk.Frame(top, style="App.TFrame")
        row1.grid(row=1, column=0, sticky="w")
        col = 0
        hidden_keys = {"x", "f", "points"} if self._uses_point_grid() else set()
        visible_fields = [fld for fld in self.method.input_fields() if fld[0] not in hidden_keys]
        for key, label, default, width in visible_fields:
            ttk.Label(row1, text=label, style="Hint.TLabel").grid(
                row=0, column=col, sticky="w", padx=(0, 4))
            col += 1
            var = tk.StringVar(value=default)
            self._input_vars[key] = var
            entry = ttk.Entry(row1, textvariable=var, width=width)
            entry.grid(row=0, column=col, sticky="w", padx=(0, 12))
            self._input_widgets[key] = entry
            col += 1

        # hook
        col = self._extra_header_widgets(row1, col)
        if col == 0:
            row1.grid_remove()

        # Start/Reset-Buttons rechts
        btn_frame = ttk.Frame(top, style="App.TFrame")
        btn_frame.grid(row=1, column=1, sticky="e", padx=(10, 0))
        self._btn_step = ttk.Button(btn_frame, text="Start",
                                    command=self._on_step_btn)
        self._btn_step.pack(side="left", padx=(0, 6))
        ttk.Button(btn_frame, text="Reset",
                   command=self._on_reset).pack(side="left")

        # Punkt-Grid + Header/Formel in gemeinsamem Container,
        # damit beides direkt nebeneinander bleibt und nicht an den rechten Rand springt
        content_row = 1 if not row1.winfo_ismapped() else 2
        self._header_content = ttk.Frame(top, style="App.TFrame")
        self._header_content.grid(row=content_row, column=0, columnspan=2,
                                  sticky="w", pady=(0, 0))
        self._header_content.columnconfigure(0, weight=0)
        self._header_content.columnconfigure(1, weight=0)

        self._points_box = ttk.Frame(self._header_content, style="App.TFrame")
        self._points_box.grid(row=0, column=0, sticky="nw")
        if self._uses_point_grid():
            self._build_point_grid_controls(self._points_box)

        self._hdr_frame = ttk.Frame(self._header_content, style="App.TFrame")
        self._hdr_frame.grid(row=0, column=1, sticky="nw", padx=(12, 0), pady=(0, 0))
        self._hdr_canvas = tk.Canvas(self._hdr_frame, bg=self.bg_app,
                                     highlightthickness=0, height=2)
        self._hdr_canvas.pack(anchor="nw")
        self._render_header()

        # Hauptbereich
        main = ttk.Frame(self, padding=(10, 0, 10, 10), style="App.TFrame")
        main.grid(row=1, column=0, sticky="nsew")
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        # Links: Schrittdetails
        left = ttk.Frame(main, style="App.TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)

        ttk.Label(left, text="Schrittdetails",
                  style="Title.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 6))
        self._history = ScrollableFrame(left)
        self._history.grid(row=1, column=0, sticky="nsew")

        # Rechts: Plot
        right = ttk.Frame(main, style="App.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        ttk.Label(right, text="Plot",
                  style="Title.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 6))

        self.plotter = InterpolationPlotter(right)
        self.plotter.widget().grid(row=1, column=0, sticky="nsew")
        self._draw_empty_plot()

    # Hooks (Unterklassen überschreiben nach Bedarf)

    def _extra_header_widgets(self, row1: tk.Frame, col: int) -> int:
        return col

    def _render_header(self):
        pass

    def _lock_controls(self, locked: bool):
        state = "disabled" if locked else "normal"
        for w in self._input_widgets.values():
            try:
                w.config(state=state)
            except Exception:
                pass

        if hasattr(self, "_point_spin"):
            try:
                self._point_spin.config(state=state)
            except Exception:
                pass

        for ex, ey in getattr(self, "_point_entries", []):
            try:
                ex.config(state=state)
                ey.config(state=state)
            except Exception:
                pass

    # Plot
    def _draw_empty_plot(self):
        self.plotter.clear_plot()
        self.plotter.ax.set_facecolor(self.bg_app)
        self.plotter.ax.set_title(self.method.title, fontsize=11)
        self.plotter.ax.grid(True, linestyle="--", alpha=0.5)
        self.plotter.ax.text(
            0.5, 0.5,
            "Funktion und Parameter eingeben\nund Start drücken",
            ha="center", va="center",
            transform=self.plotter.ax.transAxes,
            fontsize=11, color="#9ca3af"
        )
        self.plotter.canvas.draw_idle()

    def _update_plot(self, **kwargs):
        # Überschreiben in Unterklassen für verfahrensspezifische Plots
        pass

    # Schritt-Karten
    def _add_step_card(self,
                       step_nr: int,
                       formula_lines: list[tuple[str, int]],
                       title: str,
                       bg: str,
                       matrix_img: Optional[ImageTk.PhotoImage] = None):
        """
        Fügt eine LaTeX-Schritt-Karte zur History hinzu

        formula_lines: Liste von (latex_str, fontsize)
        matrix_img:    optionales gerendertes Matrixbild (rechts neben Formeln)
        """
        card = tk.Frame(self._history.inner, bg=bg, relief="solid", bd=1)
        card.pack(fill="x", padx=4, pady=3)

        tk.Label(card, text=title, bg=self.bg_head,
                 font=("Segoe UI", 10, "bold"), anchor="w",
                 padx=8, pady=3).pack(fill="x")

        if formula_lines or matrix_img:
            body = tk.Frame(card, bg=bg)
            body.pack(fill="x", padx=4, pady=3)

            if formula_lines:
                w_in = 3.8 if matrix_img is not None else 7.0
                fimg = render_formula_block(formula_lines, bg=bg,
                                            dpi=100, width_in=w_in)
                self._imgs.append(fimg)
                tk.Label(body, image=fimg, bg=bg,
                         anchor="nw").grid(row=0, column=0, sticky="nw",
                                           padx=(2, 4))

            if matrix_img is not None:
                sep = tk.Frame(body, bg="#d1d5db", width=1)
                sep.grid(row=0, column=1, sticky="ns", padx=(2, 4))
                tk.Label(body, image=matrix_img, bg=bg,
                         anchor="nw").grid(row=0, column=2, sticky="nw")

        self._history.canvas.update_idletasks()
        self._history.canvas.yview_moveto(1.0)

    # Button-Handler 
    def _on_step_btn(self):
        # Identisch in allen 4 GUIs
        if not self.started:
            self._do_start()
        else:
            self._do_step()

    # Gemeinsamer Reset-Boilerplate
    def _on_reset_base(self):
        """
        Gemeinsamer Boilerplate für Reset: state zurücksetzen, History leeren, Plot leeren
        Unterklassen rufen dies in _on_reset() auf
        """
        self.started    = False
        self.step_count = 0
        self._btn_step.config(text="Start")
        self._lock_controls(False)
        self._history.clear()
        self._imgs.clear()
        self._render_header()
        self._draw_empty_plot()

    # Abstrakte Methoden (Unterklasse muss implementieren) 
    def _do_start(self):
        raise NotImplementedError

    def _do_step(self):
        raise NotImplementedError

    def _on_reset(self):
        raise NotImplementedError
    
    # Hilfsmethoden
    def _uses_point_grid(self) -> bool:
        return False

    def _default_points(self) -> list[tuple[float, float]]:
        return [(0, 1), (1, 2), (2, 0), (3, 3)]

    def _validate_point_count(self, proposed: str) -> bool:
        return proposed == "" or (proposed.isdigit() and 2 <= int(proposed) <= 10)

    def _get_point_count(self) -> int:
        try:
            n = int(self._point_count_var.get())
            return n if 2 <= n <= 10 else 4
        except ValueError:
            return 4

    def _build_point_grid_controls(self, parent: ttk.Frame):
        old_values = []
        for ex, ey in self._point_entries:
            old_values.append((ex.get(), ey.get()))
        for w in parent.winfo_children():
            w.destroy()
        self._point_entries.clear()

        controls = ttk.Frame(parent, style="App.TFrame")
        controls.pack(fill="x", pady=(0, 2))

        ttk.Label(controls, text="Anzahl Punkte:", style="Hint.TLabel").pack(side="left")
        self._point_spin = ttk.Spinbox(
            controls, from_=2, to=10,
            textvariable=self._point_count_var, width=5,
            command=self._rebuild_point_grid
        )
        self._point_spin.pack(side="left", padx=(6, 12))

        vcmd = (self.register(self._validate_point_count), "%P")
        self._point_spin.configure(validate="key", validatecommand=vcmd)
        self._point_spin.bind("<Return>", lambda _e: self._rebuild_point_grid())
        self._point_spin.bind("<FocusOut>", lambda _e: self._rebuild_point_grid())

        box = ttk.Frame(parent, padding=(0, 2, 0, 0), style="App.TFrame")
        box.pack(fill="x")

        ttk.Label(box, text="Punkteingabe:", style="Hint.TLabel").pack(anchor="w", pady=(0, 4))

        grid = ttk.Frame(box, style="App.TFrame")
        grid.pack(anchor="w")

        ttk.Label(grid, text="Punkt", style="Hint.TLabel").grid(row=0, column=0, padx=3, pady=2)
        ttk.Label(grid, text="x", style="Hint.TLabel").grid(row=0, column=1, padx=3, pady=2)
        ttk.Label(grid, text="y / f(x)", style="Hint.TLabel").grid(row=0, column=2, padx=3, pady=2)

        defaults = self._default_points()
        n = self._get_point_count()

        for i in range(n):
            ttk.Label(grid, text=f"P{i}", style="Hint.TLabel").grid(row=i+1, column=0, padx=3, pady=3)

            ex = tk.Entry(grid, width=10, justify="center",
                          relief="solid", borderwidth=1, font=("Consolas", 11))
            ey = tk.Entry(grid, width=10, justify="center",
                          relief="solid", borderwidth=1, font=("Consolas", 11))
            ex.grid(row=i+1, column=1, padx=3, pady=3)
            ey.grid(row=i+1, column=2, padx=3, pady=3)

            if i < len(old_values):
                ex.insert(0, old_values[i][0])
                ey.insert(0, old_values[i][1])
            elif i < len(defaults):
                ex.insert(0, str(defaults[i][0]))
                ey.insert(0, str(defaults[i][1]))
            else:
                ex.insert(0, "0")
                ey.insert(0, "0")

            self._point_entries.append((ex, ey))

    def _on_point_count_changed(self, *_args):
        pass

    def _rebuild_point_grid(self):
        if hasattr(self, "_points_box"):
            self._build_point_grid_controls(self._points_box)

    def _read_point_pairs(self) -> list[tuple[float, float]]:
        pts = []
        for i, (ex, ey) in enumerate(self._point_entries, start=1):
            try:
                x = float(ex.get().replace(",", "."))
            except ValueError:
                raise ValueError(f"Ungültiger x-Wert bei Punkt P{i-1}")
            try:
                y = float(ey.get().replace(",", "."))
            except ValueError:
                raise ValueError(f"Ungültiger y-Wert bei Punkt P{i-1}")
            pts.append((x, y))
        return pts