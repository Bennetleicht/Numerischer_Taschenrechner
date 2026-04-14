from __future__ import annotations
import tkinter as tk
from tkinter import ttk
import sys
import os
from typing import Callable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plotter')))
from dgl_plotter import ComparisonPlot


class ComparisonWindow(tk.Toplevel):
    def __init__(
        self,
        parent,
        on_clear_callback: Callable[[], None] | None = None,
        on_close_callback: Callable[[], None] | None = None,
    ):
        super().__init__(parent)
        self.parent_gui = parent
        self._on_clear_callback = on_clear_callback
        self._on_close_callback = on_close_callback

        self.title("Vergleich der Verfahren")
        self.geometry("900x600")
        self.configure(bg="#f6f7fb")

        self._closed = False
        self._is_topmost = False

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top_bar = ttk.Frame(self, padding=(10, 10, 10, 0))
        top_bar.grid(row=0, column=0, sticky="ew")
        top_bar.columnconfigure(0, weight=1)

        self.lock_btn = ttk.Button(
            top_bar,
            text="🔓",
            width=3,
            command=self._toggle_topmost
        )
        self.lock_btn.grid(row=0, column=0, sticky="w", padx=(0, 8))

        self.clear_btn = ttk.Button(
            top_bar,
            text="Plots leeren",
            command=self._on_clear
        )
        self.clear_btn.grid(row=0, column=1, sticky="e")

        self.plotter = ComparisonPlot(self, xlabel="t", ylabel="y(t)")
        self.plotter.widget().grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

    def _toggle_topmost(self):
        self._is_topmost = not self._is_topmost
        self.attributes("-topmost", self._is_topmost)
        self.lock_btn.config(text="🔒" if self._is_topmost else "🔓")

    def _on_close(self):
        self._closed = True

        if self._on_close_callback is not None:
            try:
                self._on_close_callback()
            except Exception:
                pass

        self.destroy()

    def is_closed(self):
        return self._closed or not self.winfo_exists()

    def add_solution(self, ts, ys, label: str):
        return self.plotter.add_solution(ts, ys, label)

    def _on_clear(self):
        self.plotter.clear_all()

        if self._on_clear_callback is not None:
            try:
                self._on_clear_callback()
            except Exception:
                pass