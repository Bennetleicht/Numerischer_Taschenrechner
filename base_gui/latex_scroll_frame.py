from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import List
from latex_renderer import render_formula_block, render_formula

class LatexScrollFrame(tk.Frame):
    def __init__(self, parent, bg="#f6f7fb", **kw):
        super().__init__(parent, bg=bg, **kw)
        self.bg = bg
        self._images: List = []
        self._scroll_enabled = False

        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        self.sb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.sb.set)
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner = tk.Frame(self.canvas, bg=bg)
        self._window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

    def _on_inner_configure(self, _event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self._update_scrollbar()

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self._window, width=event.width)
        self._update_scrollbar()

    def _update_scrollbar(self):
        self.update_idletasks()
        inner_h = self.inner.winfo_reqheight()
        canvas_h = self.canvas.winfo_height()
        if inner_h > canvas_h:
            if not self._scroll_enabled:
                self.sb.pack(side="right", fill="y")
                self._scroll_enabled = True
        else:
            if self._scroll_enabled:
                self.sb.pack_forget()
                self._scroll_enabled = False
            self.canvas.yview_moveto(0)

    def _on_mousewheel(self, event):
        if self._scroll_enabled:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _bind_mousewheel(self, _event):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event):
        self.canvas.unbind_all("<MouseWheel>")

    def clear(self):
        for w in self.inner.winfo_children():
            w.destroy()
        self._images.clear()
        self.after(10, self._update_scrollbar)

    def add_heading(self, text: str):
        lbl = tk.Label(self.inner, text=text, font=("Segoe UI", 16, "bold"), fg="#1a6b9e", bg=self.bg, anchor="w", justify="left")
        lbl.pack(fill="x", padx=10, pady=(14, 4))
        return lbl

    def add_latex(self, latex: str, fontsize: int = 13):
        img = render_formula(latex, bg=self.bg, fontsize=fontsize)
        self._images.append(img)
        tk.Label(self.inner, image=img, bg=self.bg, anchor="w").pack(fill="x", padx=10, pady=2)
        return img

    def add_latex_block(self, lines: list, fontsize: int = 13):
        img = render_formula_block([(line, fontsize) for line in lines], bg=self.bg)
        self._images.append(img)
        tk.Label(self.inner, image=img, bg=self.bg, anchor="w").pack(fill="x", padx=10, pady=2)
        return img

    def scroll_bottom(self):
        self.update_idletasks()
        self.canvas.yview_moveto(1.0)
