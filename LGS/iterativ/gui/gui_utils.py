import tkinter as tk
from tkinter import ttk


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.vsb.pack(side="right", fill="y")

        self.inner = None
        self.inner_id = None

        self._create_inner()

        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _create_inner(self):
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", self._on_inner_configure)
        self.inner_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

    def _on_inner_configure(self, _event=None):
        bbox = self.canvas.bbox("all")
        if bbox is None:
            self.canvas.configure(scrollregion=(0, 0, 0, 0))
        else:
            self.canvas.configure(scrollregion=bbox)

    def _on_canvas_configure(self, event):
        if self.inner_id is not None:
            self.canvas.itemconfigure(self.inner_id, width=event.width)

    def _on_mousewheel(self, event):
        bbox = self.canvas.bbox("all")
        if not bbox:
            return

        x1, y1, x2, y2 = bbox
        content_height = y2 - y1
        visible_height = self.canvas.winfo_height()

        # Nur scrollen, wenn es wirklich etwas zu scrollen gibt
        if content_height > visible_height:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def clear(self):
        if self.inner_id is not None:
            self.canvas.delete(self.inner_id)

        self._create_inner()
        self.update_idletasks()
        self.canvas.yview_moveto(0.0)
        self._on_inner_configure()