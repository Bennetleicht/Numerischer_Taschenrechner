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


def _maximize_window(fenster: tk.Tk):
    fenster.minsize(480, 320)
    fenster.state("zoomed")
    try:
        fenster.attributes("-zoomed", True)
    except Exception:
        pass


class PivotTile(tk.Frame):
    def __init__(self, parent, text: str, command):
        super().__init__(parent, bd=1, relief="solid", cursor="hand2")
        self._command = command

        self.label = tk.Label(self, text=text, font=("Segoe UI", 11, "bold"))
        self.label.pack(fill="both", expand=True, padx=10, pady=10)

        self.bind("<Button-1>", self._on_click)
        self.label.bind("<Button-1>", self._on_click)

    def _on_click(self, _ev):
        self._command()

    def set_selected(self, selected: bool, bg_normal: str, bg_selected: str, border_selected: str, border_normal: str):
        if selected:
            self.configure(bg=bg_selected, highlightthickness=2, highlightbackground=border_selected)
            self.label.configure(bg=bg_selected)
        else:
            self.configure(bg=bg_normal, highlightthickness=1, highlightbackground=border_normal)
            self.label.configure(bg=bg_normal)