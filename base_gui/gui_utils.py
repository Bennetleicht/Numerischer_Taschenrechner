import tkinter as tk
from tkinter import ttk
import os
import sys  
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from GUI import maximize_and_lock

class ScrollableFrame(ttk.Frame):
    """Scrollbarer Frame – identisch in allen drei Modulen."""

    def __init__(self, parent):
        super().__init__(parent)

        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical",
                                  command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.vsb.pack(side="right", fill="y")

        self.inner = None
        self.inner_id = None
        self._create_inner()

        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

    def _create_inner(self):
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", self._on_inner_configure)
        self.inner_id = self.canvas.create_window(
            (0, 0), window=self.inner, anchor="nw")

    def _on_inner_configure(self, _event=None):
        bbox = self.canvas.bbox("all")
        self.canvas.configure(scrollregion=bbox if bbox else (0, 0, 0, 0))

    def _on_canvas_configure(self, event):
        if self.inner_id is not None:
            self.canvas.itemconfigure(self.inner_id, width=event.width)

    def _bind_mousewheel(self, _event=None):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event=None):
        self.canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        widget = self.winfo_containing(event.x_root, event.y_root)
        if widget is None:
            return
        parent = widget
        inside = False
        while parent is not None:
            if parent == self.canvas:
                inside = True
                break
            parent = getattr(parent, 'master', None)
        if not inside:
            return

        bbox = self.canvas.bbox("all")
        if not bbox:
            return
        _, y1, _, y2 = bbox
        if (y2 - y1) > self.canvas.winfo_height():
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def clear(self):
        if self.inner_id is not None:
            self.canvas.delete(self.inner_id)
        self._create_inner()
        self.update_idletasks()
        self.canvas.yview_moveto(0.0)
        self._on_inner_configure()


class PivotTile(tk.Frame):
    """Klickbare Kachel für Pivot-Modus-Auswahl (Gauss-GUI)."""

    def __init__(self, parent, text: str, command):
        super().__init__(parent, relief="solid", borderwidth=1,
                         cursor="hand2")
        self.command = command
        self.label = tk.Label(self, text=text, font=("Segoe UI", 10),
                              padx=10, pady=6)
        self.label.pack()
        self.bind("<Button-1>", self._on_click)
        self.label.bind("<Button-1>", self._on_click)

    def _on_click(self, _ev):
        self.command()

    def set_selected(self, selected: bool,
                     bg_normal: str, bg_selected: str,
                     border_selected: str, border_normal: str):
        bg     = bg_selected     if selected else bg_normal
        border = border_selected if selected else border_normal
        self.configure(bg=bg, highlightbackground=border,
                       highlightthickness=1)
        self.label.configure(bg=bg)


def maximize_window(fenster: tk.Tk):
    """Fenster maximieren"""
    
    maximize_and_lock(fenster)




# Alias für Rückwärtskompatibilität mit altem Code
_maximize_window = maximize_window