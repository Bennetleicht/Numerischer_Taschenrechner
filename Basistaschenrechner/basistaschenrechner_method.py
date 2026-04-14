from __future__ import annotations

import importlib.util
import pathlib
import tkinter as tk
from tkinter import ttk, messagebox


_solver_path = pathlib.Path(__file__).with_name("basistaschenrechner_solver.py")
_spec = importlib.util.spec_from_file_location("solver", _solver_path)
_solver_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_solver_module)

BasicCalculatorSolver = _solver_module.BasicCalculatorSolver
CalculatorError = _solver_module.CalculatorError


class BasicCalculatorApp(tk.Tk):
    BG = "#f3f5fa"

    NUM_COLOR = "#ffffff"
    OP_COLOR = "#e6ebf5"
    EQ_COLOR = "#4a90e2"

    def __init__(self):
        super().__init__()

        self.title("Basistaschenrechner")
        self.geometry("430x600")
        self.resizable(False, False)

        self.solver = BasicCalculatorSolver()
        self._is_topmost = False

        self.display_var = tk.StringVar(value="0")
        self.expression_var = tk.StringVar(value="")

        self._build_ui()
        self._refresh()

    
    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        self.lock_btn = ttk.Button(top, text="🔓", width=3, command=self._toggle_topmost)
        self.lock_btn.pack(side="left")

        #  Rechenweg (oben, grau)
        expr = tk.Label(
            self,
            textvariable=self.expression_var,
            font=("Segoe UI", 14),
            fg="#7a869a",
            anchor="e"
        )
        expr.pack(fill="x", padx=10)

        #  Hauptanzeige 
        display = tk.Label(
            self,
            textvariable=self.display_var,
            font=("Segoe UI", 28),
            anchor="e",
            bg="white",
            relief="solid",
            bd=1
        )
        display.pack(fill="x", padx=10, pady=(0, 10))

        grid = tk.Frame(self)
        grid.pack(expand=True, fill="both", padx=10, pady=10)

        buttons = [
            ["C", "⌫", "xʸ", "√x"],
            ["7", "8", "9", "÷"],
            ["4", "5", "6", "×"],
            ["1", "2", "3", "-"],
            ["0", ".", "=", "+"],
        ]

        for r, row in enumerate(buttons):
            for c, txt in enumerate(row):
                btn = tk.Button(
                    grid,
                    text=txt,
                    command=lambda t=txt: self._on_button(t),
                    font=("Segoe UI", 12),
                    bg=self._get_color(txt),
                    activebackground="#d0d8e8",
                    bd=0
                )

                btn.grid(row=r, column=c, sticky="nsew", padx=1, pady=1)

        for i in range(len(buttons)):
            grid.rowconfigure(i, weight=1)
        for i in range(4):
            grid.columnconfigure(i, weight=1)

    #  Farben 
    def _get_color(self, txt):
        if txt == "=":
            return self.EQ_COLOR
        elif txt in ["+", "-", "*", "/", "÷", "×", "xʸ", "√x"]:
            return self.OP_COLOR
        else:
            return self.NUM_COLOR

    #  Logik
    def _on_button(self, t):
        try:
            if t.isdigit():
                self.solver.input_digit(t)
            elif t == ".":
                self.solver.input_decimal()
            elif t in ["+", "-", "*", "/"]:
                self.solver.set_operator(t)
            elif t == "÷":
                self.solver.set_operator("/")
            elif t == "×":
                self.solver.set_operator("*")
            elif t == "xʸ":
                self.solver.set_operator("^")
            elif t == "=":
                self.solver.calculate_result()
            elif t == "C":
                self.solver.clear_all()
            elif t == "⌫":
                self.solver.backspace()
            elif t == "√x":
                self.solver.sqrt()

        except CalculatorError as e:
            messagebox.showerror("Fehler", str(e))

        self._refresh()

    def _refresh(self):
        self.display_var.set(self.solver.get_display())
        self.expression_var.set(self.solver.get_expression())

    #  Always on Top
    def _toggle_topmost(self):
        self._is_topmost = not self._is_topmost
        self.attributes("-topmost", self._is_topmost)
        self.lock_btn.config(text="🔒" if self._is_topmost else "🔓")


if __name__ == "__main__":
    app = BasicCalculatorApp()
    app.mainloop()