import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Tuple, Optional

from gui.gui_utils import ScrollableFrame
from gauss_seidel_methode import GaussSeidelStepper
from gauss_seidel_solver import GaussSeidelRowDetail, Step as GaussStep
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class GaussSeidelGUI(tk.Tk):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.title("Gauß-Seidel-Verfahren")
        self.minsize(480, 320)
        self.state("zoomed")         
        try:
            self.attributes("-zoomed", True) 
        except Exception:
            pass

        self.n_var = tk.StringVar(value="4")
        self.tol_var = tk.StringVar(value="1e-6")

        self.entries_A: List[List[tk.Entry]] = []
        self.entries_b: List[tk.Entry] = []
        self.entries_x0: List[tk.Entry] = []

        self.stepper: Optional[GaussSeidelStepper] = None
        self.started = False
        self.step_count = 0
        self.formula_canvas = None

        self._init_colors()
        self._init_style()
        self.configure(bg=self.app_bg)
        self._build_ui()
        self._build_input_matrix()

    def _init_colors(self):
        self.bg_default = "#ffffff"
        self.bg_diag = "#f0d484"
        self.bg_offdiag = "#e9bcc9"
        self.bg_rhs = "#b8e0b8"
        self.bg_oldx = "#f4d2a7"
        self.bg_newx = "#b8d5f0"

        self.card_bg = "#ffffff"
        self.header_bg = "#eef1f5"
        self.app_bg = "#f6f7fb"
        self.formula_bg = "#f3f3f5"
        self.calc_bg = "#fbfbfd"
        self.border = "#cfd4dc"
        self.text_dark = "#1f2937"

    def _init_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("App.TFrame", background=self.app_bg)
        style.configure("Card.TFrame", background=self.card_bg, relief="solid", borderwidth=1)
        style.configure("Header.TLabel", background=self.header_bg, font=("Segoe UI", 12, "bold"))
        style.configure("Sub.TLabel", background=self.card_bg, font=("Segoe UI", 10))
        style.configure("Small.TLabel", background=self.card_bg, font=("Segoe UI", 9))
        style.configure("Title.TLabel", background=self.app_bg, font=("Segoe UI", 14, "bold"))
        style.configure("Hint.TLabel", background=self.app_bg, foreground="#4b5563", font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 10))
        style.configure("TSpinbox", font=("Segoe UI", 10))

    def _fmt(self, x: float) -> str:
        if abs(x) < 1e-12:
            x = 0.0
        return "{:.6g}".format(x)

    def _sup_digit(self, s) -> str:
        sup_map = str.maketrans("0123456789-()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁽⁾")
        return str(s).translate(sup_map)

    def _sub_digit(self, s) -> str:
        sub_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        return str(s).translate(sub_map)

    def _x_symbol(self, idx: Optional[int] = None, iteration: Optional[int] = None) -> str:
        base = "x" if idx is None else f"x{self._sub_digit(idx)}"
        if iteration is None:
            return base
        return f"{base}{self._sup_digit(f'({iteration})')}"

    def _build_ui(self):
        main = ttk.Frame(self, padding=10, style="App.TFrame")
        main.pack(fill="both", expand=True)

        head = ttk.Frame(main, style="App.TFrame")
        head.pack(fill="x", pady=(0, 8))
        ttk.Label(head, text="Gauß-Seidel-Verfahren", style="Title.TLabel").pack(side="left")

        controls = ttk.Frame(main, style="App.TFrame")
        controls.pack(fill="x", pady=(0, 8))

        ttk.Label(controls, text="Groesse n:", style="Hint.TLabel").pack(side="left")
        self.n_spin = ttk.Spinbox(controls, from_=2, to=7, textvariable=self.n_var, width=5)
        self.n_spin.pack(side="left", padx=(8, 16))

        ttk.Label(controls, text="Toleranz:", style="Hint.TLabel").pack(side="left")
        self.tol_entry = ttk.Entry(controls, textvariable=self.tol_var, width=10)
        self.tol_entry.pack(side="left", padx=(8, 16))

        self.btn_start = ttk.Button(controls, text="Start", command=self.on_start_or_next)
        self.btn_reset = ttk.Button(controls, text="Reset", command=self.on_reset)
        self.btn_start.pack(side="left", padx=6)
        self.btn_reset.pack(side="left", padx=6)

        vcmd = (self.register(self._validate_n_input), "%P")
        self.n_spin.configure(validate="key", validatecommand=vcmd)
        self.n_var.trace_add("write", self._on_n_changed)
        self.n_spin.bind("<Return>", lambda _e: self._rebuild_input())
        self.n_spin.bind("<FocusOut>", lambda _e: self._rebuild_input())

        self.input_matrix_frame = ttk.Frame(main, style="App.TFrame")
        self.input_matrix_frame.pack(fill="x")

        ttk.Separator(main).pack(fill="x", pady=8)

        self.history = ScrollableFrame(main)
        self.history.pack(fill="both", expand=True)

    def _set_input_locked(self, locked: bool):
        state = "disabled" if locked else "normal"

        self.n_spin.configure(state=state)
        self.tol_entry.configure(state=state)

        for row in self.entries_A:
            for e in row:
                e.configure(state=state)

        for e in self.entries_b:
            e.configure(state=state)

        for e in self.entries_x0:
            e.configure(state=state)


    def _validate_n_input(self, proposed: str) -> bool:
        if proposed == "":
            return True
        if not proposed.isdigit():
            return False
        val = int(proposed)
        return 2 <= val <= 10

    def _on_n_changed(self, *_args):
        try:
            val = int(self.n_var.get())
        except Exception:
            return
        if 2 <= val <= 10:
            self._rebuild_input()

    def _get_n(self) -> int:
        try:
            return int(self.n_var.get())
        except Exception:
            return 4

    def _rebuild_input(self):
        self.stepper = None
        self.started = False
        self.btn_start.configure(text="Start")
        self._build_input_matrix()

    def _render_formula(self, parent):
        formula = (
            r"$x_i^{(k+1)}"
            r" = \left( b_i"
            r" - \sum_{j=1}^{i-1} a_{ij}x_j^{(k+1)}"
            r" - \sum_{j=i+1}^{n} a_{ij}x_j^{(k)} \right) / a_{ii}$"
        )

        fig = Figure(figsize=(10, 1.8), dpi=120)
        ax = fig.add_subplot(111)

        fig.patch.set_facecolor(self.formula_bg)
        ax.set_facecolor(self.formula_bg)
        ax.axis("off")

        ax.text(
            0.02,
            0.5,
            formula,
            fontsize=20,
            ha="left",
            va="center"
        )

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()

        widget = canvas.get_tk_widget()
        widget.configure(width=1000, height=140)
        widget.pack(padx=8, pady=8)

        self.formula_canvas = canvas

    def _build_input_matrix(self):
        for w in self.input_matrix_frame.winfo_children():
            w.destroy()

        self.entries_A.clear()
        self.entries_b.clear()
        self.entries_x0.clear()

        n = self._get_n()

        box = ttk.Frame(self.input_matrix_frame, padding=8, style="Card.TFrame")
        box.pack(fill="x")

        top = ttk.Frame(box, style="Card.TFrame")
        top.pack(fill="x", pady=(0, 6))
        ttk.Label(top, text="Eingabe: Matrix A | Vektor b | Startvektor x⁽⁰⁾", style="Sub.TLabel").pack(side="left")

        content = ttk.Frame(box, style="Card.TFrame")
        content.pack(fill="x")
        content.columnconfigure(0, weight=0)
        content.columnconfigure(1, weight=1)

        left = ttk.Frame(content, style="Card.TFrame")
        left.grid(row=0, column=0, sticky="nw", padx=(0, 24))

        grid = ttk.Frame(left, style="Card.TFrame")
        grid.pack(anchor="w")

        for r in range(n):
            row_entries = []
            for c in range(n):
                e = tk.Entry(
                    grid,
                    width=8,
                    justify="center",
                    bg=self.bg_default,
                    relief="solid",
                    borderwidth=1,
                    font=("Consolas", 12)
                )
                e.grid(row=r, column=c, padx=3, pady=3)
                e.insert(0, "0")
                row_entries.append(e)
            self.entries_A.append(row_entries)

            ttk.Label(grid, text="|", style="Sub.TLabel").grid(row=r, column=n, padx=8)

            eb = tk.Entry(
                grid,
                width=8,
                justify="center",
                bg=self.bg_default,
                relief="solid",
                borderwidth=1,
                font=("Consolas", 12)
            )
            eb.grid(row=r, column=n + 1, padx=3, pady=3)
            eb.insert(0, "0")
            self.entries_b.append(eb)

            ttk.Label(grid, text="|", style="Sub.TLabel").grid(row=r, column=n + 2, padx=8)

            ex = tk.Entry(
                grid,
                width=8,
                justify="center",
                bg=self.bg_default,
                relief="solid",
                borderwidth=1,
                font=("Consolas", 12)
            )
            ex.grid(row=r, column=n + 3, padx=3, pady=3)
            ex.insert(0, "0")
            self.entries_x0.append(ex)

        right = ttk.Frame(content, padding=(10, 0), style="Card.TFrame")
        right.grid(row=0, column=1, sticky="nsew")

        ttk.Label(right, text="Gauß-Seidel-Formel", style="Sub.TLabel").pack(anchor="w", pady=(0, 8))

        formula_holder = tk.Frame(
            right,
            bg=self.formula_bg,
            relief="solid",
            borderwidth=1,
            width=1020,
            height=150
        )
        formula_holder.pack_propagate(False)
        formula_holder.pack(anchor="w")

        self._render_formula(formula_holder)

    def _read_inputs(self) -> Tuple[List[List[float]], List[float], List[float], float]:
        n = self._get_n()
        A: List[List[float]] = []
        b: List[float] = []
        x0: List[float] = []

        try:
            for r in range(n):
                row = []
                for c in range(n):
                    row.append(float(self.entries_A[r][c].get().strip()))
                A.append(row)
                b.append(float(self.entries_b[r].get().strip()))
                x0.append(float(self.entries_x0[r].get().strip()))
            tol = float(self.tol_var.get().strip())
        except ValueError:
            raise ValueError("Ungueltige Eingabe: Bitte nur Zahlen eintragen.")

        if tol <= 0:
            raise ValueError("Toleranz muss > 0 sein.")

        return A, b, x0, tol

    def _append_colored_math_row(self, parent, detail: GaussSeidelRowDetail, k_old: int, k_new: int, row: int):
        ROW_H = 28
        PADY_BOX = 3   # vertikaler Innenabstand der farbigen Kästchen
        font_main = ("Cambria Math", 13)
        font_bold  = ("Cambria Math", 13, "bold")

        # Canvas mit fester Höhe – Breite wird später angepasst
        cv = tk.Canvas(parent, bg=self.calc_bg, height=ROW_H,
                    highlightthickness=0, bd=0)
        cv.grid(row=row, column=0, sticky="w", pady=1)

        x = 4   # aktueller X-Cursor

        def measure(text, bold=False):
            f = font_bold if bold else font_main
            return cv.tk.call("font", "measure", f, text)

        def draw_txt(text, bold=False):
            nonlocal x
            f = font_bold if bold else font_main
            cv.create_text(x, ROW_H // 2, text=text, anchor="w",
                        font=f, fill=self.text_dark)
            x += measure(text, bold) + 2

        def draw_box(text, bg, bold=False):
            nonlocal x
            f = font_bold if bold else font_main
            tw = measure(text, bold)
            pad = 4
            box_w = tw + pad * 2
            box_h = ROW_H - PADY_BOX * 2
            y0 = PADY_BOX
            y1 = ROW_H - PADY_BOX
            cv.create_rectangle(x, y0, x + box_w, y1,
                                fill=bg, outline="#888888", width=1)
            cv.create_text(x + pad, ROW_H // 2, text=text, anchor="w",
                        font=f, fill=self.text_dark)
            x += box_w + 3

        i = detail.row_index + 1
        draw_txt(f"{self._x_symbol(i, k_new)} = (")
        draw_box(self._fmt(detail.rhs), self.bg_rhs)
        draw_txt("  - (")

        for idx, (j, aij, _xj, _prod, uses_new) in enumerate(detail.terms):
            if idx > 0:
                draw_txt("  +  ")
            draw_box(self._fmt(aij), self.bg_offdiag)
            draw_txt(" · ")
            sym = self._x_symbol(j + 1, k_new if uses_new else k_old)
            draw_txt(sym)

        draw_txt("))  /  ")
        draw_box(self._fmt(detail.diag), self.bg_diag)
        draw_txt("  =  ")
        draw_box(self._fmt(detail.new_value), self.bg_newx, bold=True)

        # Canvas-Breite auf tatsächlichen Inhalt setzen
        cv.configure(width=x + 4)

    def _append_step_card(self, A: List[List[float]], b: List[float], step: GaussStep):
        self.step_count += 1
        n = len(A)

        card = ttk.Frame(self.history.inner, padding=12, style="Card.TFrame")
        card.pack(fill="x", pady=8, padx=2)

        title = f"Iteration {step.iteration}" if step.kind == "iter" else f"Abschluss nach Iteration {step.iteration}"
        header = ttk.Label(card, text=title, style="Header.TLabel", padding=(10, 8))
        header.pack(fill="x")

        body = ttk.Frame(card, style="Card.TFrame")
        body.pack(fill="x", pady=(12, 0))
        body.columnconfigure(0, weight=0)
        body.columnconfigure(1, weight=1)

        left = ttk.Frame(body, style="Card.TFrame")
        right = ttk.Frame(body, style="Card.TFrame")
        left.grid(row=0, column=0, sticky="nw", padx=(0, 24))
        right.grid(row=0, column=1, sticky="nsew")

        grid = ttk.Frame(left, style="Card.TFrame")
        grid.pack(anchor="w")

        for c in range(n):
            tk.Label(
                grid,
                text="",
                bg=self.card_bg,
                width=10,
                font=("Consolas", 12, "bold")
            ).grid(row=0, column=c, padx=3, pady=(0, 4))

        tk.Label(
            grid,
            text="",
            bg=self.card_bg,
            font=("Consolas", 12, "bold")
        ).grid(row=0, column=n, padx=(8, 8), pady=(0, 4))

        tk.Label(
            grid,
            text="Matrix A",
            bg=self.card_bg,
            width=10,
            font=("Segoe UI", 12, "bold")
        ).grid(row=0, column=n - 2, padx=3, pady=(0, 4))

        tk.Label(
            grid,
            text="b",
            bg=self.card_bg,
            width=10,
            font=("Segoe UI", 12, "bold")
        ).grid(row=0, column=n + 1, padx=3, pady=(0, 4))

        for r in range(n):
            gr = r + 1
            for c in range(n):
                bg = self.bg_diag if r == c else self.bg_offdiag
                tk.Label(
                    grid,
                    text=self._fmt(A[r][c]),
                    width=10,
                    bg=bg,
                    relief="solid",
                    borderwidth=1,
                    font=("Consolas", 12)
                ).grid(row=gr, column=c, padx=3, pady=3)

            tk.Label(
                grid,
                text="|",
                bg=self.card_bg,
                font=("Consolas", 12)
            ).grid(row=gr, column=n, padx=(8, 8), pady=3)

            tk.Label(
                grid,
                text=self._fmt(b[r]),
                width=10,
                bg=self.bg_rhs,
                relief="solid",
                borderwidth=1,
                font=("Consolas", 12)
            ).grid(row=gr, column=n + 1, padx=3, pady=3)

        legend = ttk.Frame(left, style="Card.TFrame")
        legend.pack(anchor="w", pady=(10, 0))

        self._legend_item(legend, "aᵢᵢ", self.bg_diag, 0)
        self._legend_item(legend, "aᵢⱼ", self.bg_offdiag, 1)
        self._legend_item(legend, "bᵢ", self.bg_rhs, 2)
        self._legend_item(legend, "x⁽k⁾", self.bg_oldx, 3)
        self._legend_item(legend, "x⁽k+1⁾", self.bg_newx, 4)

        calc_outer = tk.Frame(
            right,
            bg=self.calc_bg,
            relief="solid",
            borderwidth=1,
            highlightthickness=1,
            highlightbackground=self.border
        )
        calc_outer.pack(fill="x", expand=True)

        k_old = step.iteration - 1
        k_new = step.iteration

        calc_grid = tk.Frame(calc_outer, bg=self.calc_bg)
        calc_grid.pack(fill="x", padx=18, pady=12)

        # Zeilenhöhen für die Rechenzeilen begrenzen
        for i in range(20):
            calc_grid.rowconfigure(i, minsize=0)

        tk.Label(
            calc_grid,
            text=f"Rechnung {step.iteration}",
            bg=self.calc_bg,
            fg=self.text_dark,
            font=("Segoe UI", 12, "bold")
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        tk.Label(
            calc_grid,
            text=f"{self._x_symbol(None, k_old)} = [" + ", ".join(self._fmt(v) for v in step.x_old) + "]",
            bg=self.bg_oldx,
            fg=self.text_dark,
            font=("Segoe UI", 12),
            anchor="w",
            padx=10,
            pady=6
        ).grid(row=1, column=0, sticky="ew", pady=(0, 10))

        start_row = 2
        for idx, detail in enumerate(step.row_details):
            self._append_colored_math_row(calc_grid, detail, k_old, k_new, start_row + idx)

        tk.Label(
            calc_grid,
            text=f"{self._x_symbol(None, k_new)} = [" + ", ".join(self._fmt(v) for v in step.x_new) + "]",
            bg=self.bg_newx,
            fg=self.text_dark,
            font=("Segoe UI", 12, "bold"),
            anchor="w",
            padx=10,
            pady=6
        ).grid(row=start_row + len(step.row_details), column=0, sticky="ew", pady=(10, 0))

        self.history.canvas.update_idletasks()
        self.history.canvas.yview_moveto(1.0)

    def _legend_item(self, parent, text, color, col):
        frame = ttk.Frame(parent, style="Card.TFrame")
        frame.grid(row=0, column=col, padx=(0, 10), sticky="w")

        swatch = tk.Label(frame, bg=color, width=2, relief="solid", borderwidth=1)
        swatch.pack(side="left", padx=(0, 4))
        ttk.Label(frame, text=text, style="Small.TLabel").pack(side="left")

    def _append_final_solution_card(self):
        if not self.stepper:
            return

        x = self.stepper.x[:]
        k = self.stepper.iteration

        card = ttk.Frame(self.history.inner, padding=10, style="Card.TFrame")
        card.pack(fill="x", pady=8, padx=2)

        header = ttk.Label(card, text="LOESUNG / NAEHERUNG", style="Header.TLabel", padding=(8, 6))
        header.pack(fill="x")

        body = ttk.Frame(card, style="Card.TFrame")
        body.pack(fill="x", pady=(10, 0))

        parts = [f"{self._x_symbol(i + 1)} = {self._fmt(v)}" for i, v in enumerate(x)]
        sol = "   |   ".join(parts)

        txt = tk.Text(
            body,
            height=3,
            wrap="word",
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 10),
            spacing1=0,
            spacing3=0
        )
        txt.pack(fill="x")
        txt.tag_configure("newx", background=self.bg_newx, font=("Segoe UI", 10, "bold"))
        txt.insert("1.0", f"Endwert {self._x_symbol(None, k)}:\n")
        txt.insert("end", sol, ("newx",))
        txt.configure(state="disabled")

        messagebox.showinfo("Gauß-Seidel-Ergebnis", sol)

    def on_start_or_next(self):
        if self.started:
            self._do_one_step()
            return

        if self.step_count > 0:
            self.history.clear()
            self.step_count = 0
        self.stepper = None

        try:
            A, b, x0, tol = self._read_inputs()
            self.stepper = GaussSeidelStepper(A, b, x0, tol=tol, safety_limit=100)
        except ValueError as e:
            messagebox.showerror("Eingabefehler", str(e))
            return

        self.started = True
        self.btn_start.configure(text="Weiter")
        self._set_input_locked(True)
        self.after(0, self._do_one_step)

    def on_reset(self):
        self.stepper = None
        self.started = False
        self.btn_start.configure(text="Start", state="normal")
        self._set_input_locked(False)
        self.history.clear()
        self.step_count = 0

        for row in self.entries_A:
            for e in row:
                e.delete(0, tk.END)
                e.insert(0, "0")
                e.configure(bg=self.bg_default)

        for e in self.entries_b:
            e.delete(0, tk.END)
            e.insert(0, "0")

        for e in self.entries_x0:
            e.delete(0, tk.END)
            e.insert(0, "0")

        self.tol_var.set("1e-6")

    def _do_one_step(self):
        if not self.stepper:
            return

        step = self.stepper.next_step()
        self._append_step_card(self.stepper.A, self.stepper.b, step)

        if step.kind == "done":
            self._append_final_solution_card()
            self.btn_start.configure(state="disabled")
