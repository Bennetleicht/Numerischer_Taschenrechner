import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Optional, Dict, Tuple, Any

from cholesky_solver import CholeskySolver, Step 
from gui.gui_utils import ScrollableFrame, _maximize_window

class CholeskyStepper(CholeskySolver):
    pass

class CholeskyGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cholesky-Verfahren")
        self.geometry("1280x720")
        _maximize_window(self)

        self.n_var = tk.StringVar(value="4")
        self.entries_A: List[List[tk.Entry]] = []
        self.entries_b: List[tk.Entry] = []

        self.stepper: Optional[CholeskyStepper] = None
        self.step_count = 0
        self.started = False

        self._init_colors()
        self._init_style()
        self._build_ui()
        self._build_input_matrix()

    def _fmt(self, x: float) -> str:
        if abs(x) < 1e-12:
            x = 0.0
        return "{:.6g}".format(x)

    def _init_colors(self):
        self.bg_default = "#ffffff"
        self.bg_pivot = "#ffe08a"
        self.bg_changed = "#ffb3c6"
        self.card_bg = "#ffffff"
        self.header_bg = "#f3f4f6"

    def _init_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("App.TFrame", background="#f6f7fb")
        style.configure("Card.TFrame", background=self.card_bg, relief="solid", borderwidth=1)
        style.configure("Header.TLabel", background=self.header_bg, font=("Segoe UI", 11, "bold"))
        style.configure("Sub.TLabel", background=self.card_bg, font=("Segoe UI", 10))
        style.configure("Small.TLabel", background=self.card_bg, font=("Segoe UI", 9))
        style.configure("Title.TLabel", background="#f6f7fb", font=("Segoe UI", 13, "bold"))
        style.configure("Hint.TLabel", background="#f6f7fb", foreground="#4b5563", font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 10))
        style.configure("TSpinbox", font=("Segoe UI", 10))

    def _build_ui(self):
        main = ttk.Frame(self, padding=10, style="App.TFrame")
        main.pack(fill="both", expand=True)

        head = ttk.Frame(main, style="App.TFrame")
        head.pack(fill="x", pady=(0, 8))
        ttk.Label(head, text="Cholesky-Zerlegung", style="Title.TLabel").pack(side="left")

        controls = ttk.Frame(main, style="App.TFrame")
        controls.pack(fill="x", pady=(0, 8))

        ttk.Label(controls, text="Groesse n:", style="Hint.TLabel").pack(side="left")
        self.n_spin = ttk.Spinbox(controls, from_=2, to=10, textvariable=self.n_var, width=5)
        self.n_spin.pack(side="left", padx=8)

        vcmd = (self.register(self._validate_n_input), "%P")
        self.n_spin.configure(validate="key", validatecommand=vcmd)
        self.n_var.trace_add("write", self._on_n_changed)
        self.n_spin.bind("<Return>", lambda _e: self._rebuild_input())
        self.n_spin.bind("<FocusOut>", lambda _e: self._rebuild_input())

        self.btn_start = ttk.Button(controls, text="Start", command=self.on_start_or_next)
        self.btn_reset = ttk.Button(controls, text="Reset", command=self.on_reset)
        self.btn_start.pack(side="left", padx=6)
        self.btn_reset.pack(side="left", padx=6)

        self.input_matrix_frame = ttk.Frame(main, style="App.TFrame")
        self.input_matrix_frame.pack(fill="x")

        ttk.Separator(main).pack(fill="x", pady=8)

        self.history = ScrollableFrame(main)
        self.history.pack(fill="both", expand=True)

    def _set_input_locked(self, locked: bool):
        state = "disabled" if locked else "normal"

        self.n_spin.configure(state=state)

        for row in self.entries_A:
            for e in row:
                e.configure(state=state)

        for e in self.entries_b:
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

    def _build_input_matrix(self):
        for w in self.input_matrix_frame.winfo_children():
            w.destroy()

        self.entries_A.clear()
        self.entries_b.clear()

        n = self._get_n()

        box = ttk.Frame(self.input_matrix_frame, padding=8, style="Card.TFrame")
        box.pack(fill="x")

        top = ttk.Frame(box, style="Card.TFrame")
        top.pack(fill="x", pady=(0, 4))
        ttk.Label(top, text="Eingabe: Matrix A | Vektor b", style="Small.TLabel").pack(side="left")

        grid = ttk.Frame(box, style="Card.TFrame")
        grid.pack(anchor="w")

        for r in range(n):
            row_entries = []
            for c in range(n):
                e = tk.Entry(
                    grid,
                    width=10,
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

            ttk.Label(grid, text="|", style="Small.TLabel").grid(row=r, column=n, padx=10)

            eb = tk.Entry(
                grid,
                width=10,
                justify="center",
                bg=self.bg_default,
                relief="solid",
                borderwidth=1,
                font=("Consolas", 12)
            )
            eb.grid(row=r, column=n + 1, padx=3, pady=3)
            eb.insert(0, "0")
            self.entries_b.append(eb)

    def _read_inputs(self) -> Tuple[List[List[float]], List[float]]:
        n = self._get_n()
        A: List[List[float]] = []
        b: List[float] = []
        try:
            for r in range(n):
                row = []
                for c in range(n):
                    row.append(float(self.entries_A[r][c].get().strip()))
                A.append(row)
                b.append(float(self.entries_b[r].get().strip()))
        except ValueError:
            raise ValueError("Ungueltige Eingabe: Bitte nur Zahlen in A und b eintragen.")
        return A, b

    def _is_symmetric(self, A: List[List[float]], tol: float = 1e-9) -> bool:
        n = len(A)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i][j] - A[j][i]) > tol:
                    return False
        return True

    def _append_step_card(self, snap: Dict[str, Any], step: Step, extra_title: Optional[str] = None):
        self.step_count += 1

        kind_map = {
            "chol_update": "ZERLEGUNG",
            "forward_sub": "VORWÄRTS",
            "back_sub": "RÜCKWÄRTS",
            "done": "FERTIG"
        }
        kind_txt = kind_map.get(step.kind, step.kind.upper())
        title = extra_title if extra_title else f"Schritt {self.step_count}: {kind_txt}"

        changed = set(step.changed or [])
        pivot_r, pivot_c = step.pivot

        card = ttk.Frame(self.history.inner, padding=10, style="Card.TFrame")
        card.pack(fill="x", pady=8, padx=2)

        header = ttk.Label(card, text=title, style="Header.TLabel", padding=(8, 6))
        header.pack(fill="x")

        body = ttk.Frame(card, style="Card.TFrame")
        body.pack(fill="x", pady=(10, 0))
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

        left = ttk.Frame(body, style="Card.TFrame")
        right = ttk.Frame(body, style="Card.TFrame")
        left.grid(row=0, column=0, sticky="nw", padx=(0, 12))
        right.grid(row=0, column=1, sticky="new")

        top_row = ttk.Frame(left, style="Card.TFrame")
        top_row.pack(anchor="w")

        frame_A = ttk.Frame(top_row, style="Card.TFrame")
        frame_L = ttk.Frame(top_row, style="Card.TFrame")
        frame_Lt = ttk.Frame(top_row, style="Card.TFrame")
        frame_A.pack(side="left", anchor="n", padx=(0, 18))
        frame_L.pack(side="left", anchor="n", padx=(0, 18))
        frame_Lt.pack(side="left", anchor="n")

        ttk.Label(frame_A, text="A", style="Sub.TLabel").pack(anchor="w")
        ttk.Label(frame_L, text="L", style="Sub.TLabel").pack(anchor="w")
        ttk.Label(frame_Lt, text="L^T", style="Sub.TLabel").pack(anchor="w")

        self._draw_matrix(frame_A, "A", snap["A"], changed, pivot_r, pivot_c, step)
        self._draw_matrix(frame_L, "L", snap["L"], changed, pivot_r, pivot_c, step)
        self._draw_matrix(frame_Lt, "Lt", snap["Lt"], changed, pivot_r, pivot_c, step)

        vec_row = ttk.Frame(left, style="Card.TFrame")
        vec_row.pack(anchor="w", pady=(10, 0))

        frame_b = ttk.Frame(vec_row, style="Card.TFrame")
        frame_y = ttk.Frame(vec_row, style="Card.TFrame")
        frame_x = ttk.Frame(vec_row, style="Card.TFrame")
        frame_b.pack(side="left", anchor="n", padx=(0, 18))
        frame_y.pack(side="left", anchor="n", padx=(0, 18))
        frame_x.pack(side="left", anchor="n")

        ttk.Label(frame_b, text="b", style="Sub.TLabel").pack(anchor="w")
        ttk.Label(frame_y, text="y", style="Sub.TLabel").pack(anchor="w")
        ttk.Label(frame_x, text="x", style="Sub.TLabel").pack(anchor="w")

        self._draw_vector(frame_b, "b", snap["b"], changed)
        self._draw_vector(frame_y, "y", snap["y"], changed)
        self._draw_vector(frame_x, "x", snap["x"], changed)

        #ttk.Label(right, text="Rechnung", style="Sub.TLabel").pack(anchor="w")
        msg = (step.message or "").strip()
        txt = tk.Text(
            right,
            height=max(6, min(12, msg.count("\n") + 4)),
            wrap="word",
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 10)
        )
        txt.pack(fill="x", expand=False)
        txt.insert("1.0", msg)
        txt.configure(state="disabled")

        self.history.canvas.update_idletasks()
        self.history.canvas.yview_moveto(1.0)

    def _draw_matrix(
        self,
        parent: ttk.Frame,
        name: str,
        M: List[List[float]],
        changed: set,
        pivot_r: int,
        pivot_c: int,
        step: Step
    ):
        grid = ttk.Frame(parent, style="Card.TFrame")
        grid.pack(anchor="w", pady=(4, 0))

        rows = len(M)
        cols = len(M[0]) if rows else 0

        for r in range(rows):
            for c in range(cols):
                val = self._fmt(M[r][c])
                bg = self.bg_default

                if (name, r, c) in changed:
                    bg = self.bg_changed

                if step.kind == "chol_update" and name == "L" and r == pivot_r and c == pivot_c:
                    bg = self.bg_pivot

                tk.Label(
                    grid,
                    text=val,
                    width=12,
                    bg=bg,
                    relief="solid",
                    borderwidth=1,
                    font=("Consolas", 12)
                ).grid(row=r, column=c, padx=2, pady=2)

    def _draw_vector(self, parent: ttk.Frame, name: str, v: List[float], changed: set):
        grid = ttk.Frame(parent, style="Card.TFrame")
        grid.pack(anchor="w", pady=(4, 0))
        for r in range(len(v)):
            val = self._fmt(v[r])
            bg = self.bg_changed if (name, r, 0) in changed else self.bg_default
            tk.Label(
                grid,
                text=val,
                width=12,
                bg=bg,
                relief="solid",
                borderwidth=1,
                font=("Consolas", 12)
            ).grid(row=r, column=0, padx=2, pady=2)

    def _append_final_solution_card(self):
        if not self.stepper:
            return
        x_final = self.stepper.x[:]
        parts = [f"x{i+1} = {self._fmt(v)}" for i, v in enumerate(x_final)]
        sol = "   |   ".join(parts)

        dummy = Step(kind="done", pivot=(0, 0), message="LÖSUNG\n" + sol)
        snap = self.stepper.snapshot()
        self._append_step_card(snap, dummy, extra_title="LÖSUNG (Endergebnis)")
        messagebox.showinfo("Lösung", sol)

    def on_start_or_next(self):
        if self.started:
            self._do_one_step()
            return

        if self.step_count > 0:
            for w in self.history.inner.winfo_children():
                w.destroy()
            self.step_count = 0
        self.stepper = None

        try:
            A, b = self._read_inputs()
        except ValueError as e:
            messagebox.showerror("Eingabefehler", str(e))
            return

        if not self._is_symmetric(A):
            messagebox.showerror("Fehler", "Cholesky nur fuer symmetrische Matrizen: A muss symmetrisch sein.")
            return

        self.stepper = CholeskyStepper(A, b, change_tol=1e-10)
        self.started = True
        self.btn_start.configure(text="Weiter")
        self._set_input_locked(True)

        self.after(0, self._do_one_step)

    def on_reset(self):
        self.stepper = None
        self.started = False
        self.btn_start.configure(text="Start")
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

        self.history.canvas.update_idletasks()
        self.history.canvas.yview_moveto(0.0)

    def _do_one_step(self):
        if not self.stepper:
            return

        step = self.stepper.next_step()
        snap = self.stepper.snapshot()
        self._append_step_card(snap, step)

        if step.kind == "done":
            self._append_final_solution_card()
            self.started = False
            self.stepper = None