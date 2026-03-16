import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Optional, Dict, Tuple, Any

from qr_solver import QRGivensSolver, Step
from gui.gui_utils import ScrollableFrame, _maximize_window

class QRGivensStepper(QRGivensSolver):
    pass

class QRGivensGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("QR-Zerlegung (Givens)")
        self.geometry("1280x720")
        _maximize_window(self)

        self.m_var = tk.StringVar(value="4")
        self.n_var = tk.StringVar(value="4")

        self.entries_A0: List[List[tk.Entry]] = []
        self.entries_b0: List[tk.Entry] = []

        self.R_labels: List[List[tk.Label]] = []
        self.y_labels: List[tk.Label] = []

        self.stepper: Optional[QRGivensStepper] = None
        self.step_count = 0
        self.started = False

        self.mode = "init"
        self.selected_target: Optional[Tuple[int, int]] = None

        self.bg_default = "#ffffff"
        self.bg_elim = "#d1fae5"
        self.bg_selected = "#93c5fd"
        self.bg_readonly = "#f9fafb"

        self._init_style()
        self._build_ui()
        self._build_matrices()

        self._update_selected_display()
        self._update_hint("Klicke auf 'Wähle zu eliminierendes Element'. Dann wähle im rechten Kasten das erste Element (grün).")
        self._update_main_button()

    def _init_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("App.TFrame", background="#f6f7fb")
        style.configure("Card.TFrame", background="#ffffff", relief="solid", borderwidth=1)
        style.configure("Title.TLabel", background="#f6f7fb", font=("Segoe UI", 13, "bold"))
        style.configure("Hint.TLabel", background="#f6f7fb", foreground="#374151", font=("Segoe UI", 9))
        style.configure("Small.TLabel", background="#ffffff", font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 10))
        style.configure("TSpinbox", font=("Segoe UI", 10))

    def _fmt(self, x: float) -> str:
        if abs(x) < 1e-12:
            x = 0.0
        return "{:.6g}".format(x)

    def _validate_dim_input(self, proposed: str) -> bool:
        if proposed == "":
            return True
        if not proposed.isdigit():
            return False
        val = int(proposed)
        return 1 <= val <= 12

    def _get_mn(self) -> Tuple[int, int]:
        try:
            m = int(self.m_var.get())
        except Exception:
            m = 4
        try:
            n = int(self.n_var.get())
        except Exception:
            n = 4
        m = max(2, min(12, m))
        n = max(1, min(12, n))
        return m, n

    def _build_ui(self):
        main = ttk.Frame(self, padding=10, style="App.TFrame")
        main.pack(fill="both", expand=True)

        head = ttk.Frame(main, style="App.TFrame")
        head.pack(fill="x", pady=(0, 8))
        ttk.Label(head, text="QR-Zerlegung (Givens)", style="Title.TLabel").pack(side="left")

        controls = ttk.Frame(main, style="App.TFrame")
        controls.pack(fill="x", pady=(0, 8))

        ttk.Label(controls, text="Zeilen m:", style="Hint.TLabel").pack(side="left")
        self.m_spin = ttk.Spinbox(controls, from_=2, to=12, textvariable=self.m_var, width=5)
        self.m_spin.pack(side="left", padx=8)

        ttk.Label(controls, text="Spalten n:", style="Hint.TLabel").pack(side="left", padx=(10, 0))
        self.n_spin = ttk.Spinbox(controls, from_=1, to=12, textvariable=self.n_var, width=5)
        self.n_spin.pack(side="left", padx=8)

        vcmd = (self.register(self._validate_dim_input), "%P")
        self.m_spin.configure(validate="key", validatecommand=vcmd)
        self.n_spin.configure(validate="key", validatecommand=vcmd)

        self.m_var.trace_add("write", lambda *_: self._rebuild_if_not_started())
        self.n_var.trace_add("write", lambda *_: self._rebuild_if_not_started())

        self.btn_main = ttk.Button(controls, text="...", command=self.on_main_button)
        self.btn_reset = ttk.Button(controls, text="Reset", command=self.on_reset)
        self.btn_main.pack(side="left", padx=12)
        self.btn_reset.pack(side="left", padx=6)

        self.selected_var = tk.StringVar(value="Ausgewählt: -")
        ttk.Label(controls, textvariable=self.selected_var, style="Hint.TLabel").pack(side="left", padx=(18, 0))

        self.hint_var = tk.StringVar(value="")
        ttk.Label(controls, textvariable=self.hint_var, style="Hint.TLabel").pack(side="right")

        self.matrices_frame = ttk.Frame(main, style="App.TFrame")
        self.matrices_frame.pack(fill="x")

        ttk.Separator(main).pack(fill="x", pady=8)

        self.history = ScrollableFrame(main)
        self.history.pack(fill="both", expand=True)

    def _build_matrices(self):
        for w in self.matrices_frame.winfo_children():
            w.destroy()

        self.entries_A0.clear()
        self.entries_b0.clear()
        self.R_labels.clear()
        self.y_labels.clear()

        m, n = self._get_mn()

        box = ttk.Frame(self.matrices_frame, padding=8, style="Card.TFrame")
        box.pack(fill="x")

        header = ttk.Frame(box, style="Card.TFrame")
        header.pack(fill="x", pady=(0, 6))
        header.columnconfigure(0, weight=1)
        header.columnconfigure(1, weight=1)

        ttk.Label(header, text=f"Eingabe (Original) A ({m} x {n}) | b ({m})", style="Small.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Aktuell (R und y)", style="Small.TLabel").grid(row=0, column=1, sticky="w")

        body = ttk.Frame(box, style="Card.TFrame")
        body.pack(fill="x")
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

        left = ttk.Frame(body, style="Card.TFrame")
        right = ttk.Frame(body, style="Card.TFrame")

        left.grid(row=0, column=0, sticky="w")
        right.grid(row=0, column=1, sticky="w", padx=(80, 0))

        gridL = ttk.Frame(left, style="Card.TFrame")
        gridL.pack(anchor="w")

        for r in range(m):
            row_entries: List[tk.Entry] = []
            for c in range(n):
                e = tk.Entry(
                    gridL, width=10, justify="center",
                    bg=self.bg_default, relief="solid", borderwidth=1,
                    font=("Consolas", 12)
                )
                e.grid(row=r, column=c, padx=3, pady=3)
                e.insert(0, "0")
                row_entries.append(e)
            self.entries_A0.append(row_entries)

            ttk.Label(gridL, text="|", style="Small.TLabel").grid(row=r, column=n, padx=10)

            eb = tk.Entry(
                gridL, width=10, justify="center",
                bg=self.bg_default, relief="solid", borderwidth=1,
                font=("Consolas", 12)
            )
            eb.grid(row=r, column=n + 1, padx=3, pady=3)
            eb.insert(0, "0")
            self.entries_b0.append(eb)

        gridRwrap = ttk.Frame(right, style="Card.TFrame")
        gridRwrap.pack(anchor="w")

        gridR = ttk.Frame(gridRwrap, style="Card.TFrame")
        gridY = ttk.Frame(gridRwrap, style="Card.TFrame")
        gridR.pack(side="left", anchor="n")
        ttk.Label(gridRwrap, text="|", style="Small.TLabel").pack(side="left", padx=10, anchor="n")
        gridY.pack(side="left", anchor="n")

        for r in range(m):
            row_lbls: List[tk.Label] = []
            for c in range(n):
                lbl = tk.Label(
                    gridR, text="0", width=10,
                    bg=self.bg_default, relief="solid", borderwidth=1,
                    font=("Consolas", 12)
                )
                lbl.grid(row=r, column=c, padx=3, pady=3)
                lbl.bind("<Button-1>", self._make_R_click_handler(r, c))
                row_lbls.append(lbl)
            self.R_labels.append(row_lbls)

            ylbl = tk.Label(
                gridY, text="0", width=10,
                bg=self.bg_default, relief="solid", borderwidth=1,
                font=("Consolas", 12)
            )
            ylbl.grid(row=r, column=0, padx=3, pady=3)
            self.y_labels.append(ylbl)

        self._refresh_current_display(initial=True)

    def _make_R_click_handler(self, r: int, c: int):
        def handler(_event):
            self.on_current_matrix_click(r, c)
            return "break"
        return handler

    def _rebuild_if_not_started(self):
        if self.started:
            return
        self._build_matrices()

    def _read_original_inputs(self) -> Tuple[List[List[float]], List[float]]:
        m, n = self._get_mn()
        A: List[List[float]] = []
        b: List[float] = []
        try:
            for r in range(m):
                row = []
                for c in range(n):
                    row.append(float(self.entries_A0[r][c].get().strip()))
                A.append(row)
                b.append(float(self.entries_b0[r].get().strip()))
        except ValueError:
            raise ValueError("Ungültige Eingabe: Bitte nur Zahlen in A und b eintragen.")
        return A, b

    def _set_original_state(self, running: bool):
        st = "readonly" if running else "normal"
        m, n = self._get_mn()
        for r in range(m):
            for c in range(n):
                e = self.entries_A0[r][c]
                e.configure(state=st, bg=(self.bg_readonly if running else self.bg_default))
        for r in range(m):
            eb = self.entries_b0[r]
            eb.configure(state=st, bg=(self.bg_readonly if running else self.bg_default))

        self.m_spin.configure(state=("disabled" if running else "normal"))
        self.n_spin.configure(state=("disabled" if running else "normal"))

    def _is_eliminable(self, i: int, k: int) -> bool:
        if not self.stepper or self.stepper.phase != "rot":
            return False
        if i <= k:
            return False
        return abs(self.stepper.R[i][k]) > 1e-12

    def _count_eliminables(self) -> int:
        if not self.stepper or self.stepper.phase != "rot":
            return 0
        m, n = self.stepper.m, self.stepper.n
        cnt = 0
        for i in range(1, m):
            for k in range(0, min(i, n)):
                if abs(self.stepper.R[i][k]) > 1e-12:
                    cnt += 1
        return cnt

    def _refresh_current_display(self, initial: bool = False):
        m, n = self._get_mn()

        if not self.stepper:
            for r in range(m):
                for c in range(n):
                    self.R_labels[r][c].configure(text="0", bg=self.bg_default)
                self.y_labels[r].configure(text="0", bg=self.bg_default)
            self.selected_target = None
            self._update_selected_display()
            return

        snap = self.stepper.snapshot()
        R = snap["R"]
        y = snap["y"]

        for r in range(m):
            for c in range(n):
                self.R_labels[r][c].configure(text=self._fmt(R[r][c]))
            self.y_labels[r].configure(text=self._fmt(y[r]))

        self._apply_current_highlights()

        if initial:
            self.selected_target = None
            self._update_selected_display()

    def _apply_current_highlights(self):
        m, n = self._get_mn()

        for r in range(m):
            for c in range(n):
                self.R_labels[r][c].configure(bg=self.bg_default)

        if not self.stepper or self.stepper.phase != "rot":
            return

        for r in range(self.stepper.m):
            for c in range(self.stepper.n):
                if self._is_eliminable(r, c):
                    self.R_labels[r][c].configure(bg=self.bg_elim)

        if self.selected_target is not None:
            i, k = self.selected_target
            if 0 <= i < m and 0 <= k < n:
                self.R_labels[i][k].configure(bg=self.bg_selected)

    def _update_hint(self, text: str):
        self.hint_var.set(text)

    def _update_selected_display(self):
        if not self.stepper or self.selected_target is None:
            self.selected_var.set("Ausgewählt: -")
            return
        i, k = self.selected_target
        val = self.stepper.R[i][k]
        self.selected_var.set(f"Ausgewählt: R[{i+1},{k+1}] = {self._fmt(val)}")

    def _update_main_button(self):
        if not self.started:
            self.mode = "init"
            self.btn_main.configure(text="Wähle zu eliminierendes Element", state="normal")
            return

        if not self.stepper:
            self.btn_main.configure(text="Wähle zu eliminierendes Element", state="disabled")
            return

        if self.stepper.phase == "rot":
            if self.mode == "choose":
                self.btn_main.configure(text="Wähle zu eliminierendes Element", state="normal")
            elif self.mode == "run":
                self.btn_main.configure(text=f"Iteration {self.step_count + 1} starten", state="normal")
            else:
                self.mode = "choose"
                self.btn_main.configure(text="Wähle zu eliminierendes Element", state="normal")
            return

        if self.stepper.phase == "back":
            self.mode = "back"
            self.btn_main.configure(text=f"Rückwärts Schritt {self.step_count + 1}", state="normal")
            return

        self.btn_main.configure(text="Fertig", state="disabled")

    def on_main_button(self):
        if not self.started:
            for w in self.history.inner.winfo_children():
                w.destroy()
            self.step_count = 0

            try:
                A, b = self._read_original_inputs()
            except ValueError as e:
                messagebox.showerror("Eingabefehler", str(e))
                return

            try:
                self.stepper = QRGivensStepper(A, b, change_tol=1e-10)
            except ValueError as e:
                messagebox.showerror("Fehler", str(e))
                return

            self.started = True
            self.selected_target = None
            self.mode = "choose"

            self._set_original_state(True)
            self._refresh_current_display(initial=True)

            cnt = self._count_eliminables()
            self._update_selected_display()
            self._update_main_button()
            self._update_hint(f"Wählen Sie jetzt im rechten Kasten das erste Element (grün). Offen: {cnt} Eliminierungen.")
            return

        if not self.stepper:
            return

        if self.stepper.phase == "rot":
            if self.mode == "choose":
                cnt = self._count_eliminables()
                self._update_hint(f"Klicke im rechten Kasten ein grünes Element an. Offen: {cnt} Eliminierungen.")
                return

            if self.mode == "run":
                if self.selected_target is None:
                    self.mode = "choose"
                    self._update_main_button()
                    self._update_hint("Keine Auswahl. Klicke im rechten Kasten ein grünes Element an.")
                    return

                step = self.stepper.next_step(target=self.selected_target)
                snap = self.stepper.snapshot()
                self._append_step_card(snap, step)

                self.selected_target = None
                self.mode = "choose"

                self._refresh_current_display()
                self._update_selected_display()

                if self._count_eliminables() == 0:
                    err = self.stepper.switch_to_backsub()
                    self._refresh_current_display()
                    if err is None:
                        self._update_hint("Rotation fertig. Jetzt Rückwärtseinsetzen. Button klicken für nächsten Rückwärts-Schritt.")
                    else:
                        self._update_hint("Rotation fertig, aber Rückwärts nicht möglich: " + err)
                else:
                    cnt = self._count_eliminables()
                    self._update_hint(f"Schritt fertig. Wähle das nächste Element (grün) im rechten Kasten. Offen: {cnt}.")

                self._update_main_button()
                return

        if self.stepper.phase == "back":
            step = self.stepper.next_step(target=None)
            snap = self.stepper.snapshot()
            self._append_step_card(snap, step)

            self._refresh_current_display()
            self._update_selected_display()

            if self.stepper.phase == "done":
                self._append_final_solution_card()
                self.mode = "done"
                self.selected_target = None
                self.btn_main.configure(text="Fertig", state="disabled")
                self._update_hint("Berechnung abgeschlossen. Reset zum Neustart.")
                return

            self._update_hint("Rückwärts-Schritt fertig. Nächsten Rückwärts-Schritt mit Button ausführen.")
            self._update_main_button()
            return

    def on_current_matrix_click(self, r: int, c: int):
        if not self.stepper or self.stepper.phase != "rot":
            return
        if self.mode != "choose":
            return
        if not self._is_eliminable(r, c):
            return

        self.selected_target = (r, c)
        self.mode = "run"
        self._apply_current_highlights()
        self._update_selected_display()
        self._update_main_button()
        self._update_hint(f"Element R[{r+1},{c+1}] ausgewählt. Button klicken um Iteration {self.step_count + 1} zu starten.")

    def on_reset(self):
        self.history.clear()
        self.step_count = 0
        self.started = False
        self._finish_run(reset_values=True)

    def _finish_run(self, reset_values: bool = False):
        self.stepper = None
        self.mode = "init"
        self.selected_target = None

        self._set_original_state(False)

        if reset_values:
            m, n = self._get_mn()
            for r in range(m):
                for c in range(n):
                    e = self.entries_A0[r][c]
                    e.configure(state="normal", bg=self.bg_default)
                    e.delete(0, tk.END)
                    e.insert(0, "0")
                eb = self.entries_b0[r]
                eb.configure(state="normal", bg=self.bg_default)
                eb.delete(0, tk.END)
                eb.insert(0, "0")

        self._refresh_current_display()
        self._update_selected_display()
        self._update_main_button()
        self._update_hint("Klicke auf 'Wähle zu eliminierendes Element'. Dann wähle im rechten Kasten das erste Element (grün).")

    def _append_step_card(self, snap: Dict[str, Any], step: Step, extra_title: Optional[str] = None):
        self.step_count += 1
        kind_map = {"givens": "ROTATION", "back_sub": "RÜCKWÄRTS", "done": "FERTIG"}
        kind_txt = kind_map.get(step.kind, step.kind.upper())
        title = extra_title if extra_title else f"Schritt {self.step_count}: {kind_txt}"

        changed = set(step.changed or [])
        pivot_r, pivot_c = step.pivot

        card = ttk.Frame(self.history.inner, padding=10, style="Card.TFrame")
        card.pack(fill="x", pady=8, padx=2)

        header = ttk.Label(card, text=title, style="Small.TLabel")
        header.pack(anchor="w")

        body = ttk.Frame(card, style="Card.TFrame")
        body.pack(fill="x", pady=(6, 0))
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

        left = ttk.Frame(body, style="Card.TFrame")
        right = ttk.Frame(body, style="Card.TFrame")
        left.grid(row=0, column=0, sticky="nw", padx=(0, 12))
        right.grid(row=0, column=1, sticky="new")

        ttk.Label(left, text="G, R und y", style="Small.TLabel").pack(anchor="w")

        grids = ttk.Frame(left, style="Card.TFrame")
        grids.pack(anchor="w", pady=(4, 0))

        frameG = ttk.Frame(grids, style="Card.TFrame")
        frameR = ttk.Frame(grids, style="Card.TFrame")
        frameY = ttk.Frame(grids, style="Card.TFrame")

        frameG.pack(side="left", padx=(0, 16), anchor="n")
        frameR.pack(side="left", padx=(0, 16), anchor="n")
        frameY.pack(side="left", anchor="n")

        ttk.Label(frameG, text="G", style="Small.TLabel").pack(anchor="w")
        ttk.Label(frameR, text="R", style="Small.TLabel").pack(anchor="w")
        ttk.Label(frameY, text="y", style="Small.TLabel").pack(anchor="w")

        self._draw_matrix(frameG, "G", snap["G"], changed, pivot_r, pivot_c, step)
        self._draw_matrix(frameR, "R", snap["R"], changed, pivot_r, pivot_c, step)
        self._draw_vector(frameY, "y", snap["y"], changed)

        #ttk.Label(right, text="Rechnung", style="Small.TLabel").pack(anchor="w")
        msg = (step.message or "").strip()
        txt = tk.Text(
            right,
            height=max(10, min(16, msg.count("\n") + 6)),
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

    def _draw_matrix(self, parent: ttk.Frame, name: str, M: List[List[float]],
                     changed: set, pivot_r: int, pivot_c: int, step: Step):
        grid = ttk.Frame(parent, style="Card.TFrame")
        grid.pack(anchor="w", pady=(4, 0))

        rows = len(M)
        cols = len(M[0]) if rows else 0

        for r in range(rows):
            for c in range(cols):
                val = self._fmt(M[r][c])
                bg = "#ffffff"
                if (name, r, c) in changed:
                    bg = "#ffb3c6"
                if step.kind == "givens" and name == "R" and r == pivot_r and c == pivot_c:
                    bg = "#93c5fd"
                tk.Label(
                    grid, text=val, width=12,
                    bg=bg, relief="solid", borderwidth=1,
                    font=("Consolas", 12)
                ).grid(row=r, column=c, padx=2, pady=2)

    def _draw_vector(self, parent: ttk.Frame, name: str, v: List[float], changed: set):
        grid = ttk.Frame(parent, style="Card.TFrame")
        grid.pack(anchor="w", pady=(4, 0))
        for r in range(len(v)):
            val = self._fmt(v[r])
            bg = "#ffffff"
            if (name, r, 0) in changed:
                bg = "#ffb3c6"
            tk.Label(
                grid, text=val, width=12,
                bg=bg, relief="solid", borderwidth=1,
                font=("Consolas", 12)
            ).grid(row=r, column=0, padx=2, pady=2)

    def _append_final_solution_card(self):
        if not self.stepper:
            return
        x_final = self.stepper.x[:]
        parts = [f"x{i+1} = {self._fmt(v)}" for i, v in enumerate(x_final)]
        sol = "   |   ".join(parts)
        messagebox.showinfo("Lösung", sol)