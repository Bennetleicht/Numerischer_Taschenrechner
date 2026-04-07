from tkinter import *
import tkinter.font as tkfont
import os
import sys
import subprocess


def add_back_button(frame, button_font, go_back):
    # Zurueck-Button
    back_button = Button(
        frame,
        text="Zurück",
        font=button_font,
        bg="#e0e0e0",
        activebackground="#d0d0d0",
        bd=2,
        relief="solid",
        command=go_back
    )
    back_button.grid(row=99, column=0, pady=25)

#erstellt das Hauptmenue mit den 6 Buttons, die zu den verschiedenen Themen fuehren
def create_main_frame(fenster, button_font, show_frame):
    frame = Frame(fenster)

    for i in range(3):
        frame.columnconfigure(i, weight=1)

    # Symmetrisch: oben/unten Abstand + 2 Button-Reihen
    frame.rowconfigure(0, weight=1)
    frame.rowconfigure(1, weight=3)
    frame.rowconfigure(2, weight=3)
    frame.rowconfigure(3, weight=1)

    Nullstellen_button = Button(
        frame,
        text="Nullstellen",
        font=button_font,
        bg="#e0e0e0",
        activebackground="#d0d0d0",
        bd=2,
        relief="solid",
        command=lambda: show_frame("nullstellen")
    )
    Nullstellen_button.grid(row=1, column=0, sticky="nsew", padx=40, pady=35)

    LGS_button = Button(
        frame,
        text="lineare Gleichungssysteme",
        font=button_font,
        bg="#e0e0e0",
        activebackground="#d0d0d0",
        bd=2,
        relief="solid",
        command=lambda: show_frame("lgs")
    )
    LGS_button.grid(row=1, column=1, sticky="nsew", padx=40, pady=35)

    Interpolation_button = Button(
        frame,
        text="Interpolation",
        font=button_font,
        bg="#e0e0e0",
        activebackground="#d0d0d0",
        bd=2,
        relief="solid",
        command=lambda: show_frame("interpolation")
    )
    Interpolation_button.grid(row=1, column=2, sticky="nsew", padx=40, pady=35)

    Integration_button = Button(
        frame,
        text="Integration",
        font=button_font,
        bg="#e0e0e0",
        activebackground="#d0d0d0",
        bd=2,
        relief="solid",
        command=lambda: show_frame("integration")
    )
    Integration_button.grid(row=2, column=0, sticky="nsew", padx=40, pady=35)

    DGL_button = Button(
        frame,
        text="DGL",
        font=button_font,
        bg="#e0e0e0",
        activebackground="#d0d0d0",
        bd=2,
        relief="solid",
        command=lambda: show_frame("dgl")
    )
    DGL_button.grid(row=2, column=1, sticky="nsew", padx=40, pady=35)

    BTR_button = Button(
        frame,
        text="Basistaschenrechner",
        font=button_font,
        bg="#e0e0e0",
        activebackground="#d0d0d0",
        bd=2,
        relief="solid",
        command=lambda: show_frame("Basistaschenrechner")
    )
    BTR_button.grid(row=2, column=2, sticky="nsew", padx=40, pady=35)

    return frame

#erstellt die Untermenues mit Titel, Buttons und Zurueck-Button dynamisch anhand der übergebenen items-Liste
def create_simple_menu_frame(fenster, button_font, title, items, show_frame, go_back):
    """
    Baut eine Seite mit:
    - Titel oben
    - Buttons darunter (einspaltig)
    - Zurueck unten
    """
    frame = Frame(fenster)
    frame.columnconfigure(0, weight=1)

    # Titel / Buttons / Space / Back
    frame.rowconfigure(0, weight=0)
    for r in range(1, 1 + len(items)):
        frame.rowconfigure(r, weight=1)
    frame.rowconfigure(98, weight=1)
    frame.rowconfigure(99, weight=0)

    label = Label(frame, text=title, font=tkfont.Font(size=24))
    label.grid(row=0, column=0, pady=(25, 15))

    row = 1
    for text, target in items:
        if target[0] == "frame":
            cmd = lambda name=target[1]: show_frame(name)
        else:
            cmd = target[1]

        b = Button(
            frame,
            text=text,
            font=button_font,
            bg="#e0e0e0",
            activebackground="#d0d0d0",
            bd=2,
            relief="solid",
            command=cmd
        )
        b.grid(row=row, column=0, sticky="nsew", padx=120, pady=12)
        row += 1

    add_back_button(frame, button_font, go_back)
    return frame


def main():
    fenster = Tk()
    fenster.title("Numerischer Taschenrechner")

    fenster.minsize(480, 320)
    fenster.state("zoomed")
    try:
        fenster.attributes("-zoomed", True)
    except Exception:
        pass

    fenster.columnconfigure(0, weight=1)
    fenster.rowconfigure(0, weight=1)

    button_font = tkfont.Font(size=16)

    frames = {}
    nav_stack = []

    # Startet ein Verfahren in einem eigenen Python-Prozess und übergibt optional Argumente
    def launch_method(script_rel_path: str, *args):
        here = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(here, script_rel_path)

        if not os.path.exists(script_path):
            script_path = os.path.abspath(script_rel_path)

        try:
            subprocess.Popen(
                [sys.executable, script_path, *args],
                cwd=os.path.dirname(script_path) or None
            )
        except Exception as e:
            import tkinter.messagebox as mbox
            mbox.showerror("Fehler", f"Programm konnte nicht gestartet werden:\n{script_path}\n\n{e}")
            
    # Wechselt zu einem anderen Frame und verwaltet die Navigation
    def show_frame(name):
        current = nav_stack[-1] if nav_stack else None
        if current != name:
            nav_stack.append(name)

        for f in frames.values():
            f.grid_remove()
        frames[name].grid(row=0, column=0, sticky="nsew")

    def go_back():
        if len(nav_stack) <= 1:
            return
        nav_stack.pop()
        prev = nav_stack[-1]
        for f in frames.values():
            f.grid_remove()
        frames[prev].grid(row=0, column=0, sticky="nsew")

    # --- Frames bauen ---

    frames["main"] = create_main_frame(fenster, button_font, show_frame)

    frames["nullstellen"] = create_simple_menu_frame(
        fenster, button_font, "Nullstellen",
        [
            ("Bisektionsverfahren", ("action", lambda: launch_method(os.path.join("Nullstellen", "bisection_method.py")))),
            ("Newton Verfahren", ("action", lambda: launch_method(os.path.join("Nullstellen", "newton_method.py")))),
            ("Wurzelberechnung von Heron", ("action", lambda: launch_method(os.path.join("Nullstellen", "heron_method.py")))),
            ("Sekantenverfahren", ("action", lambda: launch_method(os.path.join("Nullstellen", "secant_method.py")))),
            ("Regula Falsi", ("action", lambda: launch_method(os.path.join("Nullstellen", "regula_falsi_method.py")))),
        ],
        show_frame, go_back
    )

    frames["lgs"] = create_simple_menu_frame(
        fenster, button_font, "Lineare Gleichungssysteme",
        [
            ("inkrementelle Verfahren", ("frame", "lgs_inkrementell")),
            ("iterative Verfahren", ("frame", "lgs_iterativ")),
        ],
        show_frame, go_back
    )

    frames["lgs_inkrementell"] = create_simple_menu_frame(
        fenster, button_font, "LGS - inkrementelle Verfahren",
        [
            ("Gauss-Algorithmus", ("action", lambda: launch_method(os.path.join("LGS", "inkrementell", "gauss_methode.py")))),
            ("LR-Zerlegung", ("action", lambda: launch_method(os.path.join("LGS", "inkrementell", "lr_methode.py")))),
            ("Cholesky-Verfahren", ("action", lambda: launch_method(os.path.join("LGS", "inkrementell", "cholesky_methode.py")))),
            ("QR-Zerlegung mit Givens-Rotation", ("action", lambda: launch_method(os.path.join("LGS", "inkrementell", "qr_methode.py")))),
        ],
        show_frame, go_back
    )

    frames["lgs_iterativ"] = create_simple_menu_frame(
        fenster, button_font, "LGS - iterative Verfahren",
        [
            ("Jacobi-Verfahren", ("action", lambda: launch_method(os.path.join('LGS','iterativ','jacobi_methode.py'),"jacobi"))),
            ("Gauss-Seidel-Verfahren", ("action", lambda: launch_method(os.path.join('LGS','iterativ','gauss_seidel_methode.py'),"gauss"))),
        ],
        show_frame, go_back
    )

    frames["interpolation"] = create_simple_menu_frame(
        fenster, button_font, "Interpolation",
        [
            ("Bézierkurven",        ("action", lambda: launch_method(os.path.join("Interpolation", "bezier_method.py")))),
            ("Polynominterpolation",("action", lambda: launch_method(os.path.join("Interpolation", "polynom_method.py")))),
            ("Lagrange-Polynome",   ("action", lambda: launch_method(os.path.join("Interpolation", "lagrange_method.py")))),
            ("Spline-Interpolation",("action", lambda: launch_method(os.path.join("Interpolation", "spline_method.py")))),
        ],
        show_frame, go_back
    )

    frames["integration"] = create_simple_menu_frame(
        fenster, button_font, "Integration",
        [
            ("Newton-Cotes-Regeln", ("action", lambda: launch_method(os.path.join("Integration", "newton_cotes_method.py")))),
            ("Gauss-Legendre-Quadratur", ("action", lambda: launch_method(os.path.join("Integration", "gauss_legendre_method.py")))),
        ],
        show_frame, go_back
    )

    frames["dgl"] = create_simple_menu_frame(
        fenster, button_font, "DGL",
        [
            ("Einschrittverfahren", ("action", lambda: launch_method(os.path.join("DGL", "Einschritt", "einschritt_method.py")))),
            ("Mehrschrittverfahren", ("action", lambda: launch_method(os.path.join("DGL", "Mehrschritt", "mehrschritt_method.py")))),
        ],
        show_frame, go_back
    )

    frames["Basistaschenrechner"] = create_simple_menu_frame(
        fenster, button_font, "Basistaschenrechner",
        [
            ("(kommt noch)", ("action", lambda: print("Basistaschenrechner: kommt noch"))),
        ],
        show_frame, go_back
    )

    show_frame("main")
    fenster.mainloop()


if __name__ == "__main__":
    main()