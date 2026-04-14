from tkinter import *
import tkinter.font as tkfont
import tkinter.messagebox as mbox
from PIL import Image, ImageTk
import os
import sys
import subprocess


# Farben / Design
BG = "#f5f7fb"
SURFACE = "#ffffff"
SURFACE_HOVER = "#f8fbff"
BORDER = "#dbe3ef"
TEXT = "#1f2937"
TEXT_MUTED = "#6b7280"
ACCENT = "#2563eb"
ACCENT_SOFT = "#eaf1ff"
BACK_BTN_BG = "#eef2f7"
BACK_BTN_HOVER = "#e2e8f0"



def hex_to_rgb(value):
    value = value.lstrip("#")
    return tuple(int(value[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    return "#%02x%02x%02x" % rgb


def mix_color(c1, c2, factor=0.5):
    r1, g1, b1 = hex_to_rgb(c1)
    r2, g2, b2 = hex_to_rgb(c2)
    r = int(r1 + (r2 - r1) * factor)
    g = int(g1 + (g2 - g1) * factor)
    b = int(b1 + (b2 - b1) * factor)
    return rgb_to_hex((r, g, b))


# Klickbare Card
class MenuCard(Frame):
    def __init__(
        self,
        parent,
        title,
        subtitle="",
        icon="•",
        command=None,
        width=320,
        height=125,
        big=False
    ):
        super().__init__(
            parent,
            bg=SURFACE,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=BORDER,
            bd=0
        )

        self.command = command
        self.default_bg = SURFACE
        self.hover_bg = SURFACE_HOVER
        self.default_border = BORDER
        self.hover_border = ACCENT

        if big:
            height = 145

        self.configure(width=width, height=height)
        self.grid_propagate(False)
        self.pack_propagate(False)

        self.inner = Frame(self, bg=self.default_bg)
        self.inner.pack(fill="both", expand=True, padx=1, pady=1)

        self.left = Frame(self.inner, bg=self.default_bg)
        self.left.pack(side="left", fill="both", expand=True, padx=(18, 10), pady=18)

        self.icon_box = Frame(
            self.left,
            bg=ACCENT_SOFT,
            width=52,
            height=52,
            highlightthickness=0,
            bd=0
        )
        self.icon_box.pack(anchor="w")
        self.icon_box.pack_propagate(False)

        self.icon_label = Label(
            self.icon_box,
            text=icon,
            font=("Segoe UI Symbol", 26),
            fg=ACCENT,
            bg=ACCENT_SOFT
        )
        self.icon_label.pack(expand=True)

        self.title_label = Label(
            self.left,
            text=title,
            font=("Segoe UI", 21, "bold"),
            fg=TEXT,
            bg=self.default_bg,
            anchor="w",
            justify="left"
        )
        self.title_label.pack(anchor="w", pady=(12, 2))

        self.subtitle_label = Label(
            self.left,
            text=subtitle,
            font=("Segoe UI", 14),
            fg=TEXT_MUTED,
            bg=self.default_bg,
            anchor="w",
            justify="left",
            wraplength=width - 110
        )
        self.subtitle_label.pack(anchor="w")

        self.arrow_label = Label(
            self.inner,
            text="›",
            font=("Segoe UI", 28),
            fg="#7c8aa5",
            bg=self.default_bg
        )
        self.arrow_label.pack(side="right", padx=(8, 20))

        self.bind_all_widgets()

    
    def bind_all_widgets(self):
        widgets = [
            self, self.inner, self.left, self.icon_box,
            self.icon_label, self.title_label,
            self.subtitle_label, self.arrow_label
        ]
        for w in widgets:
            w.bind("<Enter>", self.on_enter)
            w.bind("<Leave>", self.on_leave)
            w.bind("<Button-1>", self.on_click)
            w.configure(cursor="hand2")

    
    def set_bg_recursive(self, bg):
        self.inner.configure(bg=bg)
        self.left.configure(bg=bg)
        self.title_label.configure(bg=bg)
        self.subtitle_label.configure(bg=bg)
        self.arrow_label.configure(bg=bg)

    # Hover-Effekte
    def on_enter(self, _event=None):
        self.configure(highlightbackground=self.hover_border, highlightcolor=self.hover_border)
        self.set_bg_recursive(self.hover_bg)
        self.arrow_label.configure(fg=ACCENT)

    # Hover-Effekte
    def on_leave(self, _event=None):
        self.configure(highlightbackground=self.default_border, highlightcolor=self.default_border)
        self.set_bg_recursive(self.default_bg)
        self.arrow_label.configure(fg="#7c8aa5")

    # Klick-Event
    def on_click(self, _event=None):
        if self.command:
            self.command()


#Idee / Bilder-Seite mit Scrollfunktion
class ScrollableImagePage(Frame):
    def __init__(self, parent, image_paths, bg="#ffffff"):
        super().__init__(parent, bg=bg)

        self.bg = bg
        self.image_paths = image_paths
        self.original_images = []
        self.image_labels = []
        self.tk_images = []

        self.canvas = Canvas(self, bg=bg, highlightthickness=0)
        self.scrollbar = Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = Frame(self.canvas, bg=bg)

        self.inner.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas_window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self._build_images()

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _build_images(self):
        for path in self.image_paths:
            container = Frame(self.inner, bg=self.bg)
            container.pack(fill="both", expand=True, padx=20, pady=20)

            if not os.path.exists(path):
                Label(
                    container,
                    text=f"Bild nicht gefunden:\n{path}",
                    font=("Segoe UI", 14),
                    fg="red",
                    bg=self.bg
                ).pack(anchor="w")
                continue

            img = Image.open(path)
            self.original_images.append(img)

            label = Label(container, bg=self.bg)
            label.pack(fill="both", expand=True)
            self.image_labels.append(label)

        self.after(100, self._resize_images)

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)
        self._resize_images()

    def _resize_images(self):
        available_width = self.canvas.winfo_width() - 60
        if available_width <= 100:
            return

        self.tk_images.clear()

        for img, label in zip(self.original_images, self.image_labels):
            w, h = img.size
            scale = available_width / w
            new_w = int(w * scale)
            new_h = int(h * scale)

            resized = img.resize((new_w, new_h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(resized)

            self.tk_images.append(photo)
            label.configure(image=photo)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_linux(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")






# Seiten-Builder
# Jede Seite ist ein Frame, der in der Hauptanwendung angezeigt wird.
def create_page_container(root):
    page = Frame(root, bg=BG)
    page.grid_rowconfigure(0, weight=0)
    page.grid_rowconfigure(1, weight=1)
    page.grid_columnconfigure(0, weight=1)
    return page

# Header mit Titel, Untertitel und optionalen Buttons
def create_header(parent, title, subtitle="", back_command=None, settings_command=None):
    header = Frame(parent, bg=BG)
    header.grid(row=0, column=0, sticky="ew", padx=48, pady=(34, 18))
    header.grid_columnconfigure(0, weight=1)

    text_frame = Frame(header, bg=BG)
    text_frame.grid(row=0, column=0, sticky="w")

    Label(
        text_frame,
        text=title,
        font=("Segoe UI", 36, "bold"),
        fg=TEXT,
        bg=BG
    ).pack(anchor="w")

    if subtitle:
        Label(
            text_frame,
            text=subtitle,
            font=("Segoe UI", 20),
            fg=TEXT_MUTED,
            bg=BG
        ).pack(anchor="w", pady=(6, 0))

    # RECHTS: entweder Zurück oder Einstellungen
    if back_command:
        btn = Label(
            header,
            text="← Zurück",
            font=("Segoe UI", 19, "bold"),
            fg=TEXT,
            bg=BACK_BTN_BG,
            padx=16,
            pady=8,
            cursor="hand2"
        )
        btn.grid(row=0, column=1, sticky="e")

        btn.bind("<Enter>", lambda e: btn.configure(bg=BACK_BTN_HOVER))
        btn.bind("<Leave>", lambda e: btn.configure(bg=BACK_BTN_BG))
        btn.bind("<Button-1>", lambda e: back_command())

    elif settings_command:
        btn = Label(
            header,
            text="⚙",
            font=("Segoe UI", 30),   
            fg=TEXT,
            bg=BG,                  
            cursor="hand2"
        )
        btn.grid(row=0, column=1, sticky="e")

        btn.bind("<Enter>", lambda e: btn.configure(fg=ACCENT))
        btn.bind("<Leave>", lambda e: btn.configure(fg=TEXT))
        btn.bind("<Button-1>", lambda e: settings_command())

# Grid erstellen
def create_card_grid(parent, cards, columns=2, card_width=430, card_height=125):
    content = Frame(parent, bg=BG)
    content.grid(row=1, column=0, sticky="nsew", padx=48, pady=(0, 36))

    for c in range(columns):
        content.grid_columnconfigure(c, weight=1, uniform="col")

    rows = (len(cards) + columns - 1) // columns
    for r in range(rows):
        content.grid_rowconfigure(r, weight=1)

    for index, card in enumerate(cards):
        row = index // columns
        col = index % columns

        card_widget = MenuCard(
            content,
            title=card["title"],
            subtitle=card.get("subtitle", ""),
            icon=card.get("icon", "•"),
            command=card.get("command"),
            width=card_width,
            height=card_height,
            big=card.get("big", False)
        )
        card_widget.grid(row=row, column=col, sticky="nsew", padx=12, pady=12)

    return content

# Variante für größere Karten, z.B. bei LGS mit nur 2 Optionen
def create_simple_menu_frame_custom_size(
    root,
    title,
    subtitle,
    items,
    show_frame,
    go_back,
    card_width=430,
    card_height=120,
    columns=2
):
    page = create_page_container(root)

    create_header(page, title, subtitle, back_command=go_back)

    cards = []
    for item in items:
        text, target, icon, desc = item

        if target[0] == "frame":
            cmd = lambda name=target[1]: show_frame(name)
        else:
            cmd = target[1]

        cards.append({
            "title": text,
            "subtitle": desc,
            "icon": icon,
            "command": cmd
        })

    # eigener kompakter Grid-Bereich, mittig ausgerichtet
    content = Frame(page, bg=BG)
    content.grid(row=1, column=0, sticky="nsew", padx=48, pady=(0, 36))

    content.grid_columnconfigure(0, weight=1)
    content.grid_columnconfigure(1, weight=0)
    content.grid_columnconfigure(2, weight=1)
    content.grid_rowconfigure(0, weight=1)
    content.grid_rowconfigure(2, weight=1)

    cards_frame = Frame(content, bg=BG)
    cards_frame.grid(row=1, column=1)

    for c in range(columns):
        cards_frame.grid_columnconfigure(c, weight=0)

    for index, card in enumerate(cards):
        row = index // columns
        col = index % columns

        card_widget = MenuCard(
            cards_frame,
            title=card["title"],
            subtitle=card.get("subtitle", ""),
            icon=card.get("icon", "•"),
            command=card.get("command"),
            width=card_width,
            height=card_height
        )
        card_widget.grid(row=row, column=col, padx=12, pady=12)

    return page


# Menüseiten
#MainFrame
def create_main_frame(root, show_frame, launch_method):
    page = create_page_container(root)

    create_header(
        page,
        "Numerischer Taschenrechner",
        "Wähle ein Verfahren aus",
        settings_command=lambda: show_frame("settings")
    )

    cards = [
        {
            "title": "Nullstellen",
            "subtitle": "Bisektion, Newton, Heron, Sekante, Regula Falsi",
            "icon": "pq",
            "command": lambda: show_frame("nullstellen")
        },
        {
            "title": "Lineare Gleichungssysteme",
            "subtitle": "Gauss, LR, Cholesky, QR, Jacobi, Gauss-Seidel",
            "icon": "≡",
            "command": lambda: show_frame("lgs")
        },
        {
            "title": "Interpolation",
            "subtitle": "Bézier, Polynominterpolation, Lagrange, Splines",
            "icon": "⌁",
            "command": lambda: show_frame("interpolation")
        },
        {
            "title": "Integration",
            "subtitle": "Newton-Cotes, Gauss-Legendre",
            "icon": "∫",
            "command": lambda: show_frame("integration")
        },
        {
            "title": "DGL",
            "subtitle": "Einschritt- und Mehrschrittverfahren",
            "icon": "y′",
            "command": lambda: show_frame("dgl")
        },
        {
            "title": "Basistaschenrechner",
            "subtitle": "Einfacher Rechner",
            "icon": "⌗",
            "command": lambda: launch_method(os.path.join("Basistaschenrechner", "basistaschenrechner_method.py"))
        },
    ]

    create_card_grid(page, cards, columns=2, card_width=480, card_height=138)
    return page

# Einfaches Menü für Unterkategorien
def create_simple_menu_frame(root, title, subtitle, items, show_frame, go_back):
    page = create_page_container(root)

    create_header(page, title, subtitle, back_command=go_back)

    cards = []
    for item in items:
        text, target, icon, desc = item

        if target[0] == "frame":
            cmd = lambda name=target[1]: show_frame(name)
        else:
            cmd = target[1]

        cards.append({
            "title": text,
            "subtitle": desc,
            "icon": icon,
            "command": cmd
        })

    create_card_grid(page, cards, columns=2, card_width=480, card_height=118)
    return page

# Einstellungsseite
def create_settings_frame(root, go_back):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    page = Frame(root, bg="#f5f7fb")
    page.grid_rowconfigure(1, weight=1)
    page.grid_columnconfigure(0, weight=1)

    create_header(
        page,
        "Einstellungen",
        "Allgemeine Optionen",
        back_command=go_back
    )

    body = Frame(page, bg="#f5f7fb")
    body.grid(row=1, column=0, sticky="nsew", padx=48, pady=(0, 36))
    body.grid_rowconfigure(0, weight=1)
    body.grid_columnconfigure(1, weight=1)

    sidebar = Frame(body, bg="#ffffff", highlightthickness=1, highlightbackground="#dbe3ef")
    sidebar.grid(row=0, column=0, sticky="nsw", padx=(0, 20))
    sidebar.configure(width=240)
    sidebar.grid_propagate(False)

    content = Frame(body, bg="#ffffff", highlightthickness=1, highlightbackground="#dbe3ef")
    content.grid(row=0, column=1, sticky="nsew")
    content.grid_rowconfigure(0, weight=1)
    content.grid_columnconfigure(0, weight=1)

    pages = {}

    
    def show_settings_page(name):
        for p in pages.values():
            p.grid_remove()
        pages[name].grid(row=0, column=0, sticky="nsew")

    
    def make_sidebar_button(parent, text, command):
        btn = Label(
            parent,
            text=text,
            font=("Segoe UI", 15),
            fg="#1f2937",
            bg="#ffffff",
            anchor="w",
            padx=18,
            pady=12,
            cursor="hand2"
        )
        btn.pack(fill="x")

        btn.bind("<Enter>", lambda e: btn.configure(bg="#eef4ff"))
        btn.bind("<Leave>", lambda e: btn.configure(bg="#ffffff"))
        btn.bind("<Button-1>", lambda e: command())
        return btn

    # IDEA / BILDER
    idea_page = ScrollableImagePage(
        content,
        image_paths=[
            os.path.join(BASE_DIR, "Bilder", "01.png"),
            os.path.join(BASE_DIR, "Bilder", "02.png"),
        ],
        bg="#ffffff"
    )
    pages["idee"] = idea_page

    
    # # SPRACHE
    # language_page = Frame(content, bg="#ffffff")
    # Label(
    #     language_page,
    #     text="Sprache",
    #     font=("Segoe UI", 26, "bold"),
    #     fg="#1f2937",
    #     bg="#ffffff"
    # ).pack(anchor="w", padx=30, pady=(30, 10))

    # Label(
    #     language_page,
    #     text="Hier kannst du später die Spracheinstellungen einbauen.",
    #     font=("Segoe UI", 16),
    #     fg="#6b7280",
    #     bg="#ffffff"
    # ).pack(anchor="w", padx=30, pady=(0, 20))

    # pages["sprache"] = language_page
    

    # IMPRESSUM
    impressum_page = Frame(content, bg="#ffffff")
    Label(
        impressum_page,
        text="Impressum",
        font=("Segoe UI", 26, "bold"),
        fg="#1f2937",
        bg="#ffffff"
    ).pack(anchor="w", padx=30, pady=(30, 10))

    Label(
        impressum_page,
        text=(
            "Anbieterkennzeichnung\n"
            "Diese App ist ein studentisches Softwareprojekt an der Hochschule Hannover.\n\n"
            "Herausgeber:\n"
            "Hochschule Hannover\n"
            "Ricklinger Stadtweg 118\n"
            "30459 Hannover\n\n"
            "Projektverantwortung & Betreuung:\n"
            "Prof. Dr. Alexander Vendl\n"
            "E-Mail: alexander.vendl@hs-hannover.de\n\n"
            "Projektteam (Studierende SoSe 2026):\n"
            "Bennet Schweer\n"
            "bennet.schweer@stud.hs-hannover.de\n"
            "Samuel Klar\n"
            "samuel.klar@stud.hs-hannover.de\n\n"
            "Haftungshinweis:\n"
            "Die Inhalte dieser App wurden zu Lehr- und Studienzwecken erstellt.\n"
            "Eine gewerbliche Nutzung ist ausgeschlossen."
        ),
        font=("Segoe UI", 16),
        fg="#374151",
        bg="#ffffff",
        justify="left",
        anchor="nw",
        wraplength=900
    ).pack(fill="both", expand=True, padx=30, pady=(0, 20))

    pages["impressum"] = impressum_page

    make_sidebar_button(sidebar, "Idee", lambda: show_settings_page("idee"))
    #make_sidebar_button(sidebar, "Sprache", lambda: show_settings_page("sprache"))
    make_sidebar_button(sidebar, "Impressum", lambda: show_settings_page("impressum"))

    show_settings_page("idee")
    return page






# Start / Navigation / Launch
def main():
    fenster = Tk()
    fenster.title("Numerischer Taschenrechner")
    fenster.configure(bg=BG)

    #fenster.minsize(1100, 720)

    maximize_and_lock(fenster)

    fenster.columnconfigure(0, weight=1)
    fenster.rowconfigure(0, weight=1)

    frames = {}
    nav_stack = []

    # Methode zum Starten von Unterprogrammen
    def launch_method(script_rel_path: str, *args):
        here = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(here, script_rel_path)

        if not os.path.exists(script_path):
            script_path = os.path.abspath(script_rel_path)

        if not os.path.exists(script_path):
            mbox.showerror(
                "Datei nicht gefunden",
                f"Das Programm wurde nicht gefunden:\n\n{script_path}"
            )
            return

        try:
            subprocess.Popen(
                [sys.executable, script_path, *args],
                cwd=os.path.dirname(script_path) or None
            )
        except Exception as e:
            mbox.showerror(
                "Fehler",
                f"Programm konnte nicht gestartet werden:\n{script_path}\n\n{e}"
            )

    # Navigation
    def show_frame(name):
        current = nav_stack[-1] if nav_stack else None
        if current != name:
            nav_stack.append(name)

        for frame in frames.values():
            frame.grid_remove()

        frames[name].grid(row=0, column=0, sticky="nsew")

    def go_back():
        if len(nav_stack) <= 1:
            return

        nav_stack.pop()
        prev = nav_stack[-1]

        for frame in frames.values():
            frame.grid_remove()

        frames[prev].grid(row=0, column=0, sticky="nsew")


    # Frames bauen
    frames["main"] = create_main_frame(fenster, show_frame, launch_method)

    frames["nullstellen"] = create_simple_menu_frame(
        fenster,
        "Nullstellen",
        "Wähle ein Verfahren aus",
        [
            ("Bisektionsverfahren",
             ("action", lambda: launch_method(os.path.join("Nullstellen", "bisection_method.py"))),
             "1.",
             "Intervallhalbierung zur Nullstellensuche"),

            ("Newton Verfahren",
             ("action", lambda: launch_method(os.path.join("Nullstellen", "newton_method.py"))),
             "2.",
             "Tangentenverfahren mit schneller Konvergenz"),

            ("Wurzelberechnung von Heron",
             ("action", lambda: launch_method(os.path.join("Nullstellen", "heron_method.py"))),
             "3.",
             "Iteratives Newton-Verfahren zur Wurzelberechnung"),

            ("Sekantenverfahren",
             ("action", lambda: launch_method(os.path.join("Nullstellen", "secant_method.py"))),
             "4.",
             "Ableitungsfreies Näherungsverfahren"),

            ("Regula Falsi",
             ("action", lambda: launch_method(os.path.join("Nullstellen", "regula_falsi_method.py"))),
             "5.",
             "Kombination aus Sekantenverfahren und Bisektionsmethode"),
        ],
        show_frame,
        go_back
    )

    frames["lgs"] = create_simple_menu_frame_custom_size(
        fenster,
        "Lineare Gleichungssysteme",
        "Wähle eine Verfahrensklasse",
        [
            ("Inkrementelle Verfahren",
            ("frame", "lgs_inkrementell"),
            "≡",
            "Direkte Lösungsverfahren"),

            ("Iterative Verfahren",
            ("frame", "lgs_iterativ"),
            "↻",
            "Schrittweise Näherungslösungen"),
        ],
        show_frame,
        go_back,
        card_width=780,
        card_height=420
    )

    frames["lgs_inkrementell"] = create_simple_menu_frame(
        fenster,
        "LGS – Inkrementelle Verfahren",
        "Wähle ein Verfahren aus",
        [
            ("Gauss-Algorithmus",
             ("action", lambda: launch_method(os.path.join("LGS", "inkrementell", "gauss_methode.py"))),
             "1.",
             "Eliminationsverfahren"),

            ("LR-Zerlegung",
             ("action", lambda: launch_method(os.path.join("LGS", "inkrementell", "lr_methode.py"))),
             "2.",
             "Matrixzerlegung in L und R"),

            ("Cholesky-Verfahren",
             ("action", lambda: launch_method(os.path.join("LGS", "inkrementell", "cholesky_methode.py"))),
             "3.",
             "Für symmetrisch positiv definite Matrizen"),

            ("QR-Zerlegung mit Givens-Rotation",
             ("action", lambda: launch_method(os.path.join("LGS", "inkrementell", "qr_methode.py"))),
             "4.",
             "Orthogonale Zerlegung mit Rotationen"),
        ],
        show_frame,
        go_back,
    )

    frames["lgs_iterativ"] = create_simple_menu_frame_custom_size(
        fenster,
        "LGS – Iterative Verfahren",
        "Wähle ein Verfahren aus",
        [
            ("Jacobi-Verfahren",
             ("action", lambda: launch_method(os.path.join("LGS", "iterativ", "jacobi_methode.py"), "jacobi")),
             "1.",
             "Näherungsverfahren mit alten Werten"),

            ("Gauss-Seidel-Verfahren",
             ("action", lambda: launch_method(os.path.join("LGS", "iterativ", "gauss_seidel_methode.py"), "gauss")),
             "2.",
             "Näherungsverfahren mit aktuellen Werten"),
        ],
        show_frame,
        go_back,
        card_width=780,
        card_height=420
    )

    frames["interpolation"] = create_simple_menu_frame(
        fenster,
        "Interpolation",
        "Wähle ein Verfahren aus",
        [
            ("Bézierkurven",
             ("action", lambda: launch_method(os.path.join("Interpolation", "bezier_method.py"))),
             "1.",
             "Kurven über Kontrollpunkte"),

            ("Polynominterpolation",
             ("action", lambda: launch_method(os.path.join("Interpolation", "polynom_method.py"))),
             "2.",
             "Interpolation durch ein Gesamtpolynom"),

            ("Lagrange-Polynome",
             ("action", lambda: launch_method(os.path.join("Interpolation", "lagrange_method.py"))),
             "3.",
             "Explizite Interpolationsdarstellung"),

            ("Spline-Interpolation",
             ("action", lambda: launch_method(os.path.join("Interpolation", "spline_method.py"))),
             "4.",
             "Stückweise glatte Interpolation"),
        ],
        show_frame,
        go_back
    )

    frames["integration"] = create_simple_menu_frame_custom_size(
        fenster,
        "Integration",
        "Wähle ein Verfahren aus",
        [
            ("Newton-Cotes-Regeln",
            ("action", lambda: launch_method(os.path.join("Integration", "newton_cotes_method.py"))),
            "1.",
            "Numerische Quadratur mit Stützstellen"),

            ("Gauss-Legendre-Quadratur",
            ("action", lambda: launch_method(os.path.join("Integration", "gauss_legendre_method.py"))),
            "2.",
            "Gewichtete Quadratur hoher Genauigkeit"),
        ],
        show_frame,
        go_back,
        card_width=780,
        card_height=420
    )

    frames["dgl"] = create_simple_menu_frame_custom_size(
        fenster,
        "DGL",
        "Wähle eine Verfahrensklasse",
        [
            ("Einschrittverfahren",
            ("action", lambda: launch_method(os.path.join("DGL", "Einschritt", "einschritt_method.py"))),
            "1.",
            "Euler explizit, Heun, Runge-Kutta, Euler implizit"),

            ("Mehrschrittverfahren",
            ("action", lambda: launch_method(os.path.join("DGL", "Mehrschritt", "mehrschritt_method.py"))),
            "2.",
            "BDF, Adams-Bashforth, Adams-Moulton"),
        ],
        show_frame,
        go_back,
        card_width=780,
        card_height=420
    )

    frames["settings"] = create_settings_frame(fenster, go_back)

    show_frame("main")
    fenster.mainloop()

def maximize_and_lock(fenster: Tk):
    fenster.update_idletasks()

    try:
        fenster.state("zoomed")
    except Exception:
        try:
            fenster.attributes("-zoomed", True)
        except Exception:
            pass

    fenster.update_idletasks()

    w = fenster.winfo_width()
    h = fenster.winfo_height()

    fenster.minsize(w, h)
    fenster.maxsize(w, h)
    fenster.resizable(False, False)


if __name__ == "__main__":
    main()

    