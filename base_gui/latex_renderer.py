"""
base_gui/latex_renderer.py
──────────────────────────
Zentrale LaTeX-Render-Utilities für das gesamte Projekt.

Funktionen
----------
render_formula_block(lines, *, bg, dpi, width_in)
    Liste von (latex_str, fontsize) → tk.PhotoImage

render_formula(latex, *, bg, fontsize, dpi)
    Einzelner LaTeX-String → tk.PhotoImage

render_matrix_img(A, rhs, m_labels, *, bg, dpi)
    Numerische Matrix-Gleichung  A · M = rhs  → tk.PhotoImage

Verwendung
----------
    from base_gui.latex_renderer import render_formula_block, render_formula, render_matrix_img

    img = render_formula(r"$\\int_a^b f(x)\\,dx$", fontsize=14)
    label.configure(image=img)
    label.image = img          # Referenz halten!
"""
from __future__ import annotations

from io import BytesIO

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.figure as mfigure
from PIL import Image, ImageTk


# ── Farben / Defaults ─────────────────────────────────────────────────────────

_DEFAULT_BG      = "#f6f7fb"
_DEFAULT_BG_MAT  = "#fef9c3"
_TEXT_COLOR      = "#1e293b"
_MATH_COLOR      = "#374151"


# ── Öffentliche API ───────────────────────────────────────────────────────────

def render_formula_block(
    lines: list[tuple[str, int]],
    bg: str = _DEFAULT_BG,
    dpi: int = 100,
    width_in: float = 6.5,
) -> ImageTk.PhotoImage:
    """Rendert eine Liste von (latex_string, fontsize)-Zeilen als tk.PhotoImage.

    Parameters
    ----------
    lines    : Liste von (latex_str, fontsize)-Tupeln, z. B.
               [(r"$p(x) = \\ldots$", 12), (r"$L_i(x) = \\ldots$", 11)]
    bg       : Hintergrundfarbe als Hex-String
    dpi      : Auflösung
    width_in : Breite der Figur in Zoll

    Returns
    -------
    tk.PhotoImage – muss von der aufrufenden GUI als Attribut gehalten werden,
    damit der GC das Bild nicht zerstört.
    """
    import matplotlib.pyplot as plt

    n = len(lines)
    height_in = max(0.38 * n + 0.18, 0.5)
    fig = mfigure.Figure(figsize=(width_in, height_in), dpi=dpi)
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_facecolor(bg)

    step = 1.0 / (n + 0.3)
    for k, (latex, fs) in enumerate(lines):
        y = 1.0 - (k + 0.6) * step
        ax.text(
            0.02, y, latex,
            fontsize=fs, va="center", ha="left",
            transform=ax.transAxes, color=_TEXT_COLOR,
        )

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=bg, pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    return ImageTk.PhotoImage(Image.open(buf).convert("RGBA"))


def render_formula(
    latex: str,
    bg: str = _DEFAULT_BG,
    fontsize: int = 12,
    dpi: int = 100,
) -> ImageTk.PhotoImage:
    """Rendert einen einzelnen LaTeX-String als tk.PhotoImage.

    Kurzform von render_formula_block() für den Einzelzeilen-Fall.

    Parameters
    ----------
    latex    : LaTeX-String, z. B. r"$\\frac{a}{b}$"
    bg       : Hintergrundfarbe
    fontsize : Schriftgröße
    dpi      : Auflösung
    """
    return render_formula_block([(latex, fontsize)], bg=bg, dpi=dpi, width_in=7.0)


def render_matrix_img(
    A: np.ndarray,
    rhs: np.ndarray,
    m_labels: list[str],
    bg: str = _DEFAULT_BG_MAT,
    dpi: int = 100,
) -> ImageTk.PhotoImage:
    """Rendert  A · M = rhs  als Matrizenbild mit eckigen Klammern.

    Parameters
    ----------
    A        : Koeffizientenmatrix (n × n)
    rhs      : Rechte Seite als 1-D Array der Länge n
    m_labels : Bezeichnungen der Unbekannten (Länge n), z. B. ["m_0","m_1"]
    bg       : Hintergrundfarbe
    dpi      : Auflösung
    """
    import matplotlib.pyplot as plt

    nr, nc = A.shape
    CHAR_W = 0.072

    def fmt(v: float) -> str:
        if abs(v) < 1e-12:
            return "0"
        return f"{v:.4g}"

    # Spaltenbreiten
    col_w = []
    for j in range(nc):
        w = max(len(fmt(A[i, j])) for i in range(nr))
        col_w.append(max(w * CHAR_W + 0.06, 0.38))

    vec_m_w = max(len(lbl) for lbl in m_labels) * CHAR_W + 0.14
    vec_f_w = max(len(fmt(v)) for v in rhs) * CHAR_W + 0.14

    cell_h = 0.27
    pad_l  = 0.30
    eq_gap = 0.22
    fig_w  = pad_l + sum(col_w) + eq_gap + vec_m_w + eq_gap + vec_f_w + 0.16
    fig_h  = (nr + 1.5) * cell_h

    fig = mfigure.Figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.set_axis_off()
    ax.set_facecolor(bg)

    def ry(i):   return fig_h - (i + 2.0) * cell_h
    def hdr_y(): return fig_h - 1.2 * cell_h

    by_top = fig_h - 1.5 * cell_h
    by_bot = fig_h - (nr + 1.5) * cell_h
    mid_y  = (by_top + by_bot) / 2
    lw = 1.6

    def bracket(xl, xr, yt, yb, open_: bool = True):
        d = 0.06
        xs_ = [xl + d, xl, xl, xl + d] if open_ else [xr - d, xr, xr, xr - d]
        ax.plot(xs_, [yt, yt, yb, yb], color=_MATH_COLOR, lw=lw,
                solid_capstyle="round")

    cx_list = []
    x_cur = pad_l
    for w in col_w:
        cx_list.append(x_cur + w / 2)
        x_cur += w

    bx_l = pad_l - 0.05
    bx_r = x_cur + 0.05

    # A-Matrix
    bracket(bx_l, bx_r, by_top, by_bot, True)
    bracket(bx_l, bx_r, by_top, by_bot, False)
    for i in range(nr):
        for j in range(nc):
            ax.text(cx_list[j], ry(i), fmt(A[i, j]),
                    ha="center", va="center", fontsize=9, color=_TEXT_COLOR)

    # ·
    dot_x = bx_r + eq_gap * 0.35
    ax.text(dot_x, mid_y, r"$\cdot$", ha="center", va="center",
            fontsize=11, color=_MATH_COLOR)

    # M-Vektor
    mvl = dot_x + eq_gap * 0.55
    mvr = mvl + vec_m_w
    bracket(mvl, mvr, by_top, by_bot, True)
    bracket(mvl, mvr, by_top, by_bot, False)
    mvc = (mvl + mvr) / 2
    ax.text(mvc, hdr_y(), "$M$", ha="center", va="center",
            fontsize=8, color=_MATH_COLOR, fontweight="bold")
    for i, lbl in enumerate(m_labels):
        ax.text(mvc, ry(i), f"${lbl}$", ha="center", va="center",
                fontsize=9, color=_TEXT_COLOR)

    # =
    eq_x = mvr + eq_gap * 0.35
    ax.text(eq_x, mid_y, "$=$", ha="center", va="center",
            fontsize=11, color=_MATH_COLOR)

    # rhs-Vektor
    fvl = eq_x + eq_gap * 0.55
    fvr = fvl + vec_f_w
    bracket(fvl, fvr, by_top, by_bot, True)
    bracket(fvl, fvr, by_top, by_bot, False)
    fvc = (fvl + fvr) / 2
    ax.text(fvc, hdr_y(), "$d$", ha="center", va="center",
            fontsize=8, color=_MATH_COLOR, fontweight="bold")
    for i in range(nr):
        ax.text(fvc, ry(i), fmt(rhs[i]), ha="center", va="center",
                fontsize=9, color=_TEXT_COLOR)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=bg, pad_inches=0.03)
    plt.close(fig)
    buf.seek(0)
    return ImageTk.PhotoImage(Image.open(buf).convert("RGBA"))
