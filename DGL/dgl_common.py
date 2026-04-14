from __future__ import annotations

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

_TSYM = sp.Symbol("t")
_YSYM = sp.Symbol("y")

_LOCALS = {
    "t": _TSYM,
    "y": _YSYM,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "pi": sp.pi,
    "E": sp.E,
}

_TRANS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

def parse_function(expr_str: str):
    expr = parse_expr(expr_str, local_dict=_LOCALS, transformations=_TRANS)
    f = sp.lambdify((_TSYM, _YSYM), expr, modules=["numpy"])
    return expr, f
