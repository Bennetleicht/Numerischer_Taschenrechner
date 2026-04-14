from __future__ import annotations
import math


class CalculatorError(Exception):
    pass


class BasicCalculatorSolver:
    def __init__(self):
        self.clear_all()

    def clear_all(self):
        self.current = "0"
        self.first_operand = None
        self.operator = None
        self.waiting_for_new_input = False
        self.error = False

    # Clear Entry
    def clear_entry(self):
        if self.error:
            self.clear_all()
            return
        self.current = "0"

    # Backspace
    def backspace(self):
        if self.error:
            self.clear_all()
            return

        if self.waiting_for_new_input:
            return

        if len(self.current) <= 1:
            self.current = "0"
        else:
            self.current = self.current[:-1]

    # Digiteneingabe
    def input_digit(self, digit: str):
        if self.error:
            self.clear_all()

        if digit not in "0123456789":
            raise CalculatorError("Ungültige Ziffer.")

        if self.waiting_for_new_input:
            self.current = digit
            self.waiting_for_new_input = False
            return

        if self.current == "0":
            self.current = digit
        else:
            self.current += digit

    # Dezimalpunkt
    def input_decimal(self):
        if self.error:
            self.clear_all()

        if self.waiting_for_new_input:
            self.current = "0."
            self.waiting_for_new_input = False
            return

        if "." not in self.current:
            self.current += "."

    # Operator setzen
    def set_operator(self, op: str):
        if self.error:
            self.clear_all()

        if op not in {"+", "-", "*", "/", "^"}:
            raise CalculatorError("Ungültiger Operator.")

        current_value = self._current_value()

        if self.operator is not None and not self.waiting_for_new_input:
            result = self._apply_operator(self.first_operand, current_value, self.operator)
            self.first_operand = result
            self.current = self._format_number(result)
        else:
            self.first_operand = current_value

        self.operator = op
        self.waiting_for_new_input = True

    # Ergebnis berechnen
    def calculate_result(self):
        if self.error:
            self.clear_all()
            return self.current

        if self.operator is None:
            return self.current

        second_operand = self._current_value()
        result = self._apply_operator(self.first_operand, second_operand, self.operator)

        self.first_operand = result
        self.current = self._format_number(result)
        self.waiting_for_new_input = True
        self.operator = None

        return self.current

    # Wurzel berechnen
    def sqrt(self):
        if self.error:
            self.clear_all()

        value = self._current_value()
        if value < 0:
            self._set_error("Wurzel aus negativer Zahl ist nicht definiert.")

        result = math.sqrt(value)
        self.current = self._format_number(result)
        self.waiting_for_new_input = True
        return self.current


    def get_display(self) -> str:
        return self.current

    
    def get_expression(self) -> str:
        if self.error:
            return ""

        if self.operator is None or self.first_operand is None:
            return ""

        op_map = {
            "+": "+",
            "-": "-",
            "*": "×",
            "/": "÷",
            "^": "^",
        }

        first = self._format_number(self.first_operand)

        if self.waiting_for_new_input:
            return f"{first} {op_map[self.operator]}"

        return f"{first} {op_map[self.operator]} {self.current}"

    
    def _current_value(self) -> float:
        try:
            return float(self.current)
        except ValueError as exc:
            raise CalculatorError("Ungültiger Zahlenwert.") from exc

    # Operator anwenden
    def _apply_operator(self, a: float, b: float, op: str) -> float:
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            if b == 0:
                self._set_error("Division durch 0")
            return a / b
        if op == "^":
            try:
                result = a ** b
            except Exception as exc:
                raise CalculatorError("Potenz konnte nicht berechnet werden.") from exc

            if math.isinf(result) or math.isnan(result):
                self._set_error("Ungültiges Ergebnis.")
            return result

        raise CalculatorError("Unbekannter Operator.")

    # Ergebnis formatieren
    def _format_number(self, value: float) -> str:
        if math.isinf(value) or math.isnan(value):
            self._set_error("Ungültiges Ergebnis.")

        if abs(value) < 1e-15:
            value = 0.0

        text = f"{value:.12g}"

        if "." in text and "e" not in text and "E" not in text:
            text = text.rstrip("0").rstrip(".")

        return text

    def _set_error(self, message: str):
        self.current = "Fehler"
        self.error = True
        self.first_operand = None
        self.operator = None
        self.waiting_for_new_input = False
        raise CalculatorError(message)