from __future__ import annotations

import json
import os
import sys
import threading
import tempfile
import tkinter as tk

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from dgl_comparison_window import ComparisonWindow

if sys.platform == "win32":
    PIPE_PATH = r"\\.\pipe\dgl_compare"
    LOCK_PATH = os.path.join(tempfile.gettempdir(), "dgl_compare.lock")
else:
    PIPE_PATH = "/tmp/dgl_compare.fifo"
    LOCK_PATH = "/tmp/dgl_compare.lock"


class SharedCompare:
    def __init__(self, master: tk.Tk):
        self._master = master
        self._window: ComparisonWindow | None = None
        self._is_host = False

        self._pending: list[dict] = []
        self._lock = threading.Lock()
        self._compare_signatures: set[str] = set()

        self._is_host = self._try_become_host()

    def _try_become_host(self) -> bool:
        if os.path.exists(LOCK_PATH):
            return False
        return self._force_become_host()

    def _force_become_host(self) -> bool:
        try:
            with open(LOCK_PATH, "w", encoding="utf-8") as f:
                f.write(str(os.getpid()))
        except OSError:
            return False

        if sys.platform != "win32":
            try:
                if os.path.exists(PIPE_PATH):
                    os.remove(PIPE_PATH)
            except OSError:
                pass

            try:
                os.mkfifo(PIPE_PATH)
            except OSError:
                pass

        self._window = self._create_window()
        self._master.after(120, self._poll)

        reader = _win_pipe_reader if sys.platform == "win32" else _fifo_reader
        t = threading.Thread(target=reader, args=(self,), daemon=True)
        t.start()

        self._master.bind("<Destroy>", self._on_master_destroy)
        self._is_host = True
        return True

    def _create_window(self) -> ComparisonWindow:
        return ComparisonWindow(
            self._master,
            on_clear_callback=self._handle_window_clear,
            on_close_callback=self._handle_window_close,
        )

    def _handle_window_clear(self):
        self._compare_signatures.clear()

    def _handle_window_close(self):
        self._compare_signatures.clear()
        self._window = None

    def _on_master_destroy(self, event):
        if event.widget is not self._master:
            return

        try:
            if os.path.exists(LOCK_PATH):
                os.remove(LOCK_PATH)
        except OSError:
            pass

        if sys.platform != "win32":
            try:
                if os.path.exists(PIPE_PATH):
                    os.remove(PIPE_PATH)
            except OSError:
                pass

    def enqueue(self, msg: dict):
        with self._lock:
            self._pending.append(msg)

    def _poll(self):
        with self._lock:
            msgs, self._pending = self._pending, []

        for msg in msgs:
            self._apply(msg)

        try:
            self._master.after(120, self._poll)
        except tk.TclError:
            pass

    def _ensure_window(self):
        if self._window is None:
            self._window = self._create_window()
            self._compare_signatures.clear()
            return

        try:
            if hasattr(self._window, "is_closed") and self._window.is_closed():
                self._window = self._create_window()
                self._compare_signatures.clear()
        except Exception:
            self._window = self._create_window()
            self._compare_signatures.clear()

    def show(self):
        if not self._is_host:
            return

        self._ensure_window()
        if self._window is None:
            return

        try:
            self._window.deiconify()
        except Exception:
            pass

        try:
            self._window.lift()
            self._window.focus_force()
        except Exception:
            pass

    def _apply(self, msg: dict):
        if not self._is_host:
            return

        action = msg.get("action")

        if action == "add":
            key = str(msg.get("key", ""))
            label = msg.get("label", "")
            ts = msg.get("ts", [])
            ys = msg.get("ys", [])

            if not ts or not ys or len(ts) != len(ys):
                return

            self._ensure_window()
            if self._window is None:
                return

            if key in self._compare_signatures:
                self.show()
                return

            try:
                added = self._window.add_solution(ts, ys, label)
            except Exception as e:
                print(f"[shared_compare] Fehler in add_solution: {e}")
                return

            if added:
                self._compare_signatures.add(key)
                self.show()

        elif action == "clear":
            self._handle_window_clear()
            if self._window is not None:
                try:
                    self._window.plotter.clear_all()
                except Exception:
                    pass
            self.show()

    def _send(self, msg: dict):
        if self._is_host:
            self.enqueue(msg)
            return

        try:
            if sys.platform == "win32":
                self._win_send(json.dumps(msg))
            else:
                with open(PIPE_PATH, "w", encoding="utf-8") as pipe:
                    pipe.write(json.dumps(msg) + "\n")
            return

        except OSError as e:
            print(f"[shared_compare] Pipe-Fehler: {e}")

        try:
            if os.path.exists(LOCK_PATH):
                os.remove(LOCK_PATH)
        except OSError:
            pass

        if self._force_become_host():
            self.enqueue(msg)
            self.show()
        else:
            print("[shared_compare] Konnte nach Pipe-Fehler nicht Host werden.")

    def _win_send(self, text: str):
        import ctypes

        kernel32 = ctypes.windll.kernel32
        data = text.encode("utf-8")

        h = kernel32.CreateFileW(
            PIPE_PATH,
            0x40000000,
            0,
            None,
            3,
            0,
            None,
        )

        if h == ctypes.c_void_p(-1).value:
            raise OSError("Pipe nicht erreichbar")

        written = ctypes.c_ulong(0)
        ok = kernel32.WriteFile(h, data, len(data), ctypes.byref(written), None)
        kernel32.CloseHandle(h)

        if not ok:
            raise OSError("Schreiben in Pipe fehlgeschlagen")

    def add(self, signature: tuple, label: str, ts: list, ys: list):
        self._send({
            "action": "add",
            "key": str(signature),
            "label": label,
            "ts": ts,
            "ys": ys,
        })

    def clear(self):
        self._send({"action": "clear"})

    def is_empty(self):
        return len(self._compare_signatures) == 0
        

def _fifo_reader(shared: SharedCompare):
    while True:
        try:
            with open(PIPE_PATH, "r", encoding="utf-8") as pipe:
                for line in pipe:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                        shared.enqueue(msg)
                    except json.JSONDecodeError:
                        pass
        except OSError:
            break


def _win_pipe_reader(shared: SharedCompare):
    import ctypes
    import ctypes.wintypes as wt

    PIPE_ACCESS_INBOUND = 0x00000001
    PIPE_TYPE_MESSAGE = 0x00000004
    PIPE_READMODE_MESSAGE = 0x00000002
    PIPE_WAIT = 0x00000000
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    kernel32 = ctypes.windll.kernel32
    buf = ctypes.create_string_buffer(65536)

    while True:
        h = kernel32.CreateNamedPipeW(
            PIPE_PATH,
            PIPE_ACCESS_INBOUND,
            PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT,
            255,
            65536,
            65536,
            0,
            None,
        )

        if h == INVALID_HANDLE_VALUE:
            break

        connected = kernel32.ConnectNamedPipe(h, None)
        if not connected:
            err = kernel32.GetLastError()
            if err != 535:
                kernel32.CloseHandle(h)
                continue

        while True:
            read = wt.DWORD(0)
            ok = kernel32.ReadFile(h, buf, len(buf), ctypes.byref(read), None)
            if not ok or read.value == 0:
                break

            try:
                msg = json.loads(buf.raw[:read.value].decode("utf-8"))
                shared.enqueue(msg)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        kernel32.CloseHandle(h)