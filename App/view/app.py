"""
view/app.py

Ventana principal de la aplicación. Organiza el layout en dos columnas:
  - Izquierda: feed de la cámara (CameraFeed)
  - Derecha:   panel de traducción (TranslationPanel)

No importa nada de logic/ — recibe todo lo que necesita a través
de callbacks que main.py le pasa en el constructor.
"""

import tkinter as tk

from view.components.camera_feed import CameraFeed
from view.components.translation_panel import TranslationPanel


_BG      = "#1e1e1e"
_BG_BAR  = "#141414"
_FG_BAR  = "#666666"


class App:
    """
    Ventana principal de LSC Traductor.

    Uso desde main.py:
        app = App(
            on_close=cap.release,
            on_clear=db.clear,
        )
        app.update_frame(frame)
        app.update_translation("hola", confidence=0.91)
        app.update_history(db.get_recent())
        app.start()   # bloquea hasta que el usuario cierre la ventana
    """

    WINDOW_TITLE  = "LSC Traductor — Lengua de Señas Colombiana"
    FEED_WIDTH    = 640
    FEED_HEIGHT   = 480
    PANEL_WIDTH   = 320
    MIN_HEIGHT    = 540

    def __init__(self, on_close=None, on_clear=None):
        """
        Args:
            on_close: callable invocado al cerrar la ventana.
                      Debe liberar la cámara y cualquier otro recurso.
            on_clear: callable invocado al pulsar "Limpiar historial".
                      Debe borrar la base de datos.
        """
        self._on_close = on_close

        self._root = tk.Tk()
        self._root.title(self.WINDOW_TITLE)
        self._root.configure(bg=_BG)
        self._root.resizable(False, False)
        self._root.protocol("WM_DELETE_WINDOW", self._handle_close)

        # Tamaño mínimo de la ventana
        total_w = self.FEED_WIDTH + self.PANEL_WIDTH
        total_h = max(self.FEED_HEIGHT, self.MIN_HEIGHT)
        self._root.minsize(total_w, total_h)

        self._build(on_clear)
        self._center_window(total_w, total_h)

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def update_frame(self, frame):
        """
        Envía un frame BGR de OpenCV al widget de cámara.

        Args:
            frame: numpy array BGR de forma (H, W, 3).
        """
        self._feed.update_frame(frame)

    def update_translation(self, sena: str, confidence: float = 0.0):
        """
        Actualiza la seña mostrada y la barra de confianza.

        Args:
            sena:       nombre de la seña o "..." si incierta.
            confidence: valor entre 0.0 y 1.0.
        """
        self._panel.update_translation(sena, confidence)

    def update_history(self, entries: list[tuple]):
        """
        Refresca la lista del historial.

        Args:
            entries: lista de tuplas (sena, timestamp) más reciente primero.
        """
        self._panel.update_history(entries)

    def schedule(self, delay_ms: int, callback):
        """
        Registra una función para ejecutarse después de delay_ms milisegundos.
        Envoltorio de root.after() para que main.py no acceda a _root directamente.
        """
        self._root.after(delay_ms, callback)

    def start(self):
        """Inicia el loop principal de tkinter. Bloquea hasta que se cierre la ventana."""
        self._root.mainloop()

    # ------------------------------------------------------------------
    # Construcción de la interfaz
    # ------------------------------------------------------------------

    def _build(self, on_clear):
        """Construye el layout completo de la ventana."""

        # ── Contenedor principal ───────────────────────────────────────
        main = tk.Frame(self._root, bg=_BG)
        main.pack(fill="both", expand=True)

        # ── Columna izquierda: cámara ──────────────────────────────────
        left = tk.Frame(main, bg="black")
        left.pack(side="left", fill="both")

        self._feed = CameraFeed(
            left,
            width=self.FEED_WIDTH,
            height=self.FEED_HEIGHT,
        )
        self._feed.pack()

        # Etiqueta de estado bajo la cámara
        self._status_label = tk.Label(
            left,
            text="Muestra una seña frente a la cámara",
            bg="black",
            fg=_FG_BAR,
            font=("Arial", 10),
            pady=6,
        )
        self._status_label.pack(fill="x")

        # ── Divisor vertical ──────────────────────────────────────────
        tk.Frame(main, bg="#2d2d2d", width=1).pack(side="left", fill="y")

        # ── Columna derecha: panel de traducción ──────────────────────
        right = tk.Frame(main, bg=_BG, width=self.PANEL_WIDTH)
        right.pack(side="left", fill="both", expand=True)
        right.pack_propagate(False)   # respetar el ancho fijo

        self._panel = TranslationPanel(
            right,
            on_clear=on_clear,
        )
        self._panel.pack(fill="both", expand=True)

        # ── Barra de estado inferior ──────────────────────────────────
        bar = tk.Frame(self._root, bg=_BG_BAR, height=24)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        self._statusbar_label = tk.Label(
            bar,
            text="LSC Traductor  ·  Sección 4 – Grupo 7  ·  Universidad de los Andes",
            bg=_BG_BAR,
            fg=_FG_BAR,
            font=("Arial", 9),
            anchor="w",
            padx=10,
        )
        self._statusbar_label.pack(side="left", fill="y")

    # ------------------------------------------------------------------
    # Eventos
    # ------------------------------------------------------------------

    def _handle_close(self):
        """Llamado al pulsar la X de la ventana. Libera recursos y cierra."""
        if callable(self._on_close):
            self._on_close()
        self._root.destroy()

    def _center_window(self, w: int, h: int):
        """Centra la ventana en la pantalla al abrirse."""
        self._root.update_idletasks()
        screen_w = self._root.winfo_screenwidth()
        screen_h = self._root.winfo_screenheight()
        x = (screen_w - w) // 2
        y = (screen_h - h) // 2
        self._root.geometry(f"{w}x{h}+{x}+{y}")