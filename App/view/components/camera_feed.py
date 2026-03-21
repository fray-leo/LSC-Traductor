"""
view/components/camera_feed.py

Widget de tkinter que muestra el feed de la cámara en tiempo real.
Recibe frames de OpenCV (BGR, numpy array) y los convierte al formato
que tkinter puede renderizar.

No sabe nada de MediaPipe ni del clasificador — solo muestra lo que recibe.
"""

import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk


class CameraFeed(tk.Label):
    """
    Subclase de Label que muestra frames de OpenCV actualizados
    en tiempo real dentro de una ventana tkinter.

    Uso desde app.py:
        feed = CameraFeed(parent, width=640, height=480)
        feed.pack()
        feed.update_frame(frame)   # llamar desde el loop de main.py
        feed.show_placeholder()    # mostrar si la cámara no está disponible
    """

    PLACEHOLDER_COLOR = (30, 30, 30)   # BGR — fondo oscuro cuando no hay frame
    PLACEHOLDER_TEXT  = "Sin señal de camara"

    def __init__(self, parent: tk.Widget, width: int = 640, height: int = 480, **kwargs):
        """
        Args:
            parent: widget padre de tkinter.
            width:  ancho del área de video en píxeles.
            height: alto del área de video en píxeles.
        """
        super().__init__(parent, width=width, height=height,
                         bg="black", anchor="center", **kwargs)
        self._width = width
        self._height = height
        self._current_image = None   # referencia para evitar que el GC lo elimine

        self.show_placeholder()

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def update_frame(self, frame: np.ndarray):
        """
        Actualiza el widget con un nuevo frame de OpenCV.

        Args:
            frame: array BGR de forma (H, W, 3) generado por HandCapture.get_frame().
        """
        if frame is None:
            return

        resized = self._resize(frame)
        image = self._bgr_to_imagetk(resized)
        self._current_image = image          # mantener referencia activa
        self.configure(image=image)

    def show_placeholder(self):
        """
        Muestra un frame negro con texto cuando la cámara no está disponible
        o antes de que llegue el primer frame real.
        """
        placeholder = np.full(
            (self._height, self._width, 3),
            self.PLACEHOLDER_COLOR,
            dtype=np.uint8,
        )
        cv2.putText(
            placeholder,
            self.PLACEHOLDER_TEXT,
            (self._width // 2 - 140, self._height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (120, 120, 120),
            1,
            cv2.LINE_AA,
        )
        image = self._bgr_to_imagetk(placeholder)
        self._current_image = image
        self.configure(image=image)

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        """
        Redimensiona el frame al tamaño del widget preservando la proporción.
        Si el frame ya tiene el tamaño correcto, devuelve el original sin copiar.
        """
        h, w = frame.shape[:2]
        if w == self._width and h == self._height:
            return frame

        # Calcular escala preservando proporción (letterbox)
        scale = min(self._width / w, self._height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Centrar sobre fondo negro si las proporciones no coinciden exactamente
        if new_w != self._width or new_h != self._height:
            canvas = np.zeros((self._height, self._width, 3), dtype=np.uint8)
            x_off = (self._width  - new_w) // 2
            y_off = (self._height - new_h) // 2
            canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
            return canvas

        return resized

    @staticmethod
    def _bgr_to_imagetk(frame: np.ndarray) -> ImageTk.PhotoImage:
        """Convierte un array BGR de OpenCV a PhotoImage de tkinter."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        return ImageTk.PhotoImage(image=pil_image)