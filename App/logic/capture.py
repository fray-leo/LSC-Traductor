import cv2
import mediapipe as mp
import numpy as np


class HandCapture:
    """
    Abre la webcam, detecta la mano con MediaPipe y extrae
    los 21 landmarks como un array plano de 63 floats (x, y, z por punto).
    """

    NUM_LANDMARKS = 21
    VECTOR_SIZE = NUM_LANDMARKS * 3  # 63

    def __init__(self, camera_index: int = 0, max_hands: int = 1):
        """
        Args:
            camera_index: índice de la cámara (0 = webcam por defecto).
            max_hands: cuántas manos detectar simultáneamente (1 para el MVP).
        """
        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"No se pudo abrir la cámara con índice {camera_index}. "
                "Verifica que la webcam esté conectada y no esté en uso por otra app."
            )

        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles

        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )

        self._last_frame = None
        self._last_landmarks = None

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def read(self) -> bool:
        """
        Captura y procesa un frame de la cámara.
        Debe llamarse una vez por iteración del loop principal antes de
        usar get_frame() o get_landmarks().

        Returns:
            True si el frame se leyó correctamente, False si la cámara falló.
        """
        ok, frame = self._cap.read()
        if not ok:
            return False

        frame = cv2.flip(frame, 1)  # espejo horizontal (más natural para el usuario)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self._hands.process(rgb)

        self._last_landmarks = None

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]  # solo la primera mano detectada

            # Dibujar sobre el frame
            self._mp_draw.draw_landmarks(
                frame,
                hand,
                self._mp_hands.HAND_CONNECTIONS,
                self._mp_styles.get_default_hand_landmarks_style(),
                self._mp_styles.get_default_hand_connections_style(),
            )

            self._last_landmarks = self._extract_landmarks(hand, frame.shape)

        self._last_frame = frame
        return True

    def get_frame(self) -> np.ndarray | None:
        """
        Devuelve el último frame capturado (BGR) con los landmarks
        dibujados si se detectó una mano. None si no se ha leído ningún frame.
        """
        return self._last_frame

    def get_landmarks(self) -> list[float] | None:
        """
        Devuelve el vector de landmarks de la mano detectada:
        63 floats normalizados [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20].
        Devuelve None si no se detectó ninguna mano en el último frame.
        """
        return self._last_landmarks

    def release(self):
        """Libera la cámara y los recursos de MediaPipe. Llamar al cerrar la app."""
        self._cap.release()
        self._hands.close()

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _extract_landmarks(self, hand_landmarks, frame_shape: tuple) -> list[float]:
        """
        Extrae los 21 landmarks y los normaliza respecto al bounding box
        de la mano (no respecto al frame completo), lo que hace el vector
        invariante a la posición de la mano en pantalla.

        Args:
            hand_landmarks: objeto MediaPipe con los 21 puntos.
            frame_shape: (height, width, channels) del frame actual.

        Returns:
            Lista de 63 floats en el rango aproximado [-1, 1].
        """
        h, w = frame_shape[:2]

        # Coordenadas en píxeles
        points = [
            (lm.x * w, lm.y * h, lm.z * w)
            for lm in hand_landmarks.landmark
        ]

        # Normalizar respecto al bounding box de la mano
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        bbox_size = max(x_max - x_min, y_max - y_min) or 1.0  # evitar división por cero

        normalized = []
        for x, y, z in points:
            normalized.append((x - x_min) / bbox_size)
            normalized.append((y - y_min) / bbox_size)
            normalized.append(z / bbox_size)

        return normalized

    # ------------------------------------------------------------------
    # Context manager (permite usar `with HandCapture() as cap:`)
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()