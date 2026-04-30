import cv2
import mediapipe as mp
import numpy as np


class HandCapture:
    """
    Abre la webcam, detecta manos, pose corporal y rostro con MediaPipe Holistic,
    y extrae los landmarks como un array plano de floats.
    
    Usa referencias del torso (hombros) para normalizar las posiciones de las manos,
    lo que permite capturar el contexto espacial completo necesario para señas LSC.
    
    Landmarks incluidos:
      - Mano izquierda: 21 landmarks × 3 = 63
      - Mano derecha: 21 landmarks × 3 = 63
      - Pose (torso): 25 landmarks × 3 = 75
      - Total: 201 landmarks × 3 = 603 valores
    """

    # MediaPipe Holistic landmarks
    NUM_HAND_LANDMARKS = 21  # por mano
    NUM_POSE_LANDMARKS = 25  # pose completa
    
    # Índices de landmarks de pose relevantes para referencia del torso
    SHOULDER_LEFT = 11   # hombro izquierdo
    SHOULDER_RIGHT = 12  # hombro derecho
    HIP_LEFT = 23        # cadera izquierda
    HIP_RIGHT = 24       # cadera derecha
    
    VECTOR_SIZE = (NUM_HAND_LANDMARKS * 2 + NUM_POSE_LANDMARKS) * 3  # 603

    def __init__(self, camera_index: int = 0, max_hands: int = 2):
        """
        Args:
            camera_index: índice de la cámara (0 = webcam por defecto).
            max_hands: cuántas manos detectar simultáneamente (2 para ambas manos en LSC).
        """
        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"No se pudo abrir la cámara con índice {camera_index}. "
                "Verifica que la webcam esté conectada y no esté en uso por otra app."
            )

        import mediapipe.python.solutions.holistic as mp_holistic
        import mediapipe.python.solutions.drawing_utils as mp_draw
        import mediapipe.python.solutions.drawing_styles as mp_styles

        self._mp_holistic = mp_holistic
        self._mp_draw = mp_draw
        self._mp_styles = mp_styles

        self._holistic = self._mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
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

        results = self._holistic.process(rgb)

        self._last_landmarks = None
        self._mp_draw.draw_landmarks(
            frame,
            results.face_landmarks,
            list(self._mp_holistic.FACEMESH_CONTOURS),
            landmark_drawing_spec=None,
            connection_drawing_spec=self._mp_styles.get_default_face_mesh_contours_style(),
        )
        self._mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            list(self._mp_holistic.POSE_CONNECTIONS),
            landmark_drawing_spec=self._mp_styles.get_default_pose_landmarks_style(),
        )
        
        if results.left_hand_landmarks:
            self._mp_draw.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                list(self._mp_holistic.HAND_CONNECTIONS),
                self._mp_styles.get_default_hand_landmarks_style(),
                self._mp_styles.get_default_hand_connections_style(),
            )
        if results.right_hand_landmarks:
            self._mp_draw.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                list(self._mp_holistic.HAND_CONNECTIONS),
                self._mp_styles.get_default_hand_landmarks_style(),
                self._mp_styles.get_default_hand_connections_style(),
            )

        if results.pose_landmarks and (results.left_hand_landmarks or results.right_hand_landmarks):
            self._last_landmarks = self._extract_landmarks(results, frame.shape)

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
        Devuelve el vector de landmarks de las manos y pose detectadas:
        603 floats normalizados [mano_izq(63), mano_der(63), pose(75)].
        
        La normalización usa los hombros como referencia para hacer el vector
        invariante a la posición del cuerpo en pantalla.
        
        Devuelve None si no se detectó pose o ninguna mano en el último frame.
        """
        return self._last_landmarks

    def release(self):
        """Libera la cámara y los recursos de MediaPipe. Llamar al cerrar la app."""
        self._cap.release()
        self._holistic.close()

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _extract_landmarks(self, results, frame_shape: tuple) -> list[float]:
        """
        Extrae los landmarks de ambas manos y la pose, normalizándolos respecto
        al torso (distancia entre hombros) para hacer el vector invariante a la
        posición y distancia del cuerpo en pantalla.
        
        El orden del vector es:
          [mano_izq(63), mano_der(63), pose(75)] = 201 landmarks × 3 = 603 valores
        
        Args:
            results: objeto MediaPipe HolisticResults con todos los landmarks.
            frame_shape: (height, width, channels) del frame actual.

        Returns:
            Lista de 603 floats normalizados.
        """
        h, w = frame_shape[:2]
        normalized = []

        # Obtener puntos de referencia del torso para normalización
        pose_landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None
        if not pose_landmarks:
            return []
        
        # Calcular escala basada en la distancia entre hombros
        shoulder_left = pose_landmarks[self.SHOULDER_LEFT]
        shoulder_right = pose_landmarks[self.SHOULDER_RIGHT]
        shoulder_distance = ((shoulder_right.x - shoulder_left.x) ** 2 + 
                            (shoulder_right.y - shoulder_left.y) ** 2) ** 0.5
        scale = shoulder_distance if shoulder_distance > 0 else 1.0
        
        # Centro del torso (punto medio entre hombros)
        torso_center_x = (shoulder_left.x + shoulder_right.x) / 2
        torso_center_y = (shoulder_left.y + shoulder_right.y) / 2

        # Extraer landmarks de mano izquierda (si existe)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                # Coordenadas relativas al centro del torso, escaladas por distancia entre hombros
                nx = (lm.x - torso_center_x) / scale
                ny = (lm.y - torso_center_y) / scale
                nz = lm.z / scale
                normalized.extend([nx, ny, nz])
        else:
            # Rellenar con ceros si no se detecta la mano izquierda
            normalized.extend([0.0] * (self.NUM_HAND_LANDMARKS * 3))

        # Extraer landmarks de mano derecha (si existe)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                nx = (lm.x - torso_center_x) / scale
                ny = (lm.y - torso_center_y) / scale
                nz = lm.z / scale
                normalized.extend([nx, ny, nz])
        else:
            # Rellenar con ceros si no se detecta la mano derecha
            normalized.extend([0.0] * (self.NUM_HAND_LANDMARKS * 3))

        # Extraer landmarks de pose completos
        for lm in pose_landmarks:
            nx = (lm.x - torso_center_x) / scale
            ny = (lm.y - torso_center_y) / scale
            nz = lm.z / scale
            normalized.extend([nx, ny, nz])

        return normalized

    # ------------------------------------------------------------------
    # Context manager (permite usar `with HandCapture() as cap:`)
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()