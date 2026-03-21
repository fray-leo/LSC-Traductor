"""
logic/classifier.py

Carga el modelo entrenado y el encoder de etiquetas, y expone
un único método público: predict(landmarks) -> str.

Debe existir model/lsc_classifier.pkl y model/label_encoder.pkl
antes de instanciar esta clase. Si no existen, lanza un error claro
con instrucciones para generarlos.
"""

from pathlib import Path

import joblib
import numpy as np

# ------------------------------------------------------------------
# Rutas
# ------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "lsc_classifier.pkl"
ENCODER_PATH = ROOT / "model" / "label_encoder.pkl"

# ------------------------------------------------------------------
# Clase principal
# ------------------------------------------------------------------

class SignClassifier:
    """
    Clasifica un vector de landmarks en el nombre de una seña LSC.

    Uso:
        clf = SignClassifier()
        seña = clf.predict(landmarks)  # landmarks: lista de 63 floats
    """

    # Si la confianza del modelo es menor a este umbral, se devuelve
    # UNCERTAIN en lugar de una predicción potencialmente errónea.
    # Ajustar según el comportamiento observado en demo.
    CONFIDENCE_THRESHOLD = 0.6

    # Valor devuelto cuando ninguna seña supera el umbral de confianza.
    UNCERTAIN = "..."

    def __init__(self):
        self._clf, self._encoder = self._load_model()

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def predict(self, landmarks: list[float]) -> str:
        """
        Predice la seña correspondiente a un vector de landmarks.

        Args:
            landmarks: lista de 63 floats generada por HandCapture.get_landmarks().

        Returns:
            Nombre de la seña (ej. "hola") si la confianza supera el umbral,
            o UNCERTAIN ("...") en caso contrario.
        """
        if not landmarks or len(landmarks) != 63:
            return self.UNCERTAIN

        X = np.array(landmarks, dtype=np.float32).reshape(1, -1)

        probabilities = self._clf.predict_proba(X)[0]
        best_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[best_idx])

        if confidence < self.CONFIDENCE_THRESHOLD:
            return self.UNCERTAIN

        return str(self._encoder.inverse_transform([best_idx])[0])

    def predict_with_confidence(self, landmarks: list[float]) -> tuple[str, float]:
        """
        Igual que predict(), pero devuelve también el valor de confianza.
        Útil para mostrar una barra de confianza en la interfaz.

        Returns:
            Tupla (nombre_de_la_seña, confianza) donde confianza ∈ [0.0, 1.0].
            Si no supera el umbral, devuelve (UNCERTAIN, confianza).
        """
        if not landmarks or len(landmarks) != 63:
            return self.UNCERTAIN, 0.0

        X = np.array(landmarks, dtype=np.float32).reshape(1, -1)

        probabilities = self._clf.predict_proba(X)[0]
        best_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[best_idx])

        if confidence < self.CONFIDENCE_THRESHOLD:
            return self.UNCERTAIN, confidence

        label = str(self._encoder.inverse_transform([best_idx])[0])
        return label, confidence

    @property
    def signs(self) -> list[str]:
        """Lista de todas las señas que el modelo puede reconocer."""
        return list(self._encoder.classes_)

    # ------------------------------------------------------------------
    # Carga del modelo
    # ------------------------------------------------------------------

    @staticmethod
    def _load_model():
        """
        Carga el clasificador y el encoder desde disco.
        Lanza RuntimeError con instrucciones claras si los archivos no existen.
        """
        missing = [p for p in (MODEL_PATH, ENCODER_PATH) if not p.exists()]
        if missing:
            files = "\n  ".join(str(p) for p in missing)
            raise RuntimeError(
                f"No se encontraron los archivos del modelo:\n  {files}\n\n"
                "Para generarlos, corre desde la raíz del proyecto:\n"
                "  1. python -m logic.trainer --collect-all   (grabar señas)\n"
                "  2. python -m logic.trainer --train          (entrenar modelo)\n"
            )

        clf = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        return clf, encoder