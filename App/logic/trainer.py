"""
logic/trainer.py

Tiene dos modos de uso:
  1. Recolección de datos:  python -m logic.trainer --collect --sign hola
  2. Entrenamiento:         python -m logic.trainer --train

No se ejecuta durante el uso normal de la app.
"""

import argparse
import csv
import os
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from App.logic.capture import HandCapture

# ------------------------------------------------------------------
# Rutas
# ------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "Data" / "landmarks"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "lsc_classifier.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Señas disponibles en el MVP
# ------------------------------------------------------------------

DEFAULT_SIGNS = [
    "hola",
    "gracias",
    "si",
    "no",
    "ayuda",
    "agua",
    "bano",
    "nombre",
    "por_favor",
    "bien",
]

# ------------------------------------------------------------------
# Recolección
# ------------------------------------------------------------------

def collect(sign: str, n_samples: int = 80, countdown: int = 3):
    """
    Graba `n_samples` frames de landmarks para la seña indicada
    y los guarda como filas en data/landmarks/{sign}.csv.

    Si el archivo ya existe, agrega filas (no sobreescribe), lo que
    permite sesiones de grabación incrementales.

    Args:
        sign:       nombre de la seña (ej. "hola").
        n_samples:  cuántas muestras capturar.
        countdown:  segundos de cuenta regresiva antes de empezar a grabar.
    """
    csv_path = DATA_DIR / f"{sign}.csv"
    existing = _count_rows(csv_path)
    print(f"\n── Recolectando '{sign}' ──")
    print(f"   Ya existen {existing} muestras. Se agregarán {n_samples} más.")
    print(f"   Prepara la seña. Grabación inicia en {countdown}s...")

    with HandCapture() as cap:
        # Cuenta regresiva con ventana abierta
        start = time.time()
        while time.time() - start < countdown:
            if not cap.read():
                break
            frame = cap.get_frame()
            if frame is not None:
                remaining = countdown - int(time.time() - start)
                _overlay(frame, f"Preparate para: {sign.upper()}", f"Iniciando en {remaining}s...", color=(60, 180, 60))
                cv2.imshow("LSC Trainer - recoleccion", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return

        # Grabación
        collected = 0
        file_exists = csv_path.exists()

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                # Encabezado solo la primera vez
                header = [f"x{i}" for i in range(HandCapture.VECTOR_SIZE)] + ["label"]
                writer.writerow(header)

            while collected < n_samples:
                if not cap.read():
                    break

                frame = cap.get_frame()
                landmarks = cap.get_landmarks()

                if frame is not None:
                    if landmarks is not None:
                        writer.writerow(landmarks + [sign])
                        collected += 1

                    progress = collected / n_samples
                    _overlay(
                        frame,
                        f"Grabando: {sign.upper()}",
                        f"{collected}/{n_samples} muestras",
                        color=(60, 60, 220),
                        progress=progress,
                    )
                    cv2.imshow("LSC Trainer - recoleccion", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print(f"   Interrumpido. Se guardaron {collected} muestras.")
                    break

        cv2.destroyAllWindows()
    print(f"   Listo. Total acumulado para '{sign}': {_count_rows(csv_path)} muestras.")


def collect_all(n_samples: int = 80, pause_between: int = 2):
    """
    Recorre todas las señas del DEFAULT_SIGNS y llama a collect() para cada una.
    Útil para una sesión de grabación completa de cero.
    """
    print(f"\nSesión completa: {len(DEFAULT_SIGNS)} señas × {n_samples} muestras cada una.")
    for sign in DEFAULT_SIGNS:
        collect(sign, n_samples=n_samples)
        time.sleep(pause_between)
    print("\nSesión completada.")


# ------------------------------------------------------------------
# Entrenamiento
# ------------------------------------------------------------------

def train(test_size: float = 0.2, n_estimators: int = 100):
    """
    Lee todos los CSVs en data/landmarks/, entrena un RandomForestClassifier,
    evalúa en el conjunto de prueba e imprime el reporte por seña.
    Guarda el modelo en model/lsc_classifier.pkl y el encoder en model/label_encoder.pkl.

    Args:
        test_size:     proporción del dataset reservada para prueba (0.0 – 1.0).
        n_estimators:  número de árboles del RandomForest.
    """
    X, y = _load_dataset()

    if len(X) == 0:
        print("No se encontraron datos. Corre primero: python -m logic.trainer --collect --sign <seña>")
        return

    # Codificar etiquetas en enteros
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    print(f"\nDataset cargado: {len(X)} muestras · {len(encoder.classes_)} señas")
    for cls, count in zip(*np.unique(y, return_counts=True)):
        print(f"   {cls:<15} {count} muestras")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )

    print(f"\nEntrenando RandomForest ({n_estimators} árboles)...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1,  # usa todos los núcleos disponibles
    )
    clf.fit(X_train, y_train)

    # Evaluación
    y_pred = clf.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"\nAccuracy en test: {accuracy * 100:.1f}%\n")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    if accuracy < 0.75:
        print("⚠  Accuracy menor a 75%. Considera grabar más muestras por seña.")

    # Guardar modelo y encoder
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    print(f"Modelo guardado en:  {MODEL_PATH}")
    print(f"Encoder guardado en: {ENCODER_PATH}")


# ------------------------------------------------------------------
# Utilidades internas
# ------------------------------------------------------------------

def _load_dataset():
    """Lee todos los CSVs de data/landmarks/ y devuelve (X, y)."""
    X, y = [], []
    csv_files = list(DATA_DIR.glob("*.csv"))

    if not csv_files:
        return np.array(X), np.array(y)

    for csv_path in csv_files:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    features = [float(row[f"x{i}"]) for i in range(HandCapture.VECTOR_SIZE)]
                    X.append(features)
                    y.append(row["label"])
                except (KeyError, ValueError):
                    continue  # ignorar filas corruptas

    return np.array(X, dtype=np.float32), np.array(y)


def _count_rows(csv_path: Path) -> int:
    """Cuenta las filas de datos de un CSV (sin contar el encabezado)."""
    if not csv_path.exists():
        return 0
    with open(csv_path, newline="") as f:
        return max(0, sum(1 for _ in f) - 1)  # -1 por el encabezado


def _overlay(frame, title: str, subtitle: str, color: tuple, progress: float | None = None):
    """
    Dibuja un banner semitransparente en la parte inferior del frame
    con título, subtítulo y barra de progreso opcional.
    """
    if frame is None or not hasattr(frame, "shape"):
        return

    h, w = frame.shape[:2]
    color = tuple(int(c) for c in color)
    banner_h = 70
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - banner_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, title, (16, h - banner_h + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(frame, subtitle, (16, h - banner_h + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    if progress is not None:
        bar_x, bar_y = 16, h - 10
        bar_w = w - 32
        cv2.rectangle(frame, (bar_x, bar_y - 6), (bar_x + bar_w, bar_y), (60, 60, 60), -1)
        filled = int(bar_w * progress)
        if filled > 0:
            cv2.rectangle(frame, (bar_x, bar_y - 6), (bar_x + filled, bar_y), color, -1)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSC Trainer — recolección y entrenamiento")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--collect", action="store_true", help="Modo recolección de datos")
    group.add_argument("--train", action="store_true", help="Modo entrenamiento del modelo")
    group.add_argument("--collect-all", action="store_true", help="Recolectar todas las señas del MVP")

    parser.add_argument("--sign", type=str, help="Nombre de la seña a grabar (solo con --collect)")
    parser.add_argument("--samples", type=int, default=80, help="Muestras por seña (default: 80)")
    parser.add_argument("--estimators", type=int, default=100, help="Árboles del RandomForest (default: 100)")

    args = parser.parse_args()

    if args.collect:
        if not args.sign:
            parser.error("--collect requiere --sign <nombre_de_la_seña>")
        collect(args.sign, n_samples=args.samples)

    elif args.collect_all:
        collect_all(n_samples=args.samples)

    elif args.train:
        train(n_estimators=args.estimators)