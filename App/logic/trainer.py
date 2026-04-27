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
DATA_DIR = ROOT / "data" / "landmarks"
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

def collect_from_video(video_path: str, sign: str, sample_interval: int = 3):
    """
    Extrae landmarks de un video pregrabado y los guarda en data/landmarks/{sign}.csv.
    Equivalente a collect(), pero usa un archivo de video en lugar de la webcam.

    Args:
        video_path:      ruta al archivo de video (mp4, mov, avi, etc.).
        sign:            nombre de la seña (debe coincidir con el nombre del CSV destino).
        sample_interval: tomar una muestra cada N frames para evitar redundancia.
                         Con 30fps y sample_interval=3 se obtienen ~10 muestras/segundo.
    """
    from App.logic.capture import HandCapture
    import mediapipe as mp

    video_path = Path(video_path)
    if not video_path.exists():
        print(f"[Error] No se encontró el archivo: {video_path}")
        return

    csv_path = DATA_DIR / f"{sign}.csv"
    existing = _count_rows(csv_path)
    print(f"\n── Procesando video para '{sign}' ──")
    print(f"   Archivo: {video_path.name}")
    print(f"   Ya existen {existing} muestras. Se agregarán las nuevas.")

    # Reusar el pipeline de MediaPipe Holistic directamente sobre el video
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Error] No se pudo abrir el video: {video_path}")
        holistic.close()
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    collected = 0
    frame_idx = 0
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = [f"x{i}" for i in range(HandCapture.VECTOR_SIZE)] + ["label"]
            writer.writerow(header)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Tomar muestra cada sample_interval frames
            if frame_idx % sample_interval == 0:
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)

                if results.pose_landmarks and (results.left_hand_landmarks or results.right_hand_landmarks):
                    landmarks = _extract_landmarks_raw(results, frame.shape)
                    writer.writerow(landmarks + [sign])
                    collected += 1

            frame_idx += 1

    cap.release()
    holistic.close()

    total = _count_rows(csv_path)
    print(f"   Muestras extraídas: {collected}  |  Total acumulado: {total}")


def collect_from_video_dir(video_dir: str, sample_interval: int = 3):
    """
    Procesa todos los videos en una carpeta. Infiere el nombre de la seña
    a partir del nombre del archivo (sin extensión).

    Estructura esperada:
        data/raw/
            hola.mp4
            gracias.mp4
            ...

    Uso:
        python -m logic.trainer --from-videos
        python -m logic.trainer --from-videos --video-dir ruta/a/carpeta

    Args:
        video_dir:       ruta a la carpeta con los videos.
        sample_interval: tomar una muestra cada N frames.
    """
    video_dir = Path(video_dir)
    if not video_dir.exists():
        print(f"[Error] No se encontró la carpeta: {video_dir}")
        return

    extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    videos = [f for f in video_dir.iterdir() if f.suffix.lower() in extensions]

    if not videos:
        print(f"No se encontraron videos en: {video_dir}")
        return

    print(f"\nProcesando {len(videos)} video(s) en '{video_dir}'...")
    for video in sorted(videos):
        sign = video.stem   # nombre del archivo sin extensión
        collect_from_video(str(video), sign, sample_interval=sample_interval)

    print("\nProcesamiento completado.")


def _extract_landmarks_raw(results, frame_shape: tuple) -> list[float]:
    """
    Versión standalone de HandCapture._extract_landmarks() para usar
    sin instanciar HandCapture (evita abrir la webcam al procesar videos).
    Usa normalización basada en el torso (hombros) para consistencia con
    la captura en tiempo real.
    
    Args:
        results: objeto MediaPipe HolisticResults con todos los landmarks.
        frame_shape: (height, width, channels) del frame actual.
    
    Returns:
        Lista de 603 floats normalizados.
    """
    from App.logic.capture import HandCapture
    
    h, w = frame_shape[:2]
    normalized = []

    # Obtener puntos de referencia del torso para normalización
    pose_landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None
    if not pose_landmarks:
        return []
    
    # Calcular escala basada en la distancia entre hombros
    shoulder_left = pose_landmarks[HandCapture.SHOULDER_LEFT]
    shoulder_right = pose_landmarks[HandCapture.SHOULDER_RIGHT]
    shoulder_distance = ((shoulder_right.x - shoulder_left.x) ** 2 + 
                        (shoulder_right.y - shoulder_left.y) ** 2) ** 0.5
    scale = shoulder_distance if shoulder_distance > 0 else 1.0
    
    # Centro del torso (punto medio entre hombros)
    torso_center_x = (shoulder_left.x + shoulder_right.x) / 2
    torso_center_y = (shoulder_left.y + shoulder_right.y) / 2

    # Extraer landmarks de mano izquierda (si existe)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            nx = (lm.x - torso_center_x) / scale
            ny = (lm.y - torso_center_y) / scale
            nz = lm.z / scale
            normalized.extend([nx, ny, nz])
    else:
        normalized.extend([0.0] * (HandCapture.NUM_HAND_LANDMARKS * 3))

    # Extraer landmarks de mano derecha (si existe)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            nx = (lm.x - torso_center_x) / scale
            ny = (lm.y - torso_center_y) / scale
            nz = lm.z / scale
            normalized.extend([nx, ny, nz])
    else:
        normalized.extend([0.0] * (HandCapture.NUM_HAND_LANDMARKS * 3))

    # Extraer landmarks de pose completos
    for lm in pose_landmarks:
        nx = (lm.x - torso_center_x) / scale
        ny = (lm.y - torso_center_y) / scale
        nz = lm.z / scale
        normalized.extend([nx, ny, nz])

    return normalized

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
    group.add_argument("--collect",      action="store_true", help="Modo recolección en vivo")
    group.add_argument("--collect-all",  action="store_true", help="Recolectar todas las señas en vivo")
    group.add_argument("--from-video",   action="store_true", help="Procesar un video individual")
    group.add_argument("--from-videos",  action="store_true", help="Procesar todos los videos de una carpeta")
    group.add_argument("--train",        action="store_true", help="Entrenar el modelo")

    parser.add_argument("--sign",       type=str,            help="Nombre de la seña (con --collect o --from-video)")
    parser.add_argument("--video",      type=str,            help="Ruta al video (con --from-video)")
    parser.add_argument("--video-dir",  type=str,            default=str(ROOT / "data" / "raw"),
                                                             help="Carpeta de videos (con --from-videos, default: data/raw/)")
    parser.add_argument("--samples",    type=int, default=80,  help="Muestras por seña en vivo (default: 80)")
    parser.add_argument("--interval",   type=int, default=3,   help="Frames entre muestras al procesar video (default: 3)")
    parser.add_argument("--estimators", type=int, default=100, help="Árboles del RandomForest (default: 100)")

    args = parser.parse_args()

    if args.collect:
        if not args.sign:
            parser.error("--collect requiere --sign <nombre_de_la_seña>")
        collect(args.sign, n_samples=args.samples)

    elif args.collect_all:
        collect_all(n_samples=args.samples)

    elif args.from_video:
        if not args.sign or not args.video:
            parser.error("--from-video requiere --sign <seña> y --video <ruta>")
        collect_from_video(args.video, args.sign, sample_interval=args.interval)

    elif args.from_videos:
        collect_from_video_dir(args.video_dir, sample_interval=args.interval)

    elif args.train:
        train(n_estimators=args.estimators)