"""
logic/trainer.py

Tiene dos modos de uso:
  1. Recolección de datos:  python -m logic.trainer --collect --sign hola
  2. Entrenamiento:         python -m logic.trainer --train

No se ejecuta durante el uso normal de la app, pero RecordingDialog usa sus rutas por defecto.
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
# Rutas por defecto (para uso desde CLI en dev)
# ------------------------------------------------------------------

def get_default_paths():
    """ Devuelve las rutas por defecto basadas en la ubicación de este archivo. """
    root = Path(__file__).resolve().parent.parent.parent
    return {
        "root": root,
        "data_dir": root / "data" / "landmarks",
        "model_dir": root / "model",
        "model_path": root / "model" / "lsc_classifier.pkl",
        "encoder_path": root / "model" / "label_encoder.pkl"
    }

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

def collect(sign: str, n_samples: int = 80, countdown: int = 3, data_dir: Path = None):
    """
    Graba `n_samples` frames de landmarks para la seña indicada
    y los guarda como filas en data/landmarks/{sign}.csv.
    """
    if data_dir is None:
        data_dir = get_default_paths()["data_dir"]
    
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / f"{sign}.csv"
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


def collect_all(n_samples: int = 80, pause_between: int = 2, data_dir: Path = None):
    """ Recorre todas las señas del DEFAULT_SIGNS y llama a collect() para cada una. """
    print(f"\nSesión completa: {len(DEFAULT_SIGNS)} señas × {n_samples} muestras cada una.")
    for sign in DEFAULT_SIGNS:
        collect(sign, n_samples=n_samples, data_dir=data_dir)
        time.sleep(pause_between)
    print("\nSesión completada.")


def collect_from_video(video_path: str, sign: str, sample_interval: int = 3, data_dir: Path = None):
    """ Extrae landmarks de un video pregrabado y los guarda en data/landmarks/{sign}.csv. """
    if data_dir is None:
        data_dir = get_default_paths()["data_dir"]

    video_path = Path(video_path)
    if not video_path.exists():
        print(f"[Error] No se encontró el archivo: {video_path}")
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / f"{sign}.csv"
    existing = _count_rows(csv_path)
    print(f"\n── Procesando video para '{sign}' ──")
    print(f"   Archivo: {video_path.name}")
    print(f"   Ya existen {existing} muestras. Se agregarán las nuevas.")

    import mediapipe as mp
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
    print(f"   Muestras extraídas: {collected}  |  Total acumulado: {_count_rows(csv_path)}")


def collect_from_video_dir(video_dir: str, sample_interval: int = 3, data_dir: Path = None):
    """ Procesa todos los videos en una carpeta. """
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
        sign = video.stem
        collect_from_video(str(video), sign, sample_interval=sample_interval, data_dir=data_dir)

    print("\nProcesamiento completado.")


def train(test_size: float = 0.2, n_estimators: int = 100, data_dir: Path = None, model_dir: Path = None):
    """ Entrena el modelo usando los CSVs en data_dir. """
    paths = get_default_paths()
    if data_dir is None: data_dir = paths["data_dir"]
    if model_dir is None: model_dir = paths["model_dir"]

    X, y = _load_dataset(data_dir)

    if len(X) == 0:
        print(f"No se encontraron datos en {data_dir}. Graba muestras primero.")
        return

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"\nAccuracy en test: {(y_pred == y_test).mean() * 100:.1f}%\n")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_dir / "lsc_classifier.pkl")
    joblib.dump(encoder, model_dir / "label_encoder.pkl")
    print(f"Modelo guardado en: {model_dir}")


def _load_dataset(data_dir: Path):
    """Lee todos los CSVs de data_dir y devuelve (X, y)."""
    X, y = [], []
    csv_files = list(data_dir.glob("*.csv"))

    for csv_path in csv_files:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    features = [float(row[f"x{i}"]) for i in range(HandCapture.VECTOR_SIZE)]
                    X.append(features)
                    y.append(row["label"])
                except (KeyError, ValueError):
                    continue
    return np.array(X, dtype=np.float32), np.array(y)


def _count_rows(csv_path: Path) -> int:
    if not csv_path.exists(): return 0
    with open(csv_path, newline="") as f:
        return max(0, sum(1 for _ in f) - 1)


def _overlay(frame, title: str, subtitle: str, color: tuple, progress: float | None = None):
    if frame is None: return
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 70), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, title, (16, h - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(frame, subtitle, (16, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    if progress is not None:
        cv2.rectangle(frame, (16, h - 10), (w - 16, h - 4), (60, 60, 60), -1)
        cv2.rectangle(frame, (16, h - 10), (16 + int((w - 32) * progress), h - 4), color, -1)


def _extract_landmarks_raw(results, frame_shape: tuple) -> list[float]:
    # (Simplified version of HandCapture._extract_landmarks for internal use)
    # Re-using logic from HandCapture but standalone
    h, w = frame_shape[:2]
    pose_landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None
    if not pose_landmarks: return []
    
    sl = pose_landmarks[HandCapture.SHOULDER_LEFT]
    sr = pose_landmarks[HandCapture.SHOULDER_RIGHT]
    scale = ((sr.x - sl.x) ** 2 + (sr.y - sl.y) ** 2) ** 0.5 or 1.0
    cx, cy = (sl.x + sr.x) / 2, (sl.y + sr.y) / 2

    res = []
    for landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
        if landmarks:
            for lm in landmarks.landmark:
                res.extend([(lm.x - cx) / scale, (lm.y - cy) / scale, lm.z / scale])
        else:
            res.extend([0.0] * 63)
    
    for lm in pose_landmarks:
        res.extend([(lm.x - cx) / scale, (lm.y - cy) / scale, lm.z / scale])
    return res


if __name__ == "__main__":
    paths = get_default_paths()
    parser = argparse.ArgumentParser(description="LSC Trainer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--collect", action="store_true")
    group.add_argument("--collect-all", action="store_true")
    group.add_argument("--from-video", action="store_true")
    group.add_argument("--from-videos", action="store_true")
    group.add_argument("--train", action="store_true")

    parser.add_argument("--sign", type=str)
    parser.add_argument("--video", type=str)
    parser.add_argument("--video-dir", type=str, default=str(paths["data_dir"].parent / "raw"))
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--interval", type=int, default=3)
    parser.add_argument("--estimators", type=int, default=100)

    args = parser.parse_args()

    if args.collect:
        collect(args.sign, n_samples=args.samples)
    elif args.collect_all:
        collect_all(n_samples=args.samples)
    elif args.from_video:
        collect_from_video(args.video, args.sign, sample_interval=args.interval)
    elif args.from_videos:
        collect_from_video_dir(args.video_dir, sample_interval=args.interval)
    elif args.train:
        train(n_estimators=args.estimators)
