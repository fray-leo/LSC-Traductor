"""
main.py

Punto de entrada de LSC Traductor.
Instancia las capas de logic/ y view/ y las conecta mediante un loop
que corre a ~30fps usando root.after() de tkinter.

Uso:
    python main.py
"""

from App.logic.capture import HandCapture
from App.logic.classifier import SignClassifier
from App.logic.database import TranslationDB
from App.view.app import App
from App.view.components.recording_dialog import RecordingDialog

# ------------------------------------------------------------------
# Parámetros del loop
# ------------------------------------------------------------------

LOOP_INTERVAL_MS = 33          # ~30 fps
BUFFER_SIZE       = 10         # frames para estabilizar la predicción
BUFFER_THRESHOLD  = 0.6        # fracción mínima del buffer con la misma seña
HISTORY_REFRESH   = 15         # actualizar historial cada N frames


def main():
    # ── Inicializar capa lógica ────────────────────────────────────────
    try:
        cap = HandCapture()
    except RuntimeError as e:
        print(f"[Error] {e}")
        return

    try:
        clf = SignClassifier()
    except RuntimeError as e:
        print(f"[Error] {e}")
        cap.release()
        return

    db = TranslationDB()

    # ── Función para abrir diálogo de grabación ────────────────────────
    def open_recording_dialog():
        """Abre el diálogo modal para grabar nuevas señas."""
        # Pausar temporalmente el loop de traducción
        nonlocal _loop_paused
        _loop_paused = True
        
        def on_save(sign_name, count):
            """Callback cuando se guardan muestras exitosamente."""
            print(f"[Info] Se agregaron {count} muestras para '{sign_name}'")
            # Opcional: re-entrenar el modelo automáticamente
            # _retrain_model()
        
        dialog = RecordingDialog(app._root, on_save=on_save)
        dialog.start()
        
        # Reanudar el loop
        _loop_paused = False
        # Programar continuación del loop
        app.schedule(LOOP_INTERVAL_MS, loop)

    # ── Inicializar interfaz ───────────────────────────────────────────
    app = App(
        on_close=cap.release,
        on_clear=db.clear,
        on_record=open_recording_dialog,
    )

    # ── Estado del loop ────────────────────────────────────────────────
    buffer: list[str]   = []      # últimas N predicciones crudas
    last_saved: str     = ""      # última seña guardada en la BD
    frame_count: int    = 0       # contador de frames para tareas periódicas
    _loop_paused: bool  = False   # flag para pausar durante grabación

    # Cargar historial inicial
    app.update_history(_format_history(db.get_recent()))

    # ── Loop principal ─────────────────────────────────────────────────
    def loop():
        nonlocal last_saved, frame_count, _loop_paused

        # Si está pausado (durante grabación), no procesar frames
        if _loop_paused:
            return

        # 1. Leer frame de la cámara
        if not cap.read():
            app.schedule(LOOP_INTERVAL_MS, loop)
            return

        frame     = cap.get_frame()
        landmarks = cap.get_landmarks()

        # 2. Actualizar feed de video
        app.update_frame(frame)

        # 3. Clasificar si hay mano detectada
        if landmarks:
            raw_sign, confidence = clf.predict_with_confidence(landmarks)
        else:
            raw_sign, confidence = clf.UNCERTAIN, 0.0

        # 4. Estabilizar predicción con buffer de frames
        buffer.append(raw_sign)
        if len(buffer) > BUFFER_SIZE:
            buffer.pop(0)

        stable_sign = _stable_prediction(buffer, BUFFER_THRESHOLD)

        # 5. Actualizar panel de traducción
        app.update_translation(stable_sign, confidence)

        # 6. Guardar en BD solo si la seña cambió y es válida
        if (
            stable_sign != clf.UNCERTAIN
            and stable_sign != last_saved
        ):
            db.save(stable_sign)
            last_saved = stable_sign
            app.update_history(_format_history(db.get_recent()))

        # 7. Refrescar historial periódicamente (por si se limpió externamente)
        frame_count += 1
        if frame_count % HISTORY_REFRESH == 0:
            app.update_history(_format_history(db.get_recent()))

        # 8. Registrar siguiente iteración
        app.schedule(LOOP_INTERVAL_MS, loop)

    # Arrancar el loop y la ventana
    app.schedule(0, loop)
    app.start()


# ------------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------------

def _stable_prediction(buffer: list[str], threshold: float) -> str:
    """
    Devuelve la seña más frecuente del buffer si supera el umbral,
    o UNCERTAIN si ninguna lo hace.

    Ejemplo con BUFFER_SIZE=10, THRESHOLD=0.6:
        ["hola"×7, "..."×3] → "hola"   (7/10 = 0.70 ≥ 0.60)
        ["hola"×5, "..."×5] → "..."    (5/10 = 0.50 < 0.60)
    """
    if not buffer:
        return "..."

    counts: dict[str, int] = {}
    for s in buffer:
        counts[s] = counts.get(s, 0) + 1

    best = max(counts, key=lambda k: counts[k])
    if counts[best] / len(buffer) >= threshold:
        return best

    return "..."


def _format_history(rows: list[tuple]) -> list[tuple]:
    """
    Convierte las tuplas (id, sena, timestamp) de la BD
    en tuplas (sena, timestamp) que espera TranslationPanel.
    """
    return [(sena, timestamp) for _, sena, timestamp in rows]


# ------------------------------------------------------------------
# Entrada
# ------------------------------------------------------------------

if __name__ == "__main__":
    main()