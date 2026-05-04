"""
view/recording_dialog.py

Diálogo modal para grabar nuevas señas y agregarlas al dataset de entrenamiento.
Permite al usuario nombrar la seña y grabar múltiples muestras.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import csv
import time
from pathlib import Path
from datetime import datetime

from App.logic.capture import HandCapture
from App.logic.trainer import get_default_paths


class RecordingDialog:
    """
    Diálogo modal para grabar nuevas señas.
    
    Uso:
        dialog = RecordingDialog(parent, on_save=callback)
        dialog.start()  # abre el diálogo modal
    """
    
    DIALOG_TITLE = "Grabar Nueva Seña"
    DEFAULT_SAMPLES = 50
    COUNTDOWN_SECONDS = 3
    
    def __init__(self, parent, on_save=None, data_dir: Path | str = None):
        """
        Args:
            parent: ventana padre (tk.Tk o tk.Toplevel)
            on_save: callback llamado cuando se guardan las muestras exitosamente
            data_dir: ruta donde se guardarán los CSVs.
        """
        self._parent = parent
        self._on_save = on_save
        
        if data_dir is None:
            data_dir = get_default_paths()["data_dir"]
        self._data_dir = Path(data_dir)
        
        self._running = False
        self._recording = False
        self._collected = 0
        self._cap = None
        
        self._dialog = tk.Toplevel(parent)
        self._dialog.title(self.DIALOG_TITLE)
        self._dialog.transient(parent)  # mantener encima del padre
        self._dialog.grab_set()  # modal
        self._dialog.resizable(False, False)
        self._dialog.configure(bg="#1e1e1e")
        
        self._build()
        self._center_dialog()
        
    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------
    
    def start(self):
        """Inicia el loop del diálogo. Retorna cuando se cierra."""
        self._dialog.wait_window()
        
    # ------------------------------------------------------------------
    # Construcción de la interfaz
    # ------------------------------------------------------------------
    
    def _build(self):
        """Construye el layout del diálogo."""
        
        # Contenedor principal
        main = tk.Frame(self._dialog, bg="#1e1e1e", padx=20, pady=20)
        main.pack(fill="both", expand=True)
        
        # ── Campo para nombre de la seña ────────────────────────────────
        name_frame = tk.Frame(main, bg="#1e1e1e")
        name_frame.pack(fill="x", pady=(0, 15))
        
        tk.Label(
            name_frame,
            text="Nombre de la seña:",
            bg="#1e1e1e",
            fg="#ffffff",
            font=("Arial", 11),
        ).pack(anchor="w")
        
        self._name_entry = tk.Entry(
            name_frame,
            font=("Arial", 12),
            bg="#2d2d2d",
            fg="#ffffff",
            insertbackground="white",
            relief="flat",
            highlightthickness=1,
            highlightbackground="#404040",
            highlightcolor="#60a5fa",
        )
        self._name_entry.pack(fill="x", pady=(5, 0), ipady=5)
        self._name_entry.focus_set()
        
        # ── Campo para número de muestras ───────────────────────────────
        samples_frame = tk.Frame(main, bg="#1e1e1e")
        samples_frame.pack(fill="x", pady=(0, 15))
        
        tk.Label(
            samples_frame,
            text="Número de muestras:",
            bg="#1e1e1e",
            fg="#ffffff",
            font=("Arial", 11),
        ).pack(anchor="w")
        
        self._samples_spinbox = tk.Spinbox(
            samples_frame,
            from_=10,
            to=200,
            width=10,
            font=("Arial", 12),
            bg="#2d2d2d",
            fg="#ffffff",
            buttonbackground="#404040",
            buttonactivebackground="#60a5fa",
            relief="flat",
            highlightthickness=1,
            highlightbackground="#404040",
            highlightcolor="#60a5fa",
        )
        self._samples_spinbox.delete(0, "end")
        self._samples_spinbox.insert(0, str(self.DEFAULT_SAMPLES))
        self._samples_spinbox.pack(anchor="w", pady=(5, 0))
        
        # ── Botones de control ─────────────────────────────────────────
        btn_frame = tk.Frame(main, bg="#1e1e1e")
        btn_frame.pack(fill="x", pady=20)
        
        self._start_btn = tk.Button(
            btn_frame,
            text="▶ Iniciar Grabación",
            command=self._toggle_recording,
            bg="#22c55e",
            fg="white",
            font=("Arial", 11, "bold"),
            relief="flat",
            padx=15,
            pady=8,
            cursor="hand2",
        )
        self._start_btn.pack(side="left", padx=(0, 10))
        
        self._cancel_btn = tk.Button(
            btn_frame,
            text="✕ Cancelar",
            command=self._close,
            bg="#ef4444",
            fg="white",
            font=("Arial", 11),
            relief="flat",
            padx=15,
            pady=8,
            cursor="hand2",
        )
        self._cancel_btn.pack(side="right")
        
        # ── Estado y progreso ──────────────────────────────────────────
        self._status_label = tk.Label(
            main,
            text="Ingresa el nombre de la seña y presiona 'Iniciar Grabación'",
            bg="#1e1e1e",
            fg="#9ca3af",
            font=("Arial", 10),
            wraplength=350,
        )
        self._status_label.pack(pady=(10, 5))
        
        # Barra de progreso
        self._progress_var = tk.DoubleVar(value=0.0)
        self._progress_bar = ttk.Progressbar(
            main,
            variable=self._progress_var,
            maximum=100,
            mode="determinate",
            length=350,
        )
        self._progress_bar.pack(pady=(0, 10))
        
        # Contador de muestras
        self._count_label = tk.Label(
            main,
            text="Muestras: 0 / 0",
            bg="#1e1e1e",
            fg="#60a5fa",
            font=("Arial", 11, "bold"),
        )
        self._count_label.pack()
        
    def _center_dialog(self):
        """Centra el diálogo sobre la ventana padre."""
        self._dialog.update_idletasks()
        
        parent_x = self._parent.winfo_x()
        parent_y = self._parent.winfo_y()
        parent_w = self._parent.winfo_width()
        parent_h = self._parent.winfo_height()
        
        dialog_w = 400
        dialog_h = 380
        
        x = parent_x + (parent_w - dialog_w) // 2
        y = parent_y + (parent_h - dialog_h) // 2
        
        self._dialog.geometry(f"{dialog_w}x{dialog_h}+{x}+{y}")
        
    # ------------------------------------------------------------------
    # Lógica de grabación
    # ------------------------------------------------------------------
    
    def _toggle_recording(self):
        """Inicia o detiene la grabación según el estado actual."""
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()
            
    def _start_recording(self):
        """Inicia la sesión de grabación."""
        sign_name = self._name_entry.get().strip().lower().replace(" ", "_")
        
        if not sign_name:
            messagebox.showerror(
                "Error",
                "Por favor ingresa un nombre para la seña.",
                parent=self._dialog
            )
            return
            
        try:
            n_samples = int(self._samples_spinbox.get())
            if n_samples < 10:
                raise ValueError("Mínimo 10 muestras")
        except ValueError as e:
            messagebox.showerror(
                "Error",
                f"Número de muestras inválido: {e}",
                parent=self._dialog
            )
            return
        
        # Verificar si ya existe la seña
        csv_path = self._data_dir / f"{sign_name}.csv"
        existing_count = 0
        if csv_path.exists():
            with open(csv_path, 'r') as f:
                existing_count = sum(1 for _ in f) - 1  # Restar header
            
            confirm = messagebox.askyesno(
                "Seña existente",
                f"La seña '{sign_name}' ya tiene {existing_count} muestras guardadas.\n\n"
                f"¿Deseas agregar {n_samples} muestras adicionales?\n\n"
                f"Esto mejorará el reconocimiento de esta seña.",
                parent=self._dialog
            )
            if not confirm:
                return
        
        self._recording = True
        self._collected = 0
        self._total_samples = n_samples
        self._sign_name = sign_name
        self._existing_count = existing_count
        
        # Actualizar UI
        self._start_btn.config(text="⏹ Detener", bg="#f59e0b")
        self._name_entry.config(state="disabled")
        self._samples_spinbox.config(state="disabled")
        self._cancel_btn.config(state="disabled")
        
        # Iniciar captura
        try:
            self._cap = HandCapture()
            self._run_countdown()
        except RuntimeError as e:
            messagebox.showerror("Error de cámara", str(e), parent=self._dialog)
            self._reset_ui()
            
    def _run_countdown(self):
        """Ejecuta la cuenta regresiva antes de empezar a grabar."""
        remaining = self.COUNTDOWN_SECONDS
        
        def countdown_step():
            if remaining > 0:
                self._status_label.config(
                    text=f"Preparado... inicia en {remaining}s\n¡Mantén la seña visible!"
                )
                self._dialog.after(1000, lambda: countdown_step())
                return remaining - 1
            else:
                self._status_label.config(
                    text="¡Grabando! Mantén la posición."
                )
                self._start_capture_loop()
                
        countdown_step()
        
    def _start_capture_loop(self):
        """Loop de captura de muestras."""
        if not self._recording or self._collected >= self._total_samples:
            self._finish_recording()
            return
            
        if not self._cap.read():
            messagebox.showerror("Error", "Falló la lectura de cámara", parent=self._dialog)
            self._finish_recording()
            return
            
        landmarks = self._cap.get_landmarks()
        
        if landmarks:
            self._save_sample(landmarks)
            self._collected += 1
            
            # Actualizar UI
            progress = (self._collected / self._total_samples) * 100
            self._progress_var.set(progress)
            self._count_label.config(
                text=f"Muestras: {self._collected} / {self._total_samples}"
            )
            
        # Continuar loop
        self._dialog.after(50, self._start_capture_loop)
        
    def _save_sample(self, landmarks: list[float]):
        """Guarda una muestra en el CSV."""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self._data_dir / f"{self._sign_name}.csv"
        file_exists = csv_path.exists()
        
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                # Escribir encabezado
                header = [f"x{i}" for i in range(HandCapture.VECTOR_SIZE)] + ["label"]
                writer.writerow(header)
                
            writer.writerow(landmarks + [self._sign_name])
            
    def _stop_recording(self):
        """Detiene la grabación manualmente."""
        self._recording = False
        
    def _finish_recording(self):
        """Finaliza la sesión de grabación."""
        if self._cap:
            self._cap.release()
            self._cap = None
            
        if self._collected > 0:
            total_samples = self._existing_count + self._collected
            messagebox.showinfo(
                "Éxito",
                f"Se guardaron {self._collected} muestras adicionales para '{self._sign_name}'.\n\n"
                f"Total acumulado: {total_samples} muestras.\n\n"
                f"{'(Datos existentes: ' + str(self._existing_count) + ')' if self._existing_count > 0 else ''}",
                parent=self._dialog
            )
            if callable(self._on_save):
                self._on_save(self._sign_name, self._collected)
        elif self._collected == 0 and self._recording:
            messagebox.showwarning(
                "Advertencia",
                "No se detectaron manos durante la grabación.\n"
                "Asegúrate de mostrar ambas manos claramente.",
                parent=self._dialog
            )
            
        self._reset_ui()
        
    def _count_existing(self) -> int:
        """Cuenta muestras existentes para la seña actual."""
        csv_path = self._data_dir / f"{self._sign_name}.csv"
        if not csv_path.exists():
            return 0
        with open(csv_path, newline="") as f:
            return max(0, sum(1 for _ in f) - 1)
            
    def _reset_ui(self):
        """Restablece la UI al estado inicial."""
        self._recording = False
        self._start_btn.config(text="▶ Iniciar Grabación", bg="#22c55e")
        self._name_entry.config(state="normal")
        self._samples_spinbox.config(state="normal")
        self._cancel_btn.config(state="normal")
        self._progress_var.set(0)
        self._count_label.config(text="Muestras: 0 / 0")
        self._status_label.config(
            text="Ingresa el nombre de la seña y presiona 'Iniciar Grabación'"
        )
        
    def _close(self):
        """Cierra el diálogo."""
        if self._recording:
            confirm = messagebox.askyesno(
                "Confirmar",
                "La grabación está en curso.\n¿Seguro que deseas cancelar?",
                parent=self._dialog
            )
            if not confirm:
                return
                
        if self._cap:
            self._cap.release()
            self._cap = None
            
        self._dialog.destroy()
