"""
batch_record_alphabet.py

Script para grabar masivamente todas las letras del alfabeto LSC.
Grabará 30 muestras por letra automáticamente.

Uso:
    python batch_record_alphabet.py
    
El script mostrará una cuenta regresiva para cada letra y capturará
las muestras automáticamente cuando detecte manos.
"""

import csv
import time
from pathlib import Path
from datetime import datetime

from App.logic.capture import HandCapture


class AlphabetRecorder:
    """
    Grabadora masiva del alfabeto LSC.
    
    Recorre cada letra del alfabeto y captura muestras automáticamente
    cuando detecta una mano en la cámara.
    """
    
    # Alfabeto LSC (incluye Ñ para español)
    ALPHABET = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
        'n', 'ñ', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
    ]
    
    SAMPLES_PER_LETTER = 30
    COUNTDOWN_SECONDS = 3
    DELAY_BETWEEN_SAMPLES = 0.5  # segundos entre capturas
    
    def __init__(self, data_dir: str = "data/landmarks"):
        """
        Args:
            data_dir: directorio donde se guardarán los CSVs.
        """
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._cap = None
        
    def run(self):
        """Ejecuta la grabación masiva de todo el alfabeto."""
        print("=" * 60)
        print("GRABADORA MASIVA DE ALFABETO LSC")
        print("=" * 60)
        print(f"\nSe grabarán {self.SAMPLES_PER_LETTER} muestras por letra.")
        print(f"Total de letras: {len(self.ALPHABET)}")
        print(f"Total estimado de muestras: {len(self.ALPHABET) * self.SAMPLES_PER_LETTER}")
        print("\nInstrucciones:")
        print("1. Posiciona tu mano frente a la cámara")
        print("2. Haz la seña de la letra que se indica")
        print("3. Mantén la posición durante la captura")
        print("4. El script avanzará automáticamente a la siguiente letra")
        print("\nPresiona Ctrl+C en cualquier momento para cancelar.\n")
        
        try:
            input("Presiona ENTER para comenzar...")
        except KeyboardInterrupt:
            print("\n\nCancelado por el usuario.")
            return
            
        self._cap = HandCapture()
        
        try:
            for letter in self.ALPHABET:
                self._record_letter(letter)
        except KeyboardInterrupt:
            print("\n\nGrabación interrumpida.")
        finally:
            if self._cap:
                self._cap.release()
                
        print("\n" + "=" * 60)
        print("GRABACIÓN FINALIZADA")
        print("=" * 60)
        self._print_summary()
        
    def _record_letter(self, letter: str):
        """Graba muestras para una letra específica."""
        print("\n" + "-" * 60)
        print(f"LETRA: '{letter.upper()}'")
        print("-" * 60)
        
        # Verificar si ya existen muestras para esta letra
        csv_path = self._data_dir / f"{letter}.csv"
        existing_count = 0
        if csv_path.exists():
            with open(csv_path, 'r') as f:
                existing_count = sum(1 for _ in f) - 1  # Restar header
            print(f"⚠️  Ya existen {existing_count} muestras guardadas para '{letter}'.")
            print(f"   Se agregarán {self.SAMPLES_PER_LETTER} muestras adicionales.")
        
        collected = 0
        attempts = 0
        max_attempts = self.SAMPLES_PER_LETTER * 3  # Intentos máximos para evitar bucles infinitos
        
        # Cuenta regresiva
        print(f"\nPreparado... muestra la seña de '{letter.upper()}'")
        for i in range(self.COUNTDOWN_SECONDS, 0, -1):
            print(f"  {i}...")
            time.sleep(1)
        
        print("\n¡GRABANDO! Mantén la posición.\n")
        
        while collected < self.SAMPLES_PER_LETTER and attempts < max_attempts:
            attempts += 1
            
            if not self._cap.read():
                print("Error: No se pudo leer de la cámara.")
                break
                
            landmarks = self._cap.get_landmarks()
            
            if landmarks:
                self._save_sample(letter, landmarks)
                collected += 1
                print(f"  Muestra {collected}/{self.SAMPLES_PER_LETTER} ✓")
                
                # Pequeño delay entre capturas
                if collected < self.SAMPLES_PER_LETTER:
                    time.sleep(self.DELAY_BETWEEN_SAMPLES)
            else:
                print(f"  Intento {attempts}: No se detectó mano. Asegura buena iluminación.", end='\r')
                time.sleep(0.3)
        
        if collected > 0:
            total = existing_count + collected
            print(f"\n✓ Completado: {collected} muestras nuevas para '{letter}' (Total: {total})")
        else:
            print(f"\n✗ No se pudieron capturar muestras para '{letter}'")
            print("   Verifica iluminación y posición de la mano.")
            
    def _save_sample(self, letter: str, landmarks: list[float]):
        """Guarda una muestra en el CSV correspondiente."""
        csv_path = self._data_dir / f"{letter}.csv"
        file_exists = csv_path.exists()
        
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                # Escribir encabezado
                header = [f"x{i}" for i in range(HandCapture.VECTOR_SIZE)] + ["label"]
                writer.writerow(header)
                
            writer.writerow(landmarks + [letter])
            
    def _print_summary(self):
        """Imprime un resumen de las grabaciones."""
        print("\nResumen de archivos creados/actualizados:")
        print("-" * 40)
        
        total_samples = 0
        for letter in self.ALPHABET:
            csv_path = self._data_dir / f"{letter}.csv"
            if csv_path.exists():
                with open(csv_path, 'r') as f:
                    count = sum(1 for _ in f) - 1  # Restar header
                total_samples += count
                print(f"  {letter.upper():3s}: {count:3d} muestras")
            else:
                print(f"  {letter.upper():3s}: SIN DATOS")
                
        print("-" * 40)
        print(f"TOTAL: {total_samples} muestras")


def main():
    """Punto de entrada del script."""
    recorder = AlphabetRecorder()
    recorder.run()


if __name__ == "__main__":
    main()
