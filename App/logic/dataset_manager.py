"""
Gestor de conjuntos de datos para LSC Traductor.
Permite listar, verificar y limpiar datos de entrenamiento para evitar conflictos.
"""

import os
import csv
import glob
from pathlib import Path

DATA_DIR = Path("data/landmarks")

class DatasetManager:
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def list_signs(self):
        """Devuelve una lista de todas las señas disponibles y su conteo de muestras."""
        if not DATA_DIR.exists():
            return {}
        
        signs = {}
        for file in DATA_DIR.glob("*.csv"):
            name = file.stem
            with open(file, 'r') as f:
                count = sum(1 for _ in f) - 1  # Restar header
            signs[name] = max(0, count)
        return signs

    def get_sign_data(self, sign_name):
        """Obtiene todas las filas de una seña específica."""
        file_path = DATA_DIR / f"{sign_name}.csv"
        if not file_path.exists():
            return []
        
        data = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Saltar header
            for row in reader:
                if row:  # Ignorar líneas vacías
                    data.append([float(x) for x in row])
        return data

    def append_samples(self, sign_name, samples):
        """Añade muestras a una seña existente o crea una nueva."""
        file_path = DATA_DIR / f"{sign_name}.csv"
        is_new = not file_path.exists()
        
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if is_new:
                # Escribir header si es nuevo (603 columnas para Holistic)
                header = [f"lm_{i}" for i in range(603)]
                writer.writerow(header)
            
            for sample in samples:
                writer.writerow(sample)
        
        return is_new

    def clear_sign(self, sign_name):
        """Elimina todos los datos de una seña específica."""
        file_path = DATA_DIR / f"{sign_name}.csv"
        if file_path.exists():
            os.remove(file_path)
            return True
        return False

    def merge_signs(self, source_name, target_name):
        """Fusiona los datos de una seña en otra y borra la original."""
        source_data = self.get_sign_data(source_name)
        if not source_data:
            return False
        
        self.append_samples(target_name, source_data)
        self.clear_sign(source_name)
        return True

    def get_statistics(self):
        """Devuelve estadísticas generales del dataset."""
        signs = self.list_signs()
        total_samples = sum(signs.values())
        return {
            "total_signs": len(signs),
            "total_samples": total_samples,
            "average_samples_per_sign": total_samples / len(signs) if signs else 0,
            "signs": signs
        }

if __name__ == "__main__":
    # CLI simple para gestión
    import sys
    manager = DatasetManager()
    
    if len(sys.argv) < 2:
        print("Uso: python -m App.logic.dataset_manager [list|clear|stats]")
        print("  list                 : Listar señas y conteo")
        print("  clear <nombre>       : Borrar todos los datos de una seña")
        print("  stats                : Estadísticas del dataset")
        sys.exit(0)

    cmd = sys.argv[1]
    
    if cmd == "list":
        signs = manager.list_signs()
        if not signs:
            print("No hay datos de entrenamiento encontrados.")
        else:
            print(f"{'Seña':<20} | {'Muestras':<10}")
            print("-" * 32)
            for name, count in sorted(signs.items()):
                print(f"{name:<20} | {count:<10}")
    
    elif cmd == "stats":
        stats = manager.get_statistics()
        print(f"Total Señas: {stats['total_signs']}")
        print(f"Total Muestras: {stats['total_samples']}")
        print(f"Promedio por seña: {stats['average_samples_per_sign']:.2f}")

    elif cmd == "clear":
        if len(sys.argv) < 3:
            print("Error: Debe especificar el nombre de la seña.")
            sys.exit(1)
        sign_name = sys.argv[2]
        confirm = input(f"¿Estás seguro de BORRAR todos los datos de '{sign_name}'? (y/n): ")
        if confirm.lower() == 'y':
            if manager.clear_sign(sign_name):
                print(f"Datos de '{sign_name}' eliminados correctamente.")
            else:
                print(f"No se encontró la seña '{sign_name}'.")
        else:
            print("Operación cancelada.")
    else:
        print(f"Comando desconocido: {cmd}")
