import PyInstaller.__main__
import os
import shutil
import sys

# Nombre de la aplicación
APP_NAME = "LSC_Traductor"
MAIN_SCRIPT = "main.py"

# Directorios críticos que deben incluirse
DATA_DIRS = ["model", "data"]

def build():
    print(f"🔨 Compilando {APP_NAME}...")
    
    # Argumentos base para PyInstaller
    args = [
        MAIN_SCRIPT,
        "--name", APP_NAME,
        "--onefile",          # Un solo ejecutable (puedes quitarlo si prefieres carpeta)
        "--noconsole",        # Ocultar terminal (quitar para debug)
        "--clean",            # Limpiar caché anterior
        "--hidden-import", "mediapipe.python.solutions.holistic",
        "--hidden-import", "mediapipe.python.solutions.hands",
        "--hidden-import", "cv2",
        "--hidden-import", "numpy",
        "--hidden-import", "sklearn",
        "--hidden-import", "tensorflow",
    ]

    # Agregar recursivamente las carpetas de datos al paquete
    # Nota: En modo --onefile, esto extrae los archivos a una carpeta temporal al iniciar
    for directory in DATA_DIRS:
        if os.path.exists(directory):
            print(f"✅ Incluyendo carpeta: {directory}/")
            args.append(f"--add-data={directory}{os.pathsep}{directory}")
        else:
            print(f"⚠️  Advertencia: La carpeta {directory}/ no existe. El ejecutable podría fallar si la necesita al inicio.")

    # Ejecutar PyInstaller
    try:
        PyInstaller.__main__.run(args)
        print("\n✅ ¡Compilación exitosa!")
        print(f"📦 El ejecutable se encuentra en: dist/{APP_NAME}")
        
        if sys.platform.startswith('linux'):
            print("\n💡 Nota para Linux: Si usas --onefile, asegúrate de que los permisos de ejecución estén correctos.")
            print(f"   chmod +x dist/{APP_NAME}/{APP_NAME}")
            
    except Exception as e:
        print(f"\n❌ Error durante la compilación: {e}")

if __name__ == "__main__":
    build()