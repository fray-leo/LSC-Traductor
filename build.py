#!/usr/bin/env python3
"""
build.py

Script para crear ejecutables de LSC Traductor.
Soporta Windows (.exe), macOS (.app) y Linux (binario).

Uso:
    python build.py              # Compila para la plataforma actual
    python build.py --clean      # Limpia archivos de build anteriores
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Verifica que PyInstaller esté instalado."""
    try:
        import PyInstaller
        print(f"✓ PyInstaller {PyInstaller.__version__} encontrado")
        return True
    except ImportError:
        print("✗ PyInstaller no está instalado")
        print("\nInstala con: pip install pyinstaller")
        return False


def clean_build():
    """Limpia archivos generados por builds anteriores."""
    dirs_to_clean = ["build", "dist", "__pycache__"]
    spec_files = list(Path(".").glob("*.spec"))
    
    print("Limpiando archivos de build...")
    
    for dir_name in dirs_to_clean:
        dir_path = Path(dir_name)
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"  ✓ Eliminado: {dir_name}/")
    
    for spec_file in spec_files:
        spec_file.unlink()
        print(f"  ✓ Eliminado: {spec_file.name}")
    
    # Limpiar __pycache__ en subdirectorios
    for pycache in Path(".").rglob("__pycache__"):
        shutil.rmtree(pycache)
        print(f"  ✓ Eliminado: {pycache}/")
    
    print("Limpieza completada.")


def get_icon_path():
    """Retorna la ruta del ícono según la plataforma."""
    icons = {
        "win32": "assets/icon.ico",
        "darwin": "assets/icon.icns",
        "linux": None,  # Linux generalmente no usa íconos embebidos
    }
    return icons.get(sys.platform)


def build_executable(onefile=False):
    """Construye el ejecutable usando PyInstaller."""
    
    # Determinar configuración según plataforma
    platform = sys.platform
    app_name = "LSC_Traductor"
    
    if platform == "win32":
        ext = ".exe"
        console_flag = "--noconsole"  # Sin ventana de consola en Windows
    elif platform == "darwin":
        ext = ".app"
        console_flag = "--windowed"
    else:  # Linux
        ext = ""
        console_flag = "--windowed"
    
    # Opciones de PyInstaller
    options = [
        sys.executable, "-m", "PyInstaller",
        "--name", app_name,
        "--onefile" if onefile else "--onedir",
        console_flag,
        "--add-data", f"{Path('App')}:{Path('App')}",  # Incluir paquete App
        "--hidden-import", "mediapipe",
        "--hidden-import", "sklearn",
        "--hidden-import", "cv2",
        "--collect-all", "mediapipe",  # Incluir todos los recursos de MediaPipe
        "--clean",
        "main.py",
    ]
    
    # Agregar ícono si existe
    icon = get_icon_path()
    if icon and Path(icon).exists():
        options.extend(["--icon", icon])
    
    print(f"\n{'='*60}")
    print(f"Construyendo LSC Traductor para {platform.upper()}")
    print(f"{'='*60}\n")
    
    print("Comando:", " ".join(options))
    print()
    
    # Ejecutar PyInstaller
    result = subprocess.run(options)
    
    if result.returncode == 0:
        output_dir = Path("dist")
        if onefile:
            exe_path = output_dir / f"{app_name}{ext}"
        else:
            exe_path = output_dir / app_name / f"{app_name}{ext}"
        
        print(f"\n{'='*60}")
        print("✓ Build exitoso!")
        print(f"{'='*60}")
        print(f"\nEjecutable creado en:")
        print(f"  {exe_path.absolute()}")
        print(f"\nPara distribuir:")
        if onefile:
            print(f"  - Copia el archivo {app_name}{ext} a cualquier computadora")
            print(f"  - Asegúrate de que los usuarios tengan instalados:")
            print(f"      • Python 3.10+")
            print(f"      • Las dependencias (requirements.txt)")
        else:
            print(f"  - Comprime la carpeta {app_name}/ completa")
            print(f"  - El usuario solo necesita extraer y ejecutar")
        
        if platform == "win32":
            print(f"\n📝 Nota para Windows:")
            print(f"   Si hay errores con MediaPipe, distribuye también la carpeta 'model/' y 'data/'")
        
        return True
    else:
        print(f"\n✗ Build fallido con código {result.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Compila LSC Traductor a ejecutable"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Solo limpia archivos de build anteriores"
    )
    parser.add_argument(
        "--onefile",
        action="store_true",
        help="Crea un único archivo ejecutable (más portable pero más lento al iniciar)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Genera ambos formatos (onefile y onedir)"
    )
    
    args = parser.parse_args()
    
    # Cambiar al directorio raíz del proyecto
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)
    
    print(f"Directorio del proyecto: {script_dir}\n")
    
    if args.clean:
        clean_build()
        return
    
    if not check_dependencies():
        sys.exit(1)
    
    if args.all:
        print("\n>>> Generando versión onedir...")
        build_executable(onefile=False)
        
        print("\n>>> Generando versión onefile...")
        build_executable(onefile=True)
    else:
        success = build_executable(onefile=args.onefile)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
