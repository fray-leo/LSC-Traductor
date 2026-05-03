# 📦 Instrucciones para Crear Ejecutables

Este documento explica cómo crear ejecutables de LSC Traductor para diferentes plataformas.

## Requisitos Previos

- Python 3.10 o superior instalado
- pip (gestor de paquetes de Python)
- El repositorio clonado con todo el código fuente

## Pasos Generales

### 1. Instalar Dependencias

```bash
pip install -r requirements.txt
```

Esto instalará automáticamente PyInstaller y todas las dependencias necesarias.

### 2. Compilar el Ejecutable

#### En Linux

```bash
# Opción A: Usar el script bash (recomendado)
./build_linux.sh --onefile

# Opción B: Usar build.py directamente
python3 build.py --onefile
```

El ejecutable se creará en `dist/LSC_Traductor`

#### En Windows

```cmd
REM Opción A: Usar el script batch (recomendado)
build_windows.bat --onefile

REM Opción B: Usar build.py directamente
python build.py --onefile
```

El ejecutable se creará en `dist\LSC_Traductor.exe`

#### En macOS

```bash
python3 build.py --onefile
```

La aplicación se creará en `dist/LSC_Traductor.app`

## Opciones de Compilación

| Opción | Descripción |
|--------|-------------|
| `--onefile` | Crea un único archivo ejecutable (más portable, pero más lento al iniciar) |
| `--clean` | Solo limpia archivos de compilaciones anteriores |
| `--all` | Genera ambos formatos (onefile y onedir) |

## Después de Compilar

### Para Distribuir

1. **Versión onefile (recomendada)**:
   - Copia el archivo ejecutable (`LSC_Traductor.exe`, `LSC_Traductor`, o `LSC_Traductor.app`)
   - Incluye las carpetas `model/` y `data/` con el modelo entrenado
   - Comprime todo en un archivo ZIP

2. **Versión onedir**:
   - Comprime la carpeta completa `dist/LSC_Traductor/`
   - El usuario solo necesita extraer y ejecutar

### Subir a GitHub Releases

1. Ve a la página de Releases de tu repositorio
2. Haz clic en "Create a new release"
3. Sube los archivos comprimidos
4. Etiqueta la versión (ej: v1.0.0)
5. Agrega notas del lanzamiento

## Solución de Problemas

### Error: "No module named 'mediapipe'"

Asegúrate de que todas las dependencias estén instaladas:
```bash
pip install -r requirements.txt
```

### Error en Linux: "Permission denied"

Da permisos de ejecución al script:
```bash
chmod +x build_linux.sh
```

### El ejecutable no inicia en Windows

Ejecuta desde la línea de comandos para ver errores:
```cmd
LSC_Traductor.exe
```

### Tamaño del ejecutable muy grande

Esto es normal porque incluye todas las dependencias. Usa `--onefile` para un solo archivo.

## Notas por Plataforma

### Windows
- Los ejecutables pueden ser marcados como sospechosos por algunos antivirus
- Considera firmar digitalmente el ejecutable para distribución profesional

### macOS
- Puede requerir permisos adicionales de cámara
- Los usuarios deben permitir la aplicación en Preferencias del Sistema > Seguridad

### Linux
- Asegúrate de que las librerías gráficas estén disponibles
- Puede requerir `libgl1-mesa-glx` u otras dependencias del sistema
