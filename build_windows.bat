@echo off
REM build_windows.bat - Script para crear ejecutable de LSC Traductor en Windows
REM Uso: build_windows.bat [--clean] [--onefile]

echo ========================================
echo LSC Traductor - Build Script para Windows
echo ========================================
echo.

REM Verificar que Python esté instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python no está instalado o no está en el PATH
    echo Descarga Python desde https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo ✓ Python %PYTHON_VERSION% detectado

REM Verificar pip
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: pip no está instalado
    pause
    exit /b 1
)

echo ✓ pip detectado
echo.

REM Instalar dependencias si es necesario
echo Verificando dependencias...
pip install -q -r requirements.txt
echo ✓ Dependencias instaladas/actualizadas
echo.

REM Ejecutar build.py con los argumentos proporcionados
echo Iniciando proceso de compilación...
python build.py %*

echo.
echo ========================================
echo Proceso completado
echo ========================================
pause
