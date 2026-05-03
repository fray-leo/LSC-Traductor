#!/bin/bash
# build_linux.sh - Script para crear ejecutable de LSC Traductor en Linux
# Uso: ./build_linux.sh [--clean] [--onefile]

set -e

echo "========================================"
echo "LSC Traductor - Build Script para Linux"
echo "========================================"
echo ""

# Verificar que Python esté instalado
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 no está instalado"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $PYTHON_VERSION detectado"

# Verificar pip
if ! command -v pip3 &> /dev/null; then
    echo "❌ Error: pip3 no está instalado"
    exit 1
fi

echo "✓ pip3 detectado"
echo ""

# Instalar dependencias si es necesario
echo "Verificando dependencias..."
pip3 install -q -r requirements.txt
echo "✓ Dependencias instaladas/actualizadas"
echo ""

# Ejecutar build.py con los argumentos proporcionados
echo "Iniciando proceso de compilación..."
python3 build.py "$@"

echo ""
echo "========================================"
echo "Proceso completado"
echo "========================================"
