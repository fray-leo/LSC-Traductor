# LSC Traductor — Lengua de Señas Colombiana a Texto en Tiempo Real

> Proyecto Inicial — Introducción a la Ingeniería de Sistemas y Computación  
> Universidad de los Andes · Sección 4, Grupo 7 · 2026-1

---

## Descripción

**LSC Traductor** es una aplicación que usa visión por computador para reconocer señas de la **Lengua de Señas Colombiana (LSC)** a través de la cámara web y traducirlas a texto en tiempo real.

El objetivo es facilitar la comunicación entre personas sordas y oyentes en contextos cotidianos — consultas médicas, trámites, clases — donde no siempre hay un intérprete disponible. A diferencia de soluciones existentes basadas en ASL o LSE, este proyecto está diseñado específicamente para la LSC.

---

## Problema que resuelve

Las personas sordas en Colombia enfrentan barreras de comunicación constantes con personas oyentes que no conocen la LSC. No existe actualmente una herramienta accesible y gratuita que traduzca LSC a texto en tiempo real. Este proyecto es un primer paso hacia cerrar esa brecha.

---

## Stack tecnológico

| Componente | Tecnología |
| --- | --- |
| Lenguaje | Python 3.10+ |
| Detección de manos y pose | [MediaPipe Holistic](https://mediapipe.dev/) |
| Clasificación de señas | scikit-learn (RandomForest / SVM) |
| Interfaz | tkinter |
| Base de datos | SQLite3 |
| Control de versiones | Git / GitHub |

---

## Cambios recientes: MediaPipe Holistic

Este proyecto ha sido actualizado para usar **MediaPipe Holistic** en lugar de MediaPipe Hands, lo que permite:

- **Detección de ambas manos simultáneamente** (izquierda y derecha)
- **Referencia corporal completa** mediante landmarks del torso (hombros, caderas)
- **Normalización basada en el torso**: las posiciones de las manos se calculan relativas al centro del cuerpo, usando la distancia entre hombros como escala
- **Mayor contexto espacial** para interpretar señas que involucran movimiento del brazo o posición relativa al cuerpo

### Vector de landmarks (603 valores)

El nuevo sistema extrae un vector de 603 floats organizados así:

| Componente | Landmarks | Valores |
| --- | --- | --- |
| Mano izquierda | 21 × 3 (x, y, z) | 63 |
| Mano derecha | 21 × 3 (x, y, z) | 63 |
| Pose corporal | 25 × 3 (x, y, z) | 75 |
| **Total** | **201 × 3** | **603** |

La normalización usa los hombros como referencia, haciendo el vector invariante a:
- La distancia del usuario a la cámara
- La posición del cuerpo en el frame
- El tamaño aparente de la persona

---

## Estructura del repositorio

´´´
lsc-traductor/
│
├── data/                   # Datos de entrenamiento (landmarks por seña)
│   └── raw/                # Videos o imágenes originales (no subir al repo)
│
├── model/                  # Modelo entrenado serializado (.pkl)
│
├── src/
│   ├── capture.py          # Captura de video y extracción de landmarks
│   ├── train.py            # Entrenamiento del clasificador
│   ├── predict.py          # Predicción en tiempo real
│   └── app.py              # Interfaz principal (tkinter)
│
├── db/
│   └── historial.db        # Base de datos SQLite (generada automáticamente)
│
├── requirements.txt
└── README.md
´´´

---

## Instalación y Uso

### 1. Clonar el repositorio

```bash
git clone https://github.com/[usuario]/lsc-traductor.git
cd lsc-traductor
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar la aplicación

```bash
python main.py
```

---

## Funcionalidades de la Interfaz

La aplicación cuenta con dos botones principales:

| Botón | Función |
| --- | --- |
| **＋ Grabar Nueva Seña** | Abre un diálogo para grabar nuevas muestras de señas y agregarlas al dataset de entrenamiento |
| **Limpiar** (en el panel derecho) | Borra el historial de traducciones guardadas |

### Grabar Nuevas Señas

1. Haz clic en **"＋ Grabar Nueva Seña"**
2. Ingresa el nombre de la seña (ej: "hola", "gracias")
3. Configura el número de muestras a capturar (mínimo 10)
4. Presiona **"▶ Iniciar Grabación"**
5. Después de la cuenta regresiva, mantén la seña visible frente a la cámara
6. El sistema capturará automáticamente las muestras
7. Al finalizar, las muestras se guardan en `data/landmarks/{nombre}.csv`

> **Nota:** Para que los cambios surtan efecto en el reconocimiento, debes re-entrenar el modelo después de agregar nuevas señas:
> ```bash
> python -m App.logic.trainer --train
> ```

### Ampliar Datos de una Seña Existente

El sistema maneja automáticamente los conflictos cuando quieres agregar más datos a una seña que ya existe:

**Ejemplo: Quieres mejorar el reconocimiento de "gracias"**

1. Haces clic en "＋ Grabar Nueva Seña"
2. Escribes `gracias` como nombre
3. El sistema detecta que ya existen datos y muestra:
   ```
   La seña 'gracias' ya tiene 50 muestras guardadas.
   
   ¿Deseas agregar 50 muestras adicionales?
   
   Esto mejorará el reconocimiento de esta seña.
   ```
4. Confirmas con "Sí"
5. Grabas las nuevas muestras
6. Mensaje final:
   ```
   Se guardaron 50 muestras adicionales para 'gracias'.
   
   Total acumulado: 100 muestras.
   (Datos existentes: 50)
   ```

**Beneficios:**
- ✅ **Mejor generalización**: El modelo aprende variaciones de la misma seña
- ✅ **Más robustez**: Funciona mejor con diferentes usuarios e iluminación
- ✅ **Sin sobrescritura**: Los datos nuevos se anexan, no reemplazan los existentes

### Gestión del Dataset (Línea de Comandos)

Puedes administrar los datos de entrenamiento con herramientas CLI:

```bash
# Listar todas las señas y su conteo
python -m App.logic.dataset_manager list

# Ver estadísticas detalladas
python -m App.logic.dataset_manager stats

# Eliminar todos los datos de una seña específica
python -m App.logic.dataset_manager clear gracias
```

**Ejemplo de salida:**
```
$ python -m App.logic.dataset_manager list
Seña                 | Muestras    
--------------------------------
gracias              | 100       
hola                 | 75        
familia              | 50        

$ python -m App.logic.dataset_manager stats
Total Señas: 3
Total Muestras: 225
Promedio por seña: 75.00
```

---

## Crear Ejecutable (.exe / .app / binario)

Para distribuir la aplicación como un programa independiente:

### Opción A: Ejecutable único (recomendado para distribución)

```bash
python build.py --onefile
```

Esto crea un solo archivo ejecutable en `dist/LSC_Traductor.exe` (Windows) o `dist/LSC_Traductor` (Linux/Mac).

### Opción B: Carpeta con todos los archivos

```bash
python build.py
```

Esto crea una carpeta `dist/LSC_Traductor/` con el ejecutable y todas las dependencias incluidas.

### Requisitos para compilar

- PyInstaller (ya incluido en `requirements.txt`)
- Espacio en disco: ~200MB para el proceso de compilación

### Notas por plataforma

| Plataforma | Formato | Notas |
| --- | --- | --- |
| Windows | `.exe` | Funciona sin Python instalado si usas `--onefile` |
| macOS | `.app` | Puede requerir permisos de cámara adicionales |
| Linux | Binario | Asegúrate de tener las librerías gráficas necesarias |

### Distribución

Si creas un ejecutable `--onefile`, ten en cuenta:
- El usuario final **no necesita instalar Python**
- Debes distribuir también las carpetas `model/` y `data/` con el modelo entrenado
- El primer inicio puede ser más lento (el ejecutable se descomprime en memoria)

---

## Cómo contribuir

Este repositorio usa **ramas por funcionalidad**. Por favor no trabajen directamente sobre `main`.

```bash
# Crear tu rama
git checkout -b feature/nombre-de-tu-tarea

# Cuando termines, hacer push
git push origin feature/nombre-de-tu-tarea

# Luego abrir un Pull Request para revisión
```

### Ramas sugeridas por tarea

| Rama | Responsable | Descripción |
| --- | --- | --- |
| `feature/captura-landmarks` | — | Captura de video y extracción de puntos de mano |
| `feature/entrenamiento` | — | Recolección de datos y entrenamiento del modelo |
| `feature/interfaz` | — | Diseño de la interfaz en tkinter |
| `feature/base-de-datos` | — | Módulo de historial en SQLite |
| `docs/documentacion` | — | Documentación del proyecto (entrega escrita) |

> Asigna tu nombre en la tabla cuando tomes una tarea.

---

## Estado del proyecto

- [ ] Captura de landmarks con MediaPipe
- [ ] Recolección de datos de entrenamiento (vocabulario base LSC)
- [ ] Entrenamiento del clasificador
- [ ] Predicción en tiempo real
- [ ] Interfaz de usuario
- [ ] Base de datos de historial
- [ ] Pruebas y validación con usuarios

---

## Equipo

| Nombre | Código | Rol |
| --- | --- | --- |
| Bautista Talero, Leonardo Gael | 202420219 | Scrum Master / Desarrollo |
| Castro Gualteros, Nicolás | — | — |
| Fernández Pachón, Samuel | 202616452 | — |
| Molano Peña, Sebastián | — | — |

---

## 📄 Licencia

Proyecto académico — Universidad de los Andes, 2026.  
No tiene licencia de uso comercial.
