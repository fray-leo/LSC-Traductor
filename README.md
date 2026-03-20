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
|---|---|
| Lenguaje | Python 3.10+ |
| Detección de manos | [MediaPipe](https://mediapipe.dev/) |
| Clasificación de señas | scikit-learn (RandomForest / SVM) |
| Interfaz | tkinter |
| Base de datos | SQLite3 |
| Control de versiones | Git / GitHub |

---

## Estructura del repositorio

```
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
```

---

## Instalación

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
python src/app.py
```

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
|---|---|---|
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
|---|---|---|
| Bautista Talero, Leonardo Gael | 202420219 | Scrum Master / Desarrollo |
| Castro Gualteros, Nicolás | — | — |
| Fernández Pachón, Samuel | 202616452 | — |
| Molano Peña, Sebastián | — | — |

---

## 📄 Licencia

Proyecto académico — Universidad de los Andes, 2026.  
No tiene licencia de uso comercial.
