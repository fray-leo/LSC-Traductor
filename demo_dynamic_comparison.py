"""
demo_dynamic_comparison.py

Demostración comparativa entre enfoque estático y multi-frame para 
señas dinámicas similares (ej. "mamá" vs "mujer").

Este script muestra cómo un proyecto más avanzado podría mejorar 
la comprensión en tiempo real al considerar la información temporal.

Uso:
    python demo_dynamic_comparison.py
    
Nota: Este es un demo educativo que simula el comportamiento esperado.
No requiere un modelo entrenado real, pero explica los conceptos clave.
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
import time


# =============================================================================
# Simulación de datos para el demo
# =============================================================================

@dataclass
class FrameData:
    """Representa los landmarks de un frame individual."""
    landmarks: List[float]  # 63 floats
    timestamp: float
    frame_number: int


class SignMotionSimulator:
    """
    Simula el movimiento característico de señas dinámicas como 'mamá' y 'mujer'.
    
    En LSC:
    - 'mamá': Tocar la barbilla con los dedos, luego mover hacia abajo
    - 'mujer': Tocar la barbilla, luego hacer movimiento circular hacia el pecho
    
    La diferencia principal está en la trayectoria del movimiento.
    """
    
    def __init__(self):
        # Posición inicial común (mano cerca de la barbilla)
        self.base_position = self._create_base_pose()
        
    def _create_base_pose(self) -> List[float]:
        """Crea una pose base neutral (21 landmarks × 3 coordenadas = 63 valores)."""
        # Simulación simplificada: solo nos enfocamos en puntos clave
        landmarks = [0.0] * 63
        
        # Punta del dedo índice (landmark 8) cerca de la barbilla
        landmarks[24] = 0.5  # x
        landmarks[25] = 0.3  # y (barbilla)
        landmarks[26] = 0.2  # z
        
        # Muñeca (landmark 0) como referencia
        landmarks[0] = 0.5
        landmarks[1] = 0.6
        landmarks[2] = 0.3
        
        return landmarks
    
    def generate_mama_sequence(self, num_frames: int = 10) -> List[FrameData]:
        """
        Genera secuencia para 'mamá': movimiento vertical hacia abajo.
        """
        sequence = []
        start_time = time.time()
        
        for i in range(num_frames):
            landmarks = self.base_position.copy()
            progress = i / (num_frames - 1) if num_frames > 1 else 0
            
            # Movimiento característico: de barbilla hacia abajo (vertical)
            # El dedo se mueve en línea recta hacia abajo
            landmarks[25] = 0.3 + progress * 0.25  # y aumenta (hacia abajo)
            landmarks[24] = 0.5 + progress * 0.05  # ligero desplazamiento x
            
            sequence.append(FrameData(
                landmarks=landmarks,
                timestamp=start_time + i * 0.033,  # ~30 fps
                frame_number=i
            ))
        
        return sequence
    
    def generate_mujer_sequence(self, num_frames: int = 10) -> List[FrameData]:
        """
        Genera secuencia para 'mujer': movimiento circular hacia el pecho.
        """
        sequence = []
        start_time = time.time()
        
        for i in range(num_frames):
            landmarks = self.base_position.copy()
            progress = i / (num_frames - 1) if num_frames > 1 else 0
            angle = progress * np.pi  # medio círculo
            
            # Movimiento característico: arco circular hacia el pecho
            # El dedo hace un movimiento curvo
            landmarks[25] = 0.3 + np.sin(angle) * 0.15  # movimiento vertical oscilante
            landmarks[24] = 0.5 - np.cos(angle) * 0.1   # movimiento horizontal curvo
            landmarks[26] = 0.2 + progress * 0.15       # se acerca (z aumenta)
            
            sequence.append(FrameData(
                landmarks=landmarks,
                timestamp=start_time + i * 0.033,
                frame_number=i
            ))
        
        return sequence


# =============================================================================
# Enfoques de clasificación comparados
# =============================================================================

class StaticClassifier:
    """
    Clasificador estático: analiza cada frame independientemente.
    
    Limitación: No puede distinguir señas con poses iniciales similares
    pero trayectorias diferentes.
    """
    
    def __init__(self):
        # Umbrales simplificados para el demo
        self.chin_threshold = 0.35  # y < threshold = cerca de barbilla
        
    def predict_frame(self, landmarks: List[float]) -> Tuple[str, float]:
        """
        Predice la seña basada SOLO en la pose actual.
        
        Returns:
            Tupla (predicción, confianza)
        """
        # Extraer posición del dedo índice (landmark 8 → índices 24,25,26)
        finger_y = landmarks[25] if len(landmarks) > 25 else 0.5
        
        # Con solo información estática, ambas señas se ven iguales
        # cuando la mano está cerca de la barbilla
        if finger_y < self.chin_threshold:
            # Pose inicial ambigua: podría ser cualquiera
            return "AMBIGUO", 0.5
        elif finger_y > 0.5:
            # Mano muy abajo: probablemente 'mamá'
            return "mamá", 0.6
        else:
            return "...", 0.3
    
    def predict_sequence(self, sequence: List[FrameData]) -> Dict:
        """
        Aplica clasificación estática frame por frame.
        """
        results = []
        for frame in sequence:
            prediction, confidence = self.predict_frame(frame.landmarks)
            results.append({
                'frame': frame.frame_number,
                'prediction': prediction,
                'confidence': confidence
            })
        
        # Decisión final: votación mayoritaria
        predictions = [r['prediction'] for r in results]
        final_prediction = max(set(predictions), key=predictions.count)
        
        return {
            'approach': 'Estático (frame individual)',
            'frame_results': results,
            'final_prediction': final_prediction,
            'accuracy_estimate': 0.45  # ~45% para señas dinámicas similares
        }


class MultiFrameClassifier:
    """
    Clasificador multi-frame: analiza la secuencia completa.
    
    Ventaja: Puede capturar patrones de movimiento y trayectorias.
    """
    
    def __init__(self):
        pass
    
    def extract_motion_features(self, sequence: List[FrameData]) -> Dict:
        """
        Extrae características de movimiento de la secuencia.
        """
        if len(sequence) < 2:
            return {'total_displacement': 0, 'trajectory_type': 'unknown'}
        
        # Calcular desplazamiento total del dedo índice
        first_finger = (sequence[0].landmarks[24], sequence[0].landmarks[25])
        last_finger = (sequence[-1].landmarks[24], sequence[-1].landmarks[25])
        
        total_displacement = np.sqrt(
            (last_finger[0] - first_finger[0])**2 + 
            (last_finger[1] - first_finger[1])**2
        )
        
        # Analizar trayectoria (¿línea recta o curva?)
        mid_idx = len(sequence) // 2
        mid_finger = (sequence[mid_idx].landmarks[24], sequence[mid_idx].landmarks[25])
        
        # Desviación de la línea recta
        expected_mid_x = (first_finger[0] + last_finger[0]) / 2
        expected_mid_y = (first_finger[1] + last_finger[1]) / 2
        deviation = np.sqrt(
            (mid_finger[0] - expected_mid_x)**2 + 
            (mid_finger[1] - expected_mid_y)**2
        )
        
        trajectory_type = 'curved' if deviation > 0.05 else 'straight'
        
        # Verificar cambio en profundidad (z)
        depth_change = sequence[-1].landmarks[26] - sequence[0].landmarks[26]
        
        return {
            'total_displacement': total_displacement,
            'trajectory_type': trajectory_type,
            'deviation': deviation,
            'depth_change': depth_change
        }
    
    def predict_sequence(self, sequence: List[FrameData]) -> Dict:
        """
        Predice la seña analizando toda la secuencia.
        """
        motion_features = self.extract_motion_features(sequence)
        
        # Reglas basadas en características de movimiento
        if motion_features['trajectory_type'] == 'straight' and motion_features['depth_change'] < 0.05:
            prediction = "mamá"
            confidence = 0.78
            reasoning = "Movimiento vertical recto hacia abajo"
        elif motion_features['trajectory_type'] == 'curved' or motion_features['depth_change'] > 0.1:
            prediction = "mujer"
            confidence = 0.82
            reasoning = "Movimiento curvo hacia el pecho con cambio de profundidad"
        else:
            prediction = "..."
            confidence = 0.4
            reasoning = "Patrón de movimiento no claro"
        
        # Análisis frame por frame para visualización
        frame_results = []
        for i, frame in enumerate(sequence):
            # Enfoque multi-frame: confianza aumenta con más frames
            progressive_confidence = min(0.5 + (i / len(sequence)) * 0.4, confidence)
            frame_results.append({
                'frame': frame.frame_number,
                'prediction': prediction if i > len(sequence) // 2 else "ANALIZANDO",
                'confidence': progressive_confidence
            })
        
        return {
            'approach': 'Multi-Frame (secuencia completa)',
            'motion_features': motion_features,
            'frame_results': frame_results,
            'final_prediction': prediction,
            'confidence': confidence,
            'reasoning': reasoning,
            'accuracy_estimate': 0.75  # ~75% con información temporal
        }


# =============================================================================
# Demo y visualización
# =============================================================================

def print_comparison_table(static_result: Dict, multiframe_result: Dict, sign_name: str):
    """Imprime tabla comparativa de resultados."""
    print("\n" + "="*70)
    print(f"RESULTADOS PARA LA SEÑA: {sign_name.upper()}")
    print("="*70)
    
    print(f"\n{'Métrica':<35} {'Estático':<15} {'Multi-Frame':<15}")
    print("-"*70)
    
    print(f"{'Predicción final':<35} {static_result['final_prediction']:<15} {multiframe_result['final_prediction']:<15}")
    
    static_conf = max([r['confidence'] for r in static_result['frame_results']], default=0)
    multi_conf = multiframe_result.get('confidence', 0)
    print(f"{'Confianza máxima':<35} {static_conf:.2f}:{'':<13} {multi_conf:.2f}")
    
    print(f"{'Precisión estimada':<35} {static_result['accuracy_estimate']*100:.0f}%{'':<12} {multiframe_result['accuracy_estimate']*100:.0f}%")
    
    if 'reasoning' in multiframe_result:
        print(f"\nRazonamiento: {multiframe_result['reasoning']}")
    
    if 'motion_features' in multiframe_result:
        features = multiframe_result['motion_features']
        print(f"\nCaracterísticas de movimiento detectadas:")
        print(f"  • Tipo de trayectoria: {features['trajectory_type']}")
        print(f"  • Desplazamiento total: {features['total_displacement']:.3f}")
        print(f"  • Desviación de línea recta: {features['deviation']:.3f}")
        print(f"  • Cambio en profundidad (Z): {features['depth_change']:.3f}")


def print_frame_by_frame_analysis(static_result: Dict, multiframe_result: Dict):
    """Imprime análisis frame por frame."""
    print("\n" + "="*70)
    print("ANÁLISIS FRAME POR FRAME")
    print("="*70)
    
    print(f"\n{'Frame':<8} {'Estático':<20} {'Multi-Frame':<20}")
    print(f"{'':<8} {'Predicción (Conf.)':<20} {'Predicción (Conf.)':<20}")
    print("-"*70)
    
    static_frames = static_result['frame_results']
    multi_frames = multiframe_result['frame_results']
    
    for i in range(len(static_frames)):
        s = static_frames[i]
        m = multi_frames[i] if i < len(multi_frames) else {'prediction': 'N/A', 'confidence': 0}
        
        s_str = f"{s['prediction']} ({s['confidence']:.2f})"
        m_str = f"{m['prediction']} ({m['confidence']:.2f})"
        
        print(f"{i:<8} {s_str:<20} {m_str:<20}")


def print_obstacles_explanation():
    """Explica los obstáculos actuales para implementación real."""
    print("\n" + "="*70)
    print("OBSTÁCULOS PARA IMPLEMENTACIÓN REAL")
    print("="*70)
    
    obstacles = [
        ("1. Información temporal perdida", 
         "Los modelos estáticos analizan frames individuales, perdiendo la \n   trayectoria y velocidad del movimiento."),
        
        ("2. Velocidad variable de señado",
         "Diferentes usuarios realizan las señas a distintas velocidades, \n   requiriendo normalización temporal."),
        
        ("3. Costo computacional",
         "Procesar secuencias de video requiere 10-30× más potencia que \n   frames individuales (GPU recomendada)."),
        
        ("4. Escasez de datos temporales",
         "Se necesitan miles de secuencias completas (no solo fotos) para \n   entrenar modelos temporales efectivos."),
        
        ("5. Co-articulación",
         "En señado continuo, las señas se fusionan, haciendo difícil \n   identificar límites entre signos."),
        
        ("6. Sincronización temporal",
         "Alinear correctamente el inicio y fin de cada seña requiere \n   algoritmos adicionales de detección."),
    ]
    
    for title, description in obstacles:
        print(f"\n{title}")
        print(f"   {description}")
    
    print("\n" + "-"*70)
    print("SOLUCIONES POTENCIALES PARA PROYECTOS FUTUROS:")
    print("-" *70)
    print("""
    • Usar LSTM/GRU para modelar dependencias temporales
    • Implementar SlowFast networks para video recognition
    • Extraer features de flujo óptico (optical flow)
    • Aplicar data augmentation temporal (time warping)
    • Usar transformers con atención temporal
    • Incorporar pose estimation 3D con múltiples cámaras
    """)


def run_demo():
    """Ejecuta la demostración completa."""
    print("\n" + "#"*70)
    print("# DEMOSTRACIÓN: ENFOQUE ESTÁTICO VS MULTI-FRAMER")
    print("# Para señas dinámicas similares en LSC")
    print("#"*70)
    
    # Inicializar componentes
    simulator = SignMotionSimulator()
    static_clf = StaticClassifier()
    multiframe_clf = MultiFrameClassifier()
    
    # =========================================
    # Demo 1: Seña "mamá"
    # =========================================
    print("\n\n" + "="*70)
    print("ESCENARIO 1: Usuario realiza la seña 'mamá'")
    print("="*70)
    print("\nDescripción: Tocar barbilla → movimiento vertical hacia abajo")
    
    mama_sequence = simulator.generate_mama_sequence(num_frames=10)
    
    # Clasificación estática
    static_result_mama = static_clf.predict_sequence(mama_sequence)
    
    # Clasificación multi-frame
    multiframe_result_mama = multiframe_clf.predict_sequence(mama_sequence)
    
    print_comparison_table(static_result_mama, multiframe_result_mama, "mamá")
    print_frame_by_frame_analysis(static_result_mama, multiframe_result_mama)
    
    # =========================================
    # Demo 2: Seña "mujer"
    # =========================================
    print("\n\n" + "="*70)
    print("ESCENARIO 2: Usuario realiza la seña 'mujer'")
    print("="*70)
    print("\nDescripción: Tocar barbilla → movimiento circular hacia el pecho")
    
    mujer_sequence = simulator.generate_mujer_sequence(num_frames=10)
    
    # Clasificación estática
    static_result_mujer = static_clf.predict_sequence(mujer_sequence)
    
    # Clasificación multi-frame
    multiframe_result_mujer = multiframe_clf.predict_sequence(mujer_sequence)
    
    print_comparison_table(static_result_mujer, multiframe_result_mujer, "mujer")
    print_frame_by_frame_analysis(static_result_mujer, multiframe_result_mujer)
    
    # =========================================
    # Resumen comparativo
    # =========================================
    print("\n\n" + "#"*70)
    print("# RESUMEN COMPARATIVO")
    print("#"*70)
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│  MÉTRICA                    │  ESTÁTICO  │  MULTI-FRAME  │  MEJORA  │
├─────────────────────────────────────────────────────────────────────┤
│  Precisión (señas similares)│     45%    │      75%      │   +30%   │
│  Precisión (señas estáticas)│     90%    │      92%      │    +2%   │
│  Frames necesarios          │      1     │      5-10     │    -     │
│  Latencia                   │   ~33ms    │    200-300ms  │    -     │
│  Requerimientos GPU         │   No       │   Sí (rec.)   │    -     │
└─────────────────────────────────────────────────────────────────────┘

CONCLUSIÓN:
El enfoque multi-frame mejora significativamente la precisión para 
señas DINÁMICAS similares, pero tiene mayor costo computacional.

Para un proyecto estudiantil:
  ✓ Comenzar con enfoque estático (funciona bien para señas estáticas)
  ✓ Implementar promediado simple de 3-5 frames como paso intermedio
  ✓ Demostrar conceptualmente el enfoque multi-frame (como este script)
  ✓ Documentar limitaciones y proponer mejoras futuras
""")
    
    # =========================================
    # Explicación de obstáculos
    # =========================================
    print_obstacles_explanation()
    
    print("\n" + "#"*70)
    print("# FIN DE LA DEMOSTRACIÓN")
    print("#"*70 + "\n")


if __name__ == "__main__":
    run_demo()
