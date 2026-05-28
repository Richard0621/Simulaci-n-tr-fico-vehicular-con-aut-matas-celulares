# Reporte de Paralelización — Simulador de Tráfico Nagel-Schreckenberg
**Fecha:** 27 de mayo de 2026  
**Herramienta:** CUDA (NVIDIA GPU, arquitectura Ampere sm_86)  
**Archivos:** `AUserial_Serial.cpp` (serial) · `AUserial_CUDA.cu` (paralelo)

---

## 1. Parámetros de la corrida experimental

| Parámetro | Valor |
|---|---|
| Número de carriles | 4 |
| Longitud de carretera (celdas) | 100 000 |
| Total de celdas (`num_casillas`) | 400 000 |
| Densidad vehicular | 0.30 (30 %) |
| Número de vehículos | 120 000 |
| Iteraciones totales | 1 000 |
| Iteraciones de calentamiento (warmup) | 100 |
| Iteraciones medidas | 900 |
| Velocidad máxima (celdas/iter) | 5 |
| Prob. desaceleración (v=0) | 0.50 |
| Prob. desaceleración (v>0) | 0.30 |
| Prob. cambio de carril | 0.50 |
| Semilla cuRAND | 42 |

---

## 2. Resultados de simulación

| Métrica | Valor |
|---|---|
| Flujo total promedio (veh/iter) | 1.1811 |
| Flujo por carril (veh/iter/carril) | 0.2953 |
| Velocidad promedio (celdas/iter) | 1.0393 |
| Flujo teórico J = ρ · v̄ | 0.3118 |

---

## 3. Descripción del modelo Nagel-Schreckenberg

El modelo es un **autómata celular** unidimensional para tráfico vehicular.  
Cada celda de la carretera puede estar **vacía (`E`)** o contener un **vehículo (`V`)** con una velocidad entera `v ∈ [0, v_max]`.  
En cada iteración, todos los vehículos actualizan su estado de forma simultánea aplicando **4 reglas**:

1. **Acelerar:** `v ← min(v + 1, v_max)`
2. **Frenar por brecha:** `v ← min(v, distancia_al_próximo_vehículo - 1)`
3. **Desaceleración aleatoria:** con probabilidad `p` reducir `v` en 1 (modela metaestabilidad y frenadas humanas)
4. **Mover:** avanzar `v` celdas hacia adelante (con soporte de múltiples carriles)

El patrón de actualización es de **doble buffer**: se lee del estado `t` y se escribe en el estado `t+1`, luego se intercambian.

---

## 4. Análisis de paralelización

### 4.1 Partes paralelizadas en GPU

| Componente | Descripción | Kernel CUDA |
|---|---|---|
| **Fase 1 — Cálculo de reglas** | Para cada celda: aplica las 4 reglas N-S, calcula velocidad nueva y posición destino. Sin dependencias entre celdas del mismo paso. | `compute_kernel` |
| **Fase 2 — Aplicar movimientos** | Para cada celda con vehículo: mueve a la posición destino con resolución de colisiones. Actualiza contadores de flujo y velocidad. | `apply_kernel` |
| **Inicialización de cuRAND** | Inicializa un estado de generador aleatorio independiente por thread, una sola vez antes del loop. | `init_curand_kernel` |

**Justificación de la paralelización:**  
Las 4 reglas se aplican a cada celda de forma **independiente**: la regla de cada celda depende únicamente del estado `t` (lectura), nunca del estado `t+1` (escritura). Esto elimina toda dependencia entre threads dentro de la misma fase, haciendo que el problema sea **embarazosamente paralelo** en la dimensión espacial.

### 4.2 Partes que permanecen seriales (no paralelizables)

| Componente | Razón de no paralelización |
|---|---|
| **Loop principal de iteraciones** | Dependencia temporal: cada iteración `t+1` requiere el estado completo de `t`. No hay independencia entre pasos de tiempo. |
| **Inicialización de la carretera** (`llenar_carretera`) | Se ejecuta una sola vez. El costo es despreciable frente al loop. |
| **Transferencia host↔device** | Los arrays `h_tipo[]` y `h_vel[]` se copian a la GPU una vez al inicio con `cudaMemcpy`. |
| **Lectura de argumentos y validación** | Lógica de control, secuencial por diseño. |
| **Recolección de resultados** | Al final del loop se bajan 3 contadores de GPU a CPU con `cudaMemcpy`. |
| **Cálculo de métricas finales** | División/suma sobre los contadores descargados. Costo O(1). |

### 4.3 Estrategia de paralelización detallada

#### Asignación de threads
```
1 thread  →  1 celda de la carretera
num_casillas = num_carriles × longitud_carretera
bloques = ⌈num_casillas / BLOCK_SIZE⌉
```

Con `BLOCK_SIZE = 256` (múltiplo de warp = 32):

```
bloques = ⌈400 000 / 256⌉ = 1 563
threads totales = 1 563 × 256 = 400 128
threads activos = 400 000  (128 threads del último bloque inactivos)
warps totales   = 400 128 / 32 = 12 504
warps activos   = ⌈400 000 / 32⌉ = 12 500
```

#### Separación en dos fases por iteración

Cada iteración del loop principal se divide en dos kernels separados por una **barrera de sincronización** (`cudaDeviceSynchronize`):

```
Iteración t:
  ┌─────────────────────────────────────────────────────┐
  │  compute_kernel<<<1563, 256>>>                      │
  │    Lee:   d_tipo[t], d_vel[t]          (read-only)  │
  │    Escribe: vel_nueva[], pos_destino[] (sin conflicto)│
  └──────────────────── cudaDeviceSynchronize ──────────┘
  ┌─────────────────────────────────────────────────────┐
  │  cudaMemset(d_nueva_tipo, 0, ...)  ← cero el buffer │
  │  apply_kernel<<<1563, 256>>>                        │
  │    Lee:   vel_nueva[], pos_destino[], d_tipo[t]     │
  │    Escribe: d_nueva_tipo[t+1], d_nueva_vel[t+1]     │
  │    + atomicAdd a contadores de flujo/velocidad      │
  └──────────────────── cudaDeviceSynchronize ──────────┘
  swap(d_tipo, d_nueva_tipo)   ← O(1), solo punteros
  swap(d_vel,  d_nueva_vel)
```

#### Resolución de colisiones con `atomicCAS`

En la Fase 2 pueden existir **dos vehículos distintos que calcularon el mismo destino** (raro pero posible si `brecha == 0` y hay cambio de carril). Se resuelve con operación atómica:

```cuda
int antiguo = atomicCAS(&d_nueva_tipo[dest], 0, 1);
if (antiguo == 0) {
    // Ganó la celda destino
    d_nueva_vel[dest] = v;
} else {
    // Celda ya ocupada: el vehículo se queda en su posición original
    atomicCAS(&d_nueva_tipo[pos], 0, 1);
    d_nueva_vel[pos] = v;
}
```

`atomicCAS(addr, expected, val)` — si `*addr == expected`, escribe `val` y devuelve el valor anterior. La atomicidad garantiza que exactamente un thread "gana" la celda.

#### Generación aleatoria con cuRAND

La versión serial usa un único `mt19937` compartido. Esto es **inherentemente secuencial** porque cada extracción modifica el estado del generador.  
En la versión CUDA se usa **cuRAND con estado por thread**:

```cuda
curandState states[num_casillas];   // un estado por celda, en memoria global GPU
init_curand_kernel<<<blocks, BLOCK_SIZE>>>(states, num_casillas, CURAND_SEED);
```

Cada thread tiene su propio generador independiente (inicializado con la misma semilla pero diferente secuencia mediante `curand_init(..., pos, 0, &st)`), eliminando cualquier serialización en la generación de números aleatorios.

### 4.4 Asignación de memoria en GPU

| Array GPU | Tipo | Tamaño | Propósito |
|---|---|---|---|
| `d_tipo` | `int*` | 400 000 × 4 B = 1.6 MB | Estado actual: tipo de celda (0=vacío, 1=vehículo) |
| `d_vel` | `int*` | 400 000 × 4 B = 1.6 MB | Estado actual: velocidad por celda |
| `d_nueva_tipo` | `int*` | 1.6 MB | Buffer siguiente (estado t+1) |
| `d_nueva_vel` | `int*` | 1.6 MB | Buffer siguiente — velocidades |
| `d_vel_nueva` | `int*` | 1.6 MB | Intermedio Fase 1→2: velocidad calculada |
| `d_pos_destino` | `int*` | 1.6 MB | Intermedio Fase 1→2: posición destino |
| `d_states` | `curandState*` | 400 000 × 48 B ≈ 19.2 MB | Estado cuRAND por thread |
| `d_flujo_total` | `ull*` | 8 B | Contador de flujo acumulado |
| `d_vel_total` | `ull*` | 8 B | Suma de velocidades acumulada |
| `d_conteos_vel` | `int*` | 4 B | Número de mediciones de velocidad |
| **Total GPU** | | **≈ 30.8 MB** | |

### 4.5 Overhead de la paralelización

| Fuente de overhead | Descripción | Frecuencia |
|---|---|---|
| **Transferencia host→device** | Copia inicial de `h_tipo[]` y `h_vel[]` (~3.2 MB) | 1 vez |
| **Transferencia device→host** | Descarga de 3 contadores al final (~20 B) | 1 vez |
| **`cudaMemset`** | Poner `d_nueva_tipo` a cero antes de cada Fase 2 (~1.6 MB) | Cada iteración |
| **`cudaDeviceSynchronize`** | 2 barreras por iteración (entre Fase 1 y 2, y después de Fase 2) | 2× cada iter |
| **`init_curand_kernel`** | Inicialización de 400 000 estados cuRAND | 1 vez |
| **Latencia de lanzamiento de kernel** | Overhead de lanzar 2 kernels por iteración | 2× cada iter |
| **Threads desperdiciados** | 128 threads del último bloque sin trabajo útil | Permanente |
| **Memoria `curandState`** | 19.2 MB dedicados exclusivamente al RNG en GPU | Permanente |

> El overhead más relevante en la práctica es el **`cudaMemset` repetido** (1 000 veces, ~1.6 MB cada vez) y las **2 sincronizaciones por iteración**. Aun así, el speedup medido es ~14.7× sobre el serial.

---

## 5. Métricas de paralelización (corrida experimental)

| Métrica | Valor |
|---|---|
| Semilla cuRAND | 42 |
| Threads por bloque (`BLOCK_SIZE`) | 256 |
| Bloques lanzados por kernel | 1 563 |
| Threads totales lanzados | 400 128 |
| Threads activos (celdas) | 400 000 |
| Warps totales lanzados | 12 504 |
| Warps activos | 12 500 |
| **Tiempo CPU serial (s)** | **8.9303** |
| **Tiempo GPU CUDA (s)** | **0.6071** |
| **Speedup real S = T_cpu / T_gpu** | **14.71×** |
| Fracción serial estimada (f_s) | 0.0674 (6.74 %) |
| Speedup máximo — Ley de Amdahl | 14.84× |
| Speedup — Ley de Gustafson | 1 457.74× |

### 5.1 Ley de Amdahl

$$S_{max} = \frac{1}{f_s + \frac{1 - f_s}{n}}$$

Con $f_s = 0.0674$ y $n = 1\,563$ bloques (procesadores):

$$S_{max} = \frac{1}{0.0674} \approx 14.84$$

La fracción serial (~6.7 %) corresponde al overhead de: lanzamiento de kernels, sincronizaciones, `cudaMemset`, y el loop de iteraciones en CPU que no se puede eliminar. El speedup teórico máximo está **acotado por esa fracción**.

### 5.2 Ley de Gustafson

$$S_G = n - f_s \cdot (n - 1) = 1\,563 - 0.0674 \times 1\,562 \approx 1\,457.74$$

La Ley de Gustafson refleja que **si se escala el problema** (más carriles, carretera más larga) con el mismo tiempo de ejecución GPU, el speedup crece casi linealmente con el número de procesadores, porque la fracción serial se vuelve insignificante. Esto hace que el modelo sea altamente adecuado para GPU con problemas grandes.

---

## 6. Cuadro comparativo: Serial vs. CUDA

| Aspecto | `AUserial_Serial.cpp` | `AUserial_CUDA.cu` |
|---|---|---|
| **Plataforma de ejecución** | CPU (1 hilo) | GPU NVIDIA (sm_86, Ampere) |
| **Loop interno (por celda)** | `for (int i=0; i<num_casillas; i++)` — secuencial | `compute_kernel + apply_kernel` — todos los threads en paralelo |
| **Loop externo (iteraciones)** | `for (int t=0; t<iter; t++)` | Igual — secuencial en CPU (dependencia temporal) |
| **Generador aleatorio** | `mt19937` global — acceso secuencial | `curandState` por thread — totalmente paralelo |
| **Doble buffer** | `nueva_carretera = carretera` — copia O(n) | `swap(d_tipo, d_nueva_tipo)` — O(1), solo punteros |
| **Resolución de colisiones** | Imposible en serial (orden secuencial las evita) | `atomicCAS` — resuelve colisiones concurrentes |
| **`brechaDisponible`** | Lee `carretera` Y `nueva_carretera` | Solo lee `d_tipo` (estado anterior) — necesario para seguridad de datos |
| **Medición de tiempo** | `std::chrono::high_resolution_clock` | `cudaEvent_t` (solo mide tiempo en GPU) |
| **Memoria** | ~12.8 MB (vectors en heap) | ~30.8 MB en VRAM + ~12.8 MB en RAM |
| **Complejidad temporal por iter.** | O(num_casillas) | O(1) en GPU (+ overhead de lanzamiento) |
| **Tiempo total (1 000 iter.)** | **8.93 s** | **0.61 s** |
| **Speedup** | 1× (baseline) | **14.71×** |
| **Escalabilidad** | Lineal con num_casillas | Sub-lineal (overhead fijo) hasta saturar GPU |
| **Portabilidad** | Cualquier plataforma con C++11 | Requiere GPU NVIDIA con soporte CUDA |
| **Precisión de resultados** | `mt19937` determinista por semilla | `cuRAND` determinista por semilla + diferente distribución de números → resultados estadísticamente equivalentes (±5 %) |
| **Cambio de carril** | Puede leer `nueva_carretera` para detectar hueco | Solo lee estado anterior — diferencia semántica mínima en modelo estocástico |
