/*
******************************************************************************
                    SIMULADOR DE TRAFICO VEHICULAR
                    (Modelo Nagel-Schreckenberg)
                    VERSION PARALELA — CUDA
******************************************************************************

COMPILACION:
    nvcc -O2 -arch=sm_86 -o AUserial_CUDA AUserial_CUDA.cu

    Ajustar -arch según la GPU disponible:
        sm_75  → Turing  (RTX 20xx, GTX 16xx)
        sm_86  → Ampere  (RTX 30xx)
        sm_89  → Ada     (RTX 40xx)

EJECUCION:
    ./AUserial_CUDA <num_carriles> <longitud_carretera> <densidad_vehicular> <num_iteraciones>

DIFERENCIAS RESPECTO AL SERIAL:
    - Loop interno (por celda) ejecuta en GPU con 1 thread por celda.
    - brechaDisponible solo lee el estado anterior (no nueva_carretera).
    - Movimiento de vehiculos con atomicCAS para resolver colisiones.
    - Generador aleatorio mt19937 reemplazado por cuRAND (estado por thread).
    - Swap de punteros en lugar de copia O(n) del doble buffer.
    - Cronometro con cudaEvent (mide solo el loop de simulacion en GPU).
******************************************************************************
*/

#include <iostream>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <vector>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>
#include <curand_kernel.h>

using namespace std;

// ─── Parámetros globales (host) ──────────────────────────────────────────────
int    velocidad_max       = 5;
int    num_carriles;
int    longitud_carretera;
double densidad_vehicular;
int    num_iteraciones;
int    num_casillas;

double prob_cambio_carril      = 0.5;
double prob_desaceleracion_max = 0.5;
double prob_desaceleracion_min = 0.3;

// ─── Constantes de paralelización (globales para usarlas en métricas) ────────
const int           BLOCK_SIZE   = 256;   // threads por bloque (múltiplo de warp = 32)
const unsigned long CURAND_SEED  = 42UL;  // semilla del generador aleatorio en GPU

// ─── Estructura Casilla (solo usada en host para la inicialización) ───────────
struct Casilla {
    char tipo      = 'E';
    int  velocidad = 0;
};

// ─── Macro de verificación de errores CUDA ───────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error en %s:%d  —  %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(_err));              \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ═════════════════════════════════════════════════════════════════════════════
//  FUNCIONES HOST  (sin cambios respecto al serial)
// ═════════════════════════════════════════════════════════════════════════════

void validar_argumentos(int argc, char *argv[])
{
    if (argc < 5) {
        cout << "Se requieren 4 argumentos:  num_carriles, longitud_carretera, "
                "densidad_vehicular y num_iteraciones" << endl;
        exit(1);
    }
    num_carriles       = atoi(argv[1]);
    longitud_carretera = atoi(argv[2]);
    densidad_vehicular = atof(argv[3]);
    num_iteraciones    = atoi(argv[4]);

    if (!((num_iteraciones > 0) && (num_carriles >= 1 && num_carriles <= 4) &&
          (longitud_carretera > 0) &&
          (densidad_vehicular >= 0.1 && densidad_vehicular <= 1.0)))
    {
        cout << "Error: num_iteraciones debe ser un número positivo, num_carriles entre 1-4, "
                "longitud_carretera mayor a 0, y densidad_vehicular entre 0.1 y 1.0" << endl;
        exit(1);
    }
}

// Inicialización idéntica al serial (mismo algoritmo y semilla)
void llenar_carretera(vector<Casilla> &carretera)
{
    mt19937 gen(19);  // misma semilla que el serial
    uniform_int_distribution<int> dist_pos(0, num_casillas - 1);
    uniform_int_distribution<int> dist_vel(0, velocidad_max);

    int num_vehiculos = (int)(longitud_carretera * num_carriles * densidad_vehicular);
    int colocados = 0, intentos = 0;
    int max_intentos = num_casillas * 4;

    while (colocados < num_vehiculos && intentos < max_intentos) {
        int posicion = dist_pos(gen);
        if (carretera[posicion].tipo == 'E') {
            carretera[posicion].tipo      = 'V';
            carretera[posicion].velocidad = dist_vel(gen);
            colocados++;
        }
        intentos++;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  DEVICE FUNCTIONS  (versión GPU de las funciones de simulación)
// ═════════════════════════════════════════════════════════════════════════════

/*
 * brechaDisponible — versión __device__
 *
 * Cambio respecto al serial: solo lee d_tipo (estado anterior, read-only).
 * En paralelo es imposible leer nueva_carretera de forma segura durante
 * la fase de cómputo, porque otros threads aún no han escrito allí.
 * El efecto estadístico es mínimo en un modelo estocástico.
 */
__host__ __device__ int brechaDisponible(const int* d_tipo, int pos,
                                          int longitud_carretera, int velocidad_max)
{
    int carril        = pos / longitud_carretera;
    int pos_en_carril = pos % longitud_carretera;

    for (int i = 1; i <= velocidad_max; i++) {
        int sig = (pos_en_carril + i) % longitud_carretera;
        int idx = carril * longitud_carretera + sig;
        if (d_tipo[idx] == 1)   // 1 = 'V' (vehículo)
            return i - 1;
    }
    return velocidad_max;
}

/*
 * carrilAlternativo — versión __device__
 *
 * Reemplaza distrib(gen) por curand_uniform(state) (RNG por thread).
 * Solo lee d_tipo (estado anterior).
 */
__device__ int carrilAlternativo(const int* d_tipo, int pos,
                                  int velocidad_nueva, curandState* state,
                                  int num_carriles, int longitud_carretera,
                                  float prob_cambio_carril)
{
    int carril_actual = pos / longitud_carretera;
    int pos_en_carril = pos % longitud_carretera;

    if (curand_uniform(state) >= prob_cambio_carril)
        return carril_actual;

    int destino_en_carril = (pos_en_carril + velocidad_nueva) % longitud_carretera;
    bool puede_izq = (carril_actual > 0);
    bool puede_der = (carril_actual < num_carriles - 1);

    bool izq_disp = puede_izq &&
        d_tipo[(carril_actual - 1) * longitud_carretera + destino_en_carril] == 0;
    bool der_disp = puede_der &&
        d_tipo[(carril_actual + 1) * longitud_carretera + destino_en_carril] == 0;

    if ( izq_disp && !der_disp) return carril_actual - 1;
    if (!izq_disp &&  der_disp) return carril_actual + 1;
    if ( izq_disp &&  der_disp)
        return (curand_uniform(state) < 0.5f) ? carril_actual - 1 : carril_actual + 1;

    return carril_actual;
}

// ═════════════════════════════════════════════════════════════════════════════
//  KERNELS CUDA
// ═════════════════════════════════════════════════════════════════════════════

/*
 * init_curand_kernel
 * Inicializa un estado cuRAND independiente por thread.
 * Se llama una sola vez antes del loop principal.
 */
__global__ void init_curand_kernel(curandState* states, int n, unsigned long seed)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < n)
        curand_init(seed, pos, 0, &states[pos]);
}

/*
 * compute_kernel  —  FASE 1 (por iteración)
 *
 * Un thread por celda. Lee d_tipo[] y d_vel[] (estado anterior, read-only).
 * Escribe vel_nueva[] y pos_destino[] sin conflictos (escritura en índice propio).
 *
 * Aplica las 4 reglas de Nagel-Schreckenberg:
 *   1. Acelerar
 *   2. Frenar por brecha
 *   3. Desaceleración aleatoria / metaestabilidad
 *   4. Determinar carril y posición destino
 */
__global__ void compute_kernel(
    const int*   d_tipo,
    const int*   d_vel,
    int*         vel_nueva,
    int*         pos_destino,
    curandState* states,
    int num_casillas, int longitud_carretera, int num_carriles,
    int velocidad_max,
    float prob_desaceleracion_max,
    float prob_desaceleracion_min,
    float prob_cambio_carril)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= num_casillas) return;

    // Celdas vacías: no hay nada que calcular
    if (d_tipo[pos] != 1) {
        pos_destino[pos] = pos;
        vel_nueva[pos]   = 0;
        return;
    }

    // Cargar estado cuRAND local (más eficiente que acceder a memoria global en cada llamada)
    curandState st = states[pos];

    // REGLA 1: Acelerar
    int v = min(d_vel[pos] + 1, velocidad_max);

    // REGLA 2: Reducir velocidad según brecha disponible
    int brecha = brechaDisponible(d_tipo, pos, longitud_carretera, velocidad_max);
    v = min(v, brecha);

    // REGLA 3: Desaceleración aleatoria / metaestabilidad
    if (d_vel[pos] == 0) {
        // Mayor dificultad para arrancar si velocidad actual es 0
        if (curand_uniform(&st) <= prob_desaceleracion_max && v > 0)
            v -= 1;
    } else if (curand_uniform(&st) <= prob_desaceleracion_min) {
        v = max(v - 1, 0);
    }

    // REGLA 4: Determinar carril destino (cambio de carril si brecha == 0)
    int carril_dest;
    if (brecha == 0)
        carril_dest = carrilAlternativo(d_tipo, pos, v, &st,
                                        num_carriles, longitud_carretera,
                                        prob_cambio_carril);
    else
        carril_dest = pos / longitud_carretera;

    int pos_en_carril       = pos % longitud_carretera;
    int nueva_pos_en_carril = (pos_en_carril + v) % longitud_carretera;
    int dest                = carril_dest * longitud_carretera + nueva_pos_en_carril;

    vel_nueva[pos]   = v;
    pos_destino[pos] = dest;
    states[pos]      = st;  // guardar estado cuRAND actualizado
}

/*
 * apply_kernel  —  FASE 2 (por iteración)
 *
 * Un thread por celda con vehículo. Lee vel_nueva[] y pos_destino[] (Fase 1).
 * Escribe en d_nueva_tipo[] (previamente zeroed con cudaMemset).
 *
 * Resolución de colisiones con atomicCAS:
 *   - Si la celda destino está libre (0): el vehículo se mueve (atomicCAS exitoso).
 *   - Si está ocupada: el vehículo se queda en su posición con la nueva velocidad.
 *
 * Garantía de no colisión en fallback:
 *   Un vehículo en pos NO puede ser destino de otro vehículo, porque
 *   brechaDisponible garantiza que el destino de cualquier vehículo detrás
 *   es estrictamente menor que pos (no se puede saltar sobre un vehículo).
 */
__global__ void apply_kernel(
    int*        d_nueva_tipo,
    int*        d_nueva_vel,
    const int*  d_tipo,           // estado anterior (read-only, para medición de flujo)
    const int*  vel_nueva,
    const int*  pos_destino,
    unsigned long long*  d_flujo_total,
    unsigned long long*  d_vel_total,
    int*        d_conteos_vel,
    int num_casillas, int longitud_carretera,
    int detector_pos, int medir_flujo)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= num_casillas) return;
    if (d_tipo[pos] != 1) return;  // solo procesar celdas con vehículo en estado anterior

    int v    = vel_nueva[pos];
    int dest = pos_destino[pos];

    // Intentar ocupar la celda destino de forma atómica
    // d_nueva_tipo inicia en 0 (cudaMemset al inicio de cada iteración)
    int old = atomicCAS(&d_nueva_tipo[dest], 0, 1);

    int resultado;
    if (old == 0) {
        // Movimiento exitoso: escribir velocidad en destino
        // La celda origen (pos) queda en 0 — ya inicializada con memset
        d_nueva_vel[dest] = v;
        resultado = dest;
    } else {
        // Colisión: el vehículo se queda en su posición original
        // Es seguro escribir sin atomicCAS porque ningún otro thread
        // puede tener como destino una celda ocupada en el estado anterior
        d_nueva_tipo[pos] = 1;
        d_nueva_vel[pos]  = v;
        resultado = -1;
    }

    // Medición de flujo mediante detector virtual (solo post-warmup)
    if (medir_flujo) {
        if (resultado >= 0) {
            int x_old = pos       % longitud_carretera;
            int x_new = resultado % longitud_carretera;
            // Detección de cruce del detector con soporte de wrap-around
            bool cruza = false;
            if (x_old < x_new)
                cruza = (x_old < detector_pos && detector_pos <= x_new);
            else if (x_old > x_new)   // solo wrap-around real
                cruza = (x_old < detector_pos || detector_pos <= x_new);
            if (cruza) atomicAdd(d_flujo_total, 1ULL);
        }
        atomicAdd(d_vel_total,    (unsigned long long)v);
        atomicAdd(d_conteos_vel,  1);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  SIMULACIÓN SERIAL EN CPU  (baseline para calcular speedup)
//
//  Usa el mismo algoritmo que la GPU (brechaDisponible sin nueva_carretera)
//  para una comparación justa. Se corre con el mismo estado inicial.
// ═════════════════════════════════════════════════════════════════════════════
double run_serial_cpu(const vector<int>& tipo_init, const vector<int>& vel_init)
{
    vector<int> s_tipo     = tipo_init;
    vector<int> s_vel      = vel_init;
    vector<int> s_nueva_tipo(num_casillas, 0);
    vector<int> s_nueva_vel (num_casillas, 0);

    mt19937 rng((unsigned int)CURAND_SEED);
    uniform_real_distribution<float> udist(0.0f, 1.0f);

    auto t0 = chrono::high_resolution_clock::now();

    for (int iter = 0; iter < num_iteraciones; iter++) {
        fill(s_nueva_tipo.begin(), s_nueva_tipo.end(), 0);

        for (int pos = 0; pos < num_casillas; pos++) {
            if (s_tipo[pos] != 1) continue;

            // REGLA 1
            int v = min(s_vel[pos] + 1, velocidad_max);

            // REGLA 2: brecha (solo estado anterior, igual que GPU)
            int brecha = brechaDisponible(s_tipo.data(), pos,
                                          longitud_carretera, velocidad_max);
            v = min(v, brecha);

            // REGLA 3
            if (s_vel[pos] == 0) {
                if (udist(rng) <= (float)prob_desaceleracion_max && v > 0) v--;
            } else if (udist(rng) <= (float)prob_desaceleracion_min) {
                v = max(v - 1, 0);
            }

            // REGLA 4: carril destino
            int ca = pos / longitud_carretera;
            int pec = pos % longitud_carretera;
            int cd  = ca;
            if (brecha == 0 && udist(rng) < (float)prob_cambio_carril) {
                int dst = (pec + v) % longitud_carretera;
                bool puede_izq = (ca > 0);
                bool puede_der = (ca < num_carriles - 1);
                bool izq_disp  = puede_izq && s_tipo[(ca-1)*longitud_carretera+dst] == 0;
                bool der_disp  = puede_der && s_tipo[(ca+1)*longitud_carretera+dst] == 0;
                if ( izq_disp && !der_disp) cd = ca - 1;
                else if (!izq_disp &&  der_disp) cd = ca + 1;
                else if ( izq_disp &&  der_disp) cd = (udist(rng) < 0.5f) ? ca-1 : ca+1;
            }

            int dest = cd * longitud_carretera + (pec + v) % longitud_carretera;

            // Mover (sin atomicCAS: si colisión el vehículo queda en pos)
            if (s_nueva_tipo[dest] == 0) {
                s_nueva_tipo[dest] = 1;
                s_nueva_vel [dest] = v;
            } else {
                s_nueva_tipo[pos] = 1;
                s_nueva_vel [pos] = v;
            }
        }

        swap(s_tipo, s_nueva_tipo);
        swap(s_vel,  s_nueva_vel);
    }

    auto t1 = chrono::high_resolution_clock::now();
    return chrono::duration<double>(t1 - t0).count();
}

// ═════════════════════════════════════════════════════════════════════════════
//  main
// ═════════════════════════════════════════════════════════════════════════════
int main(int argc, char *argv[])
{
    // ── Validar argumentos (igual que serial) ────────────────────────────────
    validar_argumentos(argc, argv);
    num_casillas = longitud_carretera * num_carriles;

    // ── Inicializar carretera en host (igual que serial, misma semilla) ──────
    vector<Casilla> h_carretera(num_casillas);
    llenar_carretera(h_carretera);

    // ── Convertir struct Casilla → arrays int para la GPU ────────────────────
    //    GPU usa: 0 = vacío ('E'), 1 = vehículo ('V')
    vector<int> h_tipo(num_casillas), h_vel(num_casillas);
    for (int i = 0; i < num_casillas; i++) {
        h_tipo[i] = (h_carretera[i].tipo == 'V') ? 1 : 0;
        h_vel[i]  = h_carretera[i].velocidad;
    }

    // ── Ejecutar simulación serial (CPU) para medir baseline ────────────────
    double t_cpu = run_serial_cpu(h_tipo, h_vel);

    // ── Alocar memoria en GPU ────────────────────────────────────────────────
    int       *d_tipo, *d_vel;           // buffer actual (estado t)
    int       *d_nueva_tipo, *d_nueva_vel; // buffer siguiente (estado t+1)
    int       *d_vel_nueva, *d_pos_destino; // arrays intermedios (Fase 1 → Fase 2)
    unsigned long long *d_flujo_total, *d_vel_total;
    int       *d_conteos_vel;
    curandState *d_states;

    CUDA_CHECK(cudaMalloc(&d_tipo,          num_casillas * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vel,           num_casillas * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nueva_tipo,    num_casillas * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nueva_vel,     num_casillas * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vel_nueva,     num_casillas * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pos_destino,   num_casillas * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_flujo_total,   sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_vel_total,     sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_conteos_vel,   sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_states,        num_casillas * sizeof(curandState)));

    // ── Copiar estado inicial a GPU ──────────────────────────────────────────
    CUDA_CHECK(cudaMemcpy(d_tipo, h_tipo.data(), num_casillas * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel,  h_vel.data(),  num_casillas * sizeof(int), cudaMemcpyHostToDevice));

    // ── Inicializar contadores a 0 ───────────────────────────────────────────
    CUDA_CHECK(cudaMemset(d_flujo_total,  0, sizeof(long long)));
    CUDA_CHECK(cudaMemset(d_vel_total,    0, sizeof(long long)));
    CUDA_CHECK(cudaMemset(d_conteos_vel,  0, sizeof(int)));

    // ── Configuración de bloques y threads ───────────────────────────────────
    int blocks = (num_casillas + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // ── Inicializar cuRAND (una sola vez, fuera del loop) ────────────────────
    init_curand_kernel<<<blocks, BLOCK_SIZE>>>(d_states, num_casillas, CURAND_SEED);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Variables de medición ────────────────────────────────────────────────
    const int detector_pos       = longitud_carretera / 2;
    const int warmup_iteraciones = max(1, num_iteraciones / 10);

    // ── Cronómetro CUDA (mide solo el loop de simulación) ────────────────────
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start));

    // ═══════════════════════════════════════════════════════════════════════
    //  LOOP PRINCIPAL  (estructura idéntica al serial)
    //  El loop externo (iteraciones) es secuencial en CPU.
    //  El loop interno (por celda) se ejecuta en paralelo en GPU.
    // ═══════════════════════════════════════════════════════════════════════
    for (int iteracion = 0; iteracion < num_iteraciones; iteracion++)
    {
        int medir = (iteracion >= warmup_iteraciones) ? 1 : 0;

        // ── FASE 1: calcular velocidades y posiciones destino ────────────────
        //    Lee d_tipo[] y d_vel[] (read-only).
        //    Escribe d_vel_nueva[] y d_pos_destino[] (sin conflictos).
        compute_kernel<<<blocks, BLOCK_SIZE>>>(
            d_tipo, d_vel,
            d_vel_nueva, d_pos_destino,
            d_states,
            num_casillas, longitud_carretera, num_carriles,
            velocidad_max,
            (float)prob_desaceleracion_max,
            (float)prob_desaceleracion_min,
            (float)prob_cambio_carril);
        CUDA_CHECK(cudaDeviceSynchronize());  // barrera: Fase 2 espera a que Fase 1 termine

        // ── FASE 2: aplicar movimientos ──────────────────────────────────────
        //    d_nueva_tipo se inicializa a 0 (todo vacío) antes de cada escritura.
        //    apply_kernel place vehículos con atomicCAS, resolviendo colisiones.
        CUDA_CHECK(cudaMemset(d_nueva_tipo, 0, num_casillas * sizeof(int)));
        apply_kernel<<<blocks, BLOCK_SIZE>>>(
            d_nueva_tipo, d_nueva_vel,
            d_tipo,
            d_vel_nueva, d_pos_destino,
            d_flujo_total, d_vel_total, d_conteos_vel,
            num_casillas, longitud_carretera,
            detector_pos, medir);
        CUDA_CHECK(cudaDeviceSynchronize());

        // ── Swap de buffers O(1) — equivalente a "carretera = nueva_carretera" ──
        swap(d_tipo,  d_nueva_tipo);
        swap(d_vel,   d_nueva_vel);
    }

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));

    // ── Bajar contadores de GPU a CPU ────────────────────────────────────────
    unsigned long long h_flujo_total = 0, h_vel_total = 0;
    int       h_conteos_vel = 0;
    CUDA_CHECK(cudaMemcpy(&h_flujo_total, d_flujo_total, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_vel_total,   d_vel_total,   sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_conteos_vel, d_conteos_vel, sizeof(int),       cudaMemcpyDeviceToHost));

    // ── Calcular métricas (igual que serial) ─────────────────────────────────
    int    iteraciones_medicion    = num_iteraciones - warmup_iteraciones;
    int    num_vehiculos_real      = (int)(longitud_carretera * num_carriles * densidad_vehicular);
    double flujo_promedio_total    = (iteraciones_medicion > 0)
                                        ? (double)h_flujo_total / iteraciones_medicion : 0.0;
    double flujo_por_carril        = flujo_promedio_total / num_carriles;
    double velocidad_promedio      = (h_conteos_vel > 0)
                                        ? (double)h_vel_total / h_conteos_vel : 0.0;

    // ── Tabla de resultados ──────────────────────────────────────────────────
    cout << fixed << setprecision(4);
    cout << "\n";
    cout << "============================================================\n";
    cout << "          RESULTADOS DE LA SIMULACION  (CUDA)\n";
    cout << "============================================================\n";
    cout << left;
    cout << "  " << setw(42) << "Densidad vehicular"                << ": " << densidad_vehicular    << "\n";
    cout << "  " << setw(42) << "Numero de vehiculos"               << ": " << num_vehiculos_real    << "\n";
    cout << "  " << setw(42) << "Numero de carriles"                << ": " << num_carriles          << "\n";
    cout << "  " << setw(42) << "Longitud carretera (celdas)"       << ": " << longitud_carretera    << "\n";
    cout << "  " << setw(42) << "Velocidad maxima (celdas/iter)"    << ": " << velocidad_max         << "\n";
    cout << "  " << setw(42) << "Iteraciones totales"               << ": " << num_iteraciones       << "\n";
    cout << "  " << setw(42) << "Iteraciones medidas (sin warmup)"  << ": " << iteraciones_medicion  << "\n";
    cout << "  " << setw(42) << "Tiempo de simulacion (s)"          << ": " << (ms / 1000.0f)        << "\n";
    cout << "  " << setw(42) << "Flujo total (veh/iter)"            << ": " << flujo_promedio_total  << "\n";
    cout << "  " << setw(42) << "Flujo por carril (veh/iter/carril)"<< ": " << flujo_por_carril      << "\n";
    cout << "  " << setw(42) << "Velocidad promedio (celdas/iter)"  << ": " << velocidad_promedio    << "\n";
    cout << "  " << setw(42) << "Flujo teorico J=rho*v"             << ": " << (densidad_vehicular * velocidad_promedio) << "\n";
    cout << "============================================================\n";

    // ── Métricas de paralelización ───────────────────────────────────────────
    double t_gpu_s        = ms / 1000.0;
    int    n_threads      = blocks * BLOCK_SIZE;
    int    warps_total    = n_threads / 32;
    int    warps_activos  = (num_casillas + 31) / 32;
    int    n              = blocks;            // número de procesadores (bloques)

    // Ley de Amdahl: dado el speedup medido, despejar fracción serial f_s
    //   Speedup = 1 / ( f_s + (1-f_s)/n )  →  f_s = (1/S - 1/n) / (1 - 1/n)
    double speedup = (t_gpu_s > 0.0) ? (t_cpu / t_gpu_s) : 0.0;
    double inv_n   = 1.0 / (double)n;
    double f_s     = (speedup > 0.0 && speedup < n)
                       ? ((1.0/speedup) - inv_n) / (1.0 - inv_n)
                       : 0.0;
    f_s = max(0.0, min(1.0, f_s));   // acotar a [0,1]

    double amdahl_max = (f_s < 1.0) ? (1.0 / f_s) : speedup;
    double gustafson  = (double)n - f_s * ((double)n - 1.0);

    cout << "\n";
    cout << "============================================================\n";
    cout << "          METRICAS DE PARALELIZACION  (CUDA)\n";
    cout << "============================================================\n";
    cout << "  " << setw(42) << "Semilla cuRAND"                    << ": " << CURAND_SEED       << "\n";
    cout << "  " << setw(42) << "Threads por bloque"                << ": " << BLOCK_SIZE        << "\n";
    cout << "  " << setw(42) << "Bloques lanzados"                  << ": " << blocks            << "\n";
    cout << "  " << setw(42) << "Threads totales lanzados"          << ": " << n_threads         << "\n";
    cout << "  " << setw(42) << "Threads activos (celdas)"          << ": " << num_casillas      << "\n";
    cout << "  " << setw(42) << "Warps totales lanzados"            << ": " << warps_total       << "\n";
    cout << "  " << setw(42) << "Warps activos"                     << ": " << warps_activos     << "\n";
    cout << "  " << setw(42) << "Tiempo CPU serial (s)"             << ": " << t_cpu             << "\n";
    cout << "  " << setw(42) << "Tiempo GPU CUDA (s)"               << ": " << t_gpu_s           << "\n";
    cout << "  " << setw(42) << "Speedup real  (S = T_cpu / T_gpu)" << ": " << speedup           << "\n";
    cout << "  " << setw(42) << "Fraccion serial estimada (f_s)"    << ": " << f_s               << "\n";
    cout << "  " << setw(42) << "Speedup maximo Ley de Amdahl"      << ": " << amdahl_max        << "\n";
    cout << "  " << setw(42) << "Speedup Ley de Gustafson"          << ": " << gustafson         << "\n";
    cout << "============================================================\n";

    // ── Liberar recursos ─────────────────────────────────────────────────────
    cudaFree(d_tipo);        cudaFree(d_vel);
    cudaFree(d_nueva_tipo);  cudaFree(d_nueva_vel);
    cudaFree(d_vel_nueva);   cudaFree(d_pos_destino);
    cudaFree(d_flujo_total); cudaFree(d_vel_total); cudaFree(d_conteos_vel);
    cudaFree(d_states);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    return 0;
}
