/*
******************************************************************************
                    SIMULADOR DE TRAFICO VEHICULAR
                    (Modelo Nagel-Schreckenberg)
******************************************************************************

COMPILACION:
    g++ -O2 -o AUserialV2 AUserialV2.cpp

EJECUCION:
    ./AUserialV2 <num_carriles> <longitud_carretera> <densidad_vehicular> <num_iteraciones>

ARGUMENTOS:
    - num_carriles:         Número de carriles (1-4)
    - longitud_carretera:   Longitud de cada carril en celdas (> 0)
    - densidad_vehicular:   Proporción inicial de vehículos (0.1 - 1.0)
    - num_iteraciones:      Número de pasos de simulación (> 0)

EJEMPLOS:
    ./AUserialV2 2 100 0.3 1000
    Simula 2 carriles, 100 celdas cada uno, 30% de densidad, 1000 iteraciones

    ./AUserialV2 4 200 0.5 5000
    Simula 4 carriles, 200 celdas cada uno, 50% de densidad, 5000 iteraciones

SALIDA:
    Tabla con métricas de la simulación:
    - Densidad y número de vehículos
    - Flujo promedio (vehículos/iteración)
    - Velocidad promedio (celdas/iteración)
    - Comparación con flujo teórico (J = ρ × v)
    - Tiempo de ejecución
******************************************************************************
*/

#include <iostream>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <iterator>
#include <vector>
#include <chrono>
#include <iomanip>

// Variables globales
int velocidad_max = 5;
int num_carriles;
int longitud_carretera;
double densidad_vehicular;
int num_iteraciones;
int num_casillas;
using namespace std;

mt19937 gen(19); // Semilla fija para reproducibilidad
uniform_real_distribution<double> distrib(0.0, 1.0);

// Probabilidades
double prob_cambio_carril = 0.5;      // Probabilidad de que un vehículo intente cambiar de carril
double prob_desaceleracion_max = 0.5; // Si su V=0, esta es su prob de que no arranque de nuevo
double prob_desaceleracion_min = 0.3; // Si su V>0, esta es su prob de que reduzca su velocidad

struct Casilla
{
    char tipo = 'E';
    int velocidad = 0;
};

void validar_argumentos(int argc, char *argv[])
{
    if (argc < 5)
    {
        cout << "Se requieren 4 argumentos:  num_carriles, longitud_carretera, densidad_vehicular y num_iteraciones" << endl;
        exit(1);
    }

    num_carriles = atoi(argv[1]);
    longitud_carretera = atoi(argv[2]);
    densidad_vehicular = atof(argv[3]);
    num_iteraciones = atoi(argv[4]);

    if (!((num_iteraciones > 0) && (num_carriles >= 1 && num_carriles <= 4) && (longitud_carretera > 0) && (densidad_vehicular >= 0.1 && densidad_vehicular <= 1.0)))
    {
        cout << "Error: num_iteraciones debe ser un número positivo, num_carriles entre 1-4, longitud_carretera mayor a 0, y densidad_vehicular entre 0.1 y 1.0" << endl;
        exit(1);
    }
}

void llenar_carretera(vector<Casilla> &carretera)
{
    num_casillas = longitud_carretera * num_carriles;
    int num_vehiculos = (int)(longitud_carretera * num_carriles * densidad_vehicular);

    uniform_int_distribution<int> dist_pos(0, num_casillas - 1);
    uniform_int_distribution<int> dist_vel(0, velocidad_max);

    int colocados = 0;
    int intentos = 0;
    int max_intentos = num_casillas * 4;
    while (colocados < num_vehiculos && intentos < max_intentos)
    {
        int posicion = dist_pos(gen);
        if (carretera[posicion].tipo == 'E')
        {
            carretera[posicion].tipo = 'V';
            carretera[posicion].velocidad = dist_vel(gen);
            colocados++;
        }
        intentos++;
    }
}

int brechaDisponible(const vector<Casilla> &carretera, const vector<Casilla> &nueva_carretera, int pos)
{
    int carril = pos / longitud_carretera;        // Determinar el carril actual
    int pos_en_carril = pos % longitud_carretera; // Posición dentro del carril

    // Solo mira hacia delante en el mismo carril, no considera cambios de carril para esta función
    for (int i = 1; i <= velocidad_max; i++)
    {
        int siguiente_en_carril = (pos_en_carril + i) % longitud_carretera;    // Se tiene en cuenta el warp-araound de la carretera
        int pos_siguiente = carril * longitud_carretera + siguiente_en_carril; // acceso row-major a la posición siguiente en el mismo carril

        if (carretera[pos_siguiente].tipo == 'V' || nueva_carretera[pos_siguiente].tipo == 'V')
        {
            return i - 1;
        }
    }
    return velocidad_max;
}

int carrilAlternativo(const vector<Casilla> &carretera, const vector<Casilla> &nueva_carretera, int pos, int velocidad_nueva)
{
    int pos_en_carril = pos % longitud_carretera; // Posición dentro del carril actual

    if (distrib(gen) >= prob_cambio_carril) // No se cumple la probabilidad de cambio de carril, se mantiene en el mismo carril
    {
        return pos / longitud_carretera;
    }
    // Si se cumple la probabilidad de cambio de carril, se evalúan las opciones de cambio a la izquierda o derecha
    int carril_actual = pos / longitud_carretera;
    bool puede_izq = (carril_actual > 0);
    bool puede_der = (carril_actual < num_carriles - 1);

    int destino_en_carril = (pos_en_carril + velocidad_nueva) % longitud_carretera;
    // Revisa carril superior e inferior para verificar que estén disponibles en la posición destino
    bool izq_disponible = puede_izq && carretera[(carril_actual - 1) * longitud_carretera + destino_en_carril].tipo == 'E' && nueva_carretera[(carril_actual - 1) * longitud_carretera + destino_en_carril].tipo == 'E';
    bool der_disponible = puede_der && carretera[(carril_actual + 1) * longitud_carretera + destino_en_carril].tipo == 'E' && nueva_carretera[(carril_actual + 1) * longitud_carretera + destino_en_carril].tipo == 'E';

    if (izq_disponible && !der_disponible)
        return carril_actual - 1;
    if (der_disponible && !izq_disponible)
        return carril_actual + 1;
    if (izq_disponible && der_disponible)
        return (distrib(gen) < 0.5) ? (carril_actual - 1) : (carril_actual + 1);

    return carril_actual;
}

int avanzar_carretera(vector<Casilla> &carretera, vector<Casilla> &nueva_carretera, int carril_destino, int pos, int velocidad_nueva)
{
    int pos_en_carril = pos % longitud_carretera;
    int nueva_pos_en_carril = (pos_en_carril + velocidad_nueva) % longitud_carretera;
    int nueva_pos = carril_destino * longitud_carretera + nueva_pos_en_carril;

    // Ya se consideró la disponibilidad de casilla en carril actual, en la función brechaDisponible
    // Se verifica que la casilla destino en el carril nuevo esté disponible
    if (nueva_carretera[nueva_pos].tipo == 'E')
    {
        nueva_carretera[pos].tipo = 'E'; // Limpiar posición anterior
        nueva_carretera[nueva_pos].tipo = 'V';
        nueva_carretera[nueva_pos].velocidad = velocidad_nueva;
        return nueva_pos;
    }
    else
    {
        // Si la casilla destino no está disponible, el vehículo se mantiene en su posición actual pero con la nueva velocidad calculada (que podría ser 0 si se frenó)
        nueva_carretera[pos].velocidad = velocidad_nueva;
        return -1;
    }
}

int main(int argc, char *argv[])
{
    // Validar argumentos de entrada
    validar_argumentos(argc, argv);

    // Inicializar carretera
    num_casillas = longitud_carretera * num_carriles;
    vector<Casilla> carretera(num_casillas);
    llenar_carretera(carretera);

    vector<Casilla> nueva_carretera;
    nueva_carretera = carretera; // Copia la carretera actual para actualizarla simultáneamente

    // Variables para medición de flujo y rendimiento
    const int detector_pos = longitud_carretera / 2; // Detector virtual en el punto medio de la carretera
    long long flujo_total = 0;       // Vehículos que cruzan el detector (post-warmup)
    long long velocidad_total = 0;   // Suma de velocidades para promedio (post-warmup)
    int conteos_velocidad = 0;       // Número de muestras de velocidad (post-warmup)
    const int warmup_iteraciones = max(1, num_iteraciones / 10); // 10% de iteraciones para estado estacionario

    // Cronómetro: solo mide el ciclo de simulación (sin inicialización)
    auto inicio_simulacion = chrono::high_resolution_clock::now();

    for (int iteracion = 0; iteracion < num_iteraciones; iteracion++)
    {
        for (int pos = 0; pos < num_casillas; pos++)
        {
            if (carretera[pos].tipo == 'V')
            {
                // PRIMERA REGLA

                int velocidad_nueva = min(carretera[pos].velocidad + 1, velocidad_max); // Acelerar

                // SEGUNDA REGLA

                int brecha = brechaDisponible(carretera, nueva_carretera, pos);
                velocidad_nueva = min(velocidad_nueva, brecha); // Reducir velocidad si la brecha es menor

                // TERCERA REGLA: Frenados y dificultades para arrancar

                // Mayor dificultad en arrancar si su velocidad actual es 0 (Regla de mataestabilidad)
                if (carretera[pos].velocidad == 0)
                {
                    // Probabilidad de arrancar con dificultad
                    if (distrib(gen) <= prob_desaceleracion_max)
                    {
                        if (velocidad_nueva > 0)
                        {
                            velocidad_nueva -= 1; // Reduce la velocidad en 1, pero no por debajo de 0
                        };
                    }
                } // Regla base de nagel para disminuir velocidad, con prob de desaceleración mínima
                else if (distrib(gen) <= prob_desaceleracion_min)
                {
                    velocidad_nueva = max(velocidad_nueva - 1, 0); // Reduce la velocidad en 1, pero no por debajo de 0
                }

                // CUARTA REGLA: Movimiento del vehículo (con posible cambio de carril si brecha == 0)
                int carril_destino;
                if (brecha == 0)
                    carril_destino = carrilAlternativo(carretera, nueva_carretera, pos, velocidad_nueva);
                else
                    carril_destino = pos / longitud_carretera;

                int resultado = avanzar_carretera(carretera, nueva_carretera, carril_destino, pos, velocidad_nueva);

                // Medición de flujo: solo después del warm-up
                if (iteracion >= warmup_iteraciones)
                {
                    // Conteo de vehículos que cruzan el detector
                    if (resultado >= 0)
                    {
                        int x_old = pos % longitud_carretera;
                        int x_new = resultado % longitud_carretera;
                        // Detección de cruce con wrap-around
                        if (x_old < x_new)
                        {
                            if (x_old < detector_pos && detector_pos <= x_new)
                                flujo_total++;
                        }
                        else if (x_old > x_new) // Solo wrap-around real, no lane changes con v=0
                        {
                            if (x_old < detector_pos || detector_pos <= x_new)
                                flujo_total++;
                        }
                    }
                    velocidad_total += velocidad_nueva;
                    conteos_velocidad++;
                }
            }
        }

        carretera = nueva_carretera;
    }

    auto fin_simulacion = chrono::high_resolution_clock::now();
    chrono::duration<double> duracion = fin_simulacion - inicio_simulacion;

    // Cálculo de métricas finales
    int iteraciones_medicion = num_iteraciones - warmup_iteraciones;
    int num_vehiculos_real = (int)(longitud_carretera * num_carriles * densidad_vehicular);
    double flujo_promedio_total = (iteraciones_medicion > 0) ? (double)flujo_total / iteraciones_medicion : 0.0;
    double flujo_por_carril = flujo_promedio_total / num_carriles;
    double velocidad_promedio = (conteos_velocidad > 0) ? (double)velocidad_total / conteos_velocidad : 0.0;

    // Tabla de resultados
    cout << fixed << setprecision(4);
    cout << "\n";
    cout << "============================================================\n";
    cout << "              RESULTADOS DE LA SIMULACION\n";
    cout << "============================================================\n";
    cout << left;
    cout << "  " << setw(42) << "Densidad vehicular" << ": " << densidad_vehicular << "\n";
    cout << "  " << setw(42) << "Numero de vehiculos" << ": " << num_vehiculos_real << "\n";
    cout << "  " << setw(42) << "Numero de carriles" << ": " << num_carriles << "\n";
    cout << "  " << setw(42) << "Longitud carretera (celdas)" << ": " << longitud_carretera << "\n";
    cout << "  " << setw(42) << "Velocidad maxima (celdas/iter)" << ": " << velocidad_max << "\n";
    cout << "  " << setw(42) << "Iteraciones totales" << ": " << num_iteraciones << "\n";
    cout << "  " << setw(42) << "Iteraciones medidas (sin warmup)" << ": " << iteraciones_medicion << "\n";
    cout << "  " << setw(42) << "Tiempo de simulacion (s)" << ": " << duracion.count() << "\n";
    cout << "  " << setw(42) << "Flujo total (veh/iter)" << ": " << flujo_promedio_total << "\n";
    cout << "  " << setw(42) << "Flujo por carril (veh/iter/carril)" << ": " << flujo_por_carril << "\n";
    cout << "  " << setw(42) << "Velocidad promedio (celdas/iter)" << ": " << velocidad_promedio << "\n";
    cout << "  " << setw(42) << "Flujo teorico J=rho*v" << ": " << (densidad_vehicular * velocidad_promedio) << "\n";
    cout << "============================================================\n";

    return 0;
}