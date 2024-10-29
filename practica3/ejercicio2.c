#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define MAX_THREADS 16

// Estructura para pasar argumentos a los hilos
typedef struct {
    int thread_id;
    int num_threads;
    int vector_size;
    double *vector1;
    double *vector2;
    double *result;
} ThreadArgs;

// Función para llenar un vector con valores aleatorios
void fill_vector(double *vector, int size) {
    for (int i = 0; i < size; ++i) {
        vector[i] = (double)rand() / RAND_MAX * 100; // Valores aleatorios en [0, 100)
    }
}

// Función que realiza el producto escalar de una porción de los vectores
void *dot_product(void *args) {
    ThreadArgs *thread_args = (ThreadArgs *)args;
    int chunk_size = thread_args->vector_size / thread_args->num_threads;
    int start = thread_args->thread_id * chunk_size;
    int end = start + chunk_size;
    
    // Asegurarse de que el último hilo tome los elementos restantes si el tamaño no es divisible por el número de hilos
    if (thread_args->thread_id == thread_args->num_threads - 1) {
        end = thread_args->vector_size;
    }

    double local_result = 0.0;
    for (int i = start; i < end; ++i) {
        local_result += thread_args->vector1[i] * thread_args->vector2[i];
    }

    // Almacenar el resultado local en el arreglo de resultados
    thread_args->result[thread_args->thread_id] = local_result;

    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Uso: %s <tamaño del vector> <número de hilos>\n", argv[0]);
        return 1;
    }

    int vector_size = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    if (vector_size <= 0 || num_threads <= 0 || num_threads > MAX_THREADS) {
        printf("Error: Tamaño del vector o número de hilos no válido.\n");
        return 1;
    }

    // Reservar memoria para los vectores y resultados
    double *vector1 = (double *)malloc(vector_size * sizeof(double));
    double *vector2 = (double *)malloc(vector_size * sizeof(double));
    double *result = (double *)malloc(num_threads * sizeof(double));

    // Llenar los vectores con valores aleatorios
    fill_vector(vector1, vector_size);
    fill_vector(vector2, vector_size);

    // Crear los hilos
    pthread_t threads[num_threads];
    ThreadArgs thread_args[num_threads];

    for (int i = 0; i < num_threads; ++i) {
        thread_args[i].thread_id = i;
        thread_args[i].num_threads = num_threads;
        thread_args[i].vector_size = vector_size;
        thread_args[i].vector1 = vector1;
        thread_args[i].vector2 = vector2;
        thread_args[i].result = result;
        pthread_create(&threads[i], NULL, dot_product, (void *)&thread_args[i]);
    }

    // Esperar a que los hilos terminen
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }

    // Calcular el producto escalar final sumando los resultados de los hilos
    double final_result = 0.0;
    for (int i = 0; i < num_threads; ++i) {
        final_result += result[i];
    }

    // Imprimir los resultados si el tamaño del vector es menor que 10
    if (vector_size < 10) {
        printf("Vector 1: ");
        for (int i = 0; i < vector_size; ++i) {
            printf("%.2f ", vector1[i]);
        }
        printf("\n");

        printf("Vector 2: ");
        for (int i = 0; i < vector_size; ++i) {
            printf("%.2f ", vector2[i]);
        }
        printf("\n");

        printf("Producto escalar: %.2f\n", final_result);
    }

    // Realizar la comprobación secuencial
    double sequential_result = 0.0;
    for (int i = 0; i < vector_size; ++i) {
        sequential_result += vector1[i] * vector2[i];
    }

    // Imprimir el resultado de la comprobación secuencial
    printf("Resultado de la comprobación secuencial: %.2f\n", sequential_result);

    // Liberar memoria
    free(vector1);
    free(vector2);
    free(result);

    return 0;
}
/*
 No utilizo exclusión mutua
 En la siguiente diseño uno con exclusión mutua
 
 
 // Función que realiza el producto escalar de una porción de los vectores
 void *dot_product(void *args) {
     ThreadArgs *thread_args = (ThreadArgs *)args;
     int chunk_size = thread_args->vector_size / thread_args->num_threads;
     int start = thread_args->thread_id * chunk_size;
     int end = start + chunk_size;
     
     // Asegurarse de que el último hilo tome los elementos restantes si el tamaño no es divisible por el número de hilos
     if (thread_args->thread_id == thread_args->num_threads - 1) {
         end = thread_args->vector_size;
     }

     double local_result = 0.0;
     for (int i = start; i < end; ++i) {
         local_result += thread_args->vector1[i] * thread_args->vector2[i];
     }

     // Bloquear antes de actualizar el resultado compartido
     pthread_mutex_lock(thread_args->mutex);
     thread_args->result[thread_args->thread_id] = local_result;
     pthread_mutex_unlock(thread_args->mutex);

     pthread_exit(NULL);
 }
 
 
 **/
