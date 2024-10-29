//PARTE 1: PREGUNTAS SOBRE EL CODIGO
/* Preguntas Ejercicio3:
void* funcionHebrada(void* arg){
    tareahilo* task=(tarea_hilo*)arg;
    for(i=task->ini;i<task->end; i++){
        pthread_mutex_lock(&mutex);
        resultadoGlobal += pow(vector[i],5);
        pthread_mutex_unlock(&mutex);
    }
    returns 0;
}

Este fragmento de código parece tener dos problemas principales:

Acceso concurrente no seguro a la variable resultadoGlobal: Varias hebras están intentando modificar la misma variable global resultadoGlobal simultáneamente sin ningún mecanismo de sincronización adecuado, lo que puede provocar condiciones de carrera y resultados incorrectos.
Bloqueo del mutex dentro del bucle de iteración: El bloqueo del mutex se realiza dentro del bucle de iteración, lo que significa que cada hebra bloquea y desbloquea el mutex varias veces durante la ejecución del bucle, lo cual es innecesario y puede causar un rendimiento deficiente.
 
 Solucion:
 
 #include <pthread.h>
 #include <math.h>

 // Definición de estructura para los parámetros del hilo
 typedef struct {
     int ini; // Índice inicial
     int end; // Índice final
     double *vector; // Vector de datos
 } tarea_hilo;

 // Declaración de variables globales
 double resultadoGlobal = 0;
 pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

 // Función hebrada corregida
 void* funcionHebrada(void* arg){
     tarea_hilo* task = (tarea_hilo*)arg;
     double localSum = 0; // Suma local para cada hebra

     // Cálculo de la suma local
     for(int i = task->ini; i < task->end; i++) {
         localSum += pow(task->vector[i], 5);
     }

     // Bloqueo del mutex para actualizar el resultado global
     pthread_mutex_lock(&mutex);
     resultadoGlobal += localSum;
     pthread_mutex_unlock(&mutex);

     // Devolución de NULL ya que la función es de tipo void*
     return NULL;
 }


*/

//PARTE 2: SOLUCION DEL PARALELISMO PARA LA IMAGEN:
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "pgm.h"

#define TAG_IMAGE 0
#define TAG_ASCII 1

char grayscale_to_ascii(unsigned char pixel) {
    if (pixel >= 0 && pixel <= 25) return '#';
    else if (pixel >= 26 && pixel <= 51) return '@';
    else if (pixel >= 52 && pixel <= 77) return '%';
    else if (pixel >= 78 && pixel <= 103) return '+';
    else if (pixel >= 104 && pixel <= 129) return '*';
    else if (pixel >= 130 && pixel <= 155) return '=';
    else if (pixel >= 156 && pixel <= 181) return ':';
    else if (pixel >= 182 && pixel <= 207) return '-';
    else if (pixel >= 208 && pixel <= 233) return '.';
    else return ' ';  // espaciado
}

void toASCII(unsigned char** Original, int rows, int cols, char** Salida, int window_size) {
    for (int i = 0; i < rows; i += window_size) {
        for (int j = 0; j < cols; j += window_size) {
            int sum = 0, count = 0;
            for (int wi = 0; wi < window_size && i + wi < rows; ++wi) {
                for (int wj = 0; wj < window_size && j + wj < cols; ++wj) {
                    sum += Original[i + wi][j + wj];
                    count++;
                }
            }
            unsigned char avg = sum / count;
            Salida[i / window_size][j / window_size] = grayscale_to_ascii(avg);
        }
    }
}

void toDisk(char** Salida, int rows, int cols) {
    FILE* file = fopen("output.txt", "w");
    if (file == NULL) {
        printf("Error al abrir el archivo de salida.\n");
        exit(1);
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(file, "%c", Salida[i][j]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

int main(int argc, char** argv) {
    int rank, size, rows, cols, window_size;
    unsigned char** Original = NULL;
    char** Salida = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) {
            printf("Uso: %s <archivo.pgm> <tamaño de ventana>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    window_size = atoi(argv[2]);
    if (window_size < 1 || window_size > 3) {
        if (rank == 0) {
            printf("El tamaño de ventana debe ser entre 1 y 3.\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        Original = pgmread(argv[1], &rows, &cols);
        Salida = (char**)GetMem2D((rows + window_size - 1) / window_size, (cols + window_size - 1) / window_size, sizeof(char));
        printf("Proceso %d: Imagen leída: %d filas, %d columnas\n", rank, rows, cols);
        printf("Proceso %d: Memoria asignada para Salida\n", rank);
    }

    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_rows = (rows / size + window_size - 1) / window_size * window_size;
    unsigned char** local_Original = (unsigned char**)GetMem2D(local_rows, cols, sizeof(unsigned char));

    MPI_Scatter(&Original[0][0], local_rows * cols, MPI_UNSIGNED_CHAR, &local_Original[0][0], local_rows * cols, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    char** local_Salida = (char**)GetMem2D(local_rows / window_size, (cols + window_size - 1) / window_size, sizeof(char));
    toASCII(local_Original, local_rows, cols, local_Salida, window_size);

    char* gathered_data = (char*)malloc((rows * cols / (window_size * window_size)) * sizeof(char));
    MPI_Gather(&local_Salida[0][0], (local_rows * cols / (window_size * window_size)), MPI_CHAR, gathered_data, (local_rows * cols / (window_size * window_size)), MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < rows / window_size; i++) {
            for (int j = 0; j < cols / window_size; j++) {
                Salida[i][j] = gathered_data[i * (cols / window_size) + j];
            }
        }
        toDisk(Salida, (rows + window_size - 1) / window_size, (cols + window_size - 1) / window_size);
        Free2D((void**)Original);
        Free2D((void**)Salida);
    }

    free(gathered_data);
    MPI_Finalize();
    return 0;
}
