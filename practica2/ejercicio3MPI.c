#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <mpi.h>

// Función para calcular el coeficiente de Tanimoto localmente en cada proceso
double calcularTanimotoLocal(int* conjuntoA, int tamA_local, int* conjuntoB, int tamB_local) {
    int interseccion = 0;
    int union_set = 0;

    // Calcular intersección local
    for (int i = 0; i < tamA_local; i++) {
        for (int j = 0; j < tamB_local; j++) {
            if (conjuntoA[i] == conjuntoB[j]) {
                interseccion++;
                break;
            }
        }
    }

    // Calcular unión local
    union_set = tamA_local + tamB_local - interseccion;

    // Calcular Tanimoto local
    double tanimoto = (double)interseccion / (double)union_set;
    return tanimoto;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) {
            printf("Uso: mpirun -np <num_procs> %s tamA tamB\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int tamA = atoi(argv[1]);
    int tamB = atoi(argv[2]);

    // Generar conjuntos aleatorios solo en el proceso 0
    int* conjuntoA = NULL;
    int* conjuntoB = NULL;
    int tamA_local = 0;
    int tamB_local = 0;

    if (rank == 0) {
        conjuntoA = (int*)malloc(tamA * sizeof(int));
        conjuntoB = (int*)malloc(tamB * sizeof(int));

        for (int i = 0; i < tamA; i++) {
            conjuntoA[i] = rand() % 100;  // Números aleatorios entre 0 y 99
        }

        for (int i = 0; i < tamB; i++) {
            conjuntoB[i] = rand() % 100;  // Números aleatorios entre 0 y 99
        }
    }

    // Broadcast los tamaños de los conjuntos
    MPI_Bcast(&tamA, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tamB, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calcular tamaños locales de los conjuntos
    tamA_local = tamA / size;
    tamB_local = tamB / size;

    // Asignar memoria para los conjuntos locales
    int* conjuntoA_local = (int*)malloc(tamA_local * sizeof(int));
    int* conjuntoB_local = (int*)malloc(tamB_local * sizeof(int));

    // Scatter los conjuntos
    MPI_Scatter(conjuntoA, tamA_local, MPI_INT, conjuntoA_local, tamA_local, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(conjuntoB, tamB_local, MPI_INT, conjuntoB_local, tamB_local, MPI_INT, 0, MPI_COMM_WORLD);

    // Medir tiempo de ejecución local
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Calcular coeficiente de Tanimoto localmente en cada proceso
    double tanimoto_local = calcularTanimotoLocal(conjuntoA_local, tamA_local, conjuntoB_local, tamB_local);

    gettimeofday(&end, NULL);
    double elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0; // Convertir segundos a milisegundos
    elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;    // Convertir microsegundos a milisegundos

    // Reducir los coeficientes de Tanimoto locales para obtener el resultado global
    double tanimoto_global;
    MPI_Reduce(&tanimoto_local, &tanimoto_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Mostrar resultados en el proceso 0
    if (rank == 0) {
        printf("Coeficiente de Tanimoto: %lf\n", tanimoto_global / size); // Dividir por el número de procesos para obtener el promedio
        printf("Tiempo de ejecución: %f ms\n", elapsedTime);
    }

    // Liberar memoria
    if (rank == 0) {
        free(conjuntoA);
        free(conjuntoB);
    }

    free(conjuntoA_local);
    free(conjuntoB_local);

    MPI_Finalize();
    return 0;
}
