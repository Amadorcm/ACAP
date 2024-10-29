#include <stdio.h>
#include <mpi.h>

#define N 1000000000  // Número de rectángulos para la aproximación
#define BASE 1.0      // Base de cada rectángulo
#define TAG_RESULT 1  // Etiqueta para enviar resultados parciales

double f(double x) {
    return 4.0 / (1.0 + x * x);
}

int main(int argc, char *argv[]) {
    int rank, numProcs;
    double pi;
    double sum = 0.0;
    double x, delta_x;
    int i;
    double start_time, end_time;  // Variables para medir el tiempo

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    delta_x = BASE / N;

    // Inicia la medición del tiempo
    if (rank == 0) {
        start_time = MPI_Wtime();
    }

    // Cada proceso calcula su parte de la suma
    for (i = rank; i < N; i += numProcs) {
        x = (i + 0.5) * delta_x;
        sum += f(x);
    }

    // Se envían los resultados parciales al proceso 0
    if (rank != 0) {
        MPI_Send(&sum, 1, MPI_DOUBLE, 0, TAG_RESULT, MPI_COMM_WORLD);
    } else {
        // El proceso 0 suma los resultados parciales y calcula pi
        double partialSum;
        for (i = 1; i < numProcs; i++) {
            MPI_Recv(&partialSum, 1, MPI_DOUBLE, i, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            sum += partialSum;
        }
        pi = sum * delta_x;

        // Finaliza la medición del tiempo
        end_time = MPI_Wtime();

        printf("Aproximación de Pi: %.15f\n", pi);
        printf("Tiempo empleado: %.6f segundos\n", end_time - start_time);
    }

    MPI_Finalize();

    return 0;
}
