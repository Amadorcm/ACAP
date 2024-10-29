#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

double piLeibniz(int steps, int rank, int size) {
    double partpi = 0.0;
    double num = 1.0;
    double denom = 1.0;
    int start = rank * (steps / size);
    int end = (rank + 1) * (steps / size);
    if (rank == size - 1) {
        end = steps; // El último proceso maneja el resto de las iteraciones
    }
    for (int i = start; i < end; i++) {
        partpi += num / denom;
        num = -1.0 * num; // Alternamos el signo
        denom += 2.0;
    }
    return 4.0 * partpi;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2 || rank != 0) { // Solo el proceso 0 maneja los argumentos de entrada
        if (rank == 0) {
            printf("Uso: mpirun -np <num_procs> %s esfuerzo\n", argv[0]);
        }
    } else {
        int steps = atoi(argv[1]);
        if (steps <= 0) {
            if (rank == 0) {
                printf("El número de iteraciones debe ser un entero positivo!\n");
            }
        } else {
            double pi = piLeibniz(steps, rank, size);

            double total_pi;
            MPI_Reduce(&pi, &total_pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                printf("PI por la serie de G. Leibniz [%d iteraciones] =\t%lf\n", steps, total_pi);
            }
        }
    }

    MPI_Finalize();
    return 0;
}

