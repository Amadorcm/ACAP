#include <mpi.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define MIN_SIZE_TO_DISPLAY 7

// Estructura para pasar los datos a los hilos
typedef struct {
    int thread_id;
    int num_threads;
    int rows_a;
    int cols_a;
    int cols_b;
    double *A;
    double *B;
    double *C;
} thread_data_t;

void *thread_matrix_multiply(void *arg) {
    thread_data_t *data = (thread_data_t *)arg;
    int tid = data->thread_id;
    int num_threads = data->num_threads;
    int rows_a = data->rows_a;
    int cols_a = data->cols_a;
    int cols_b = data->cols_b;
    double *A = data->A;
    double *B = data->B;
    double *C = data->C;

    for (int i = tid; i < rows_a; i += num_threads) {
        for (int j = 0; j < cols_b; j++) {
            C[i * cols_b + j] = 0.0;
            for (int k = 0; k < cols_a; k++) {
                C[i * cols_b + j] += A[i * cols_a + k] * B[k * cols_b + j];
            }
        }
    }
    pthread_exit(NULL);
}

void initialize_matrix(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (double)rand() / RAND_MAX * 10.0;
    }
}

void print_matrix(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    int rank, size, num_threads;
    int rows_a, cols_a, cols_b;
    double *A = NULL, *B = NULL, *C = NULL, *local_A = NULL, *local_C = NULL;
    double start_time, end_time;
    int rows_per_proc, extra_rows, local_rows;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        if (argc != 5) {
            fprintf(stderr, "Uso: %s <filas_a> <columnas_a> <columnas_b> <num_hilos>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        rows_a = atoi(argv[1]);
        cols_a = atoi(argv[2]);
        cols_b = atoi(argv[3]);
        num_threads = atoi(argv[4]);

        A = (double *)malloc(rows_a * cols_a * sizeof(double));
        B = (double *)malloc(cols_a * cols_b * sizeof(double));
        C = (double *)malloc(rows_a * cols_b * sizeof(double));
        if (!A || !B || !C) {
            fprintf(stderr, "Error al asignar memoria\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        srand(time(NULL));
        initialize_matrix(A, rows_a, cols_a);
        initialize_matrix(B, cols_a, cols_b);

        start_time = MPI_Wtime();
    }

    // Transmitir las dimensiones de las matrices y el número de hilos a todos los procesos
    MPI_Bcast(&rows_a, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols_a, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols_b, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_threads, 1, MPI_INT, 0, MPI_COMM_WORLD);

    rows_per_proc = rows_a / size;
    extra_rows = rows_a % size;
    local_rows = (rank < extra_rows) ? (rows_per_proc + 1) : rows_per_proc;

    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            int dest_rows = (i < extra_rows) ? (rows_per_proc + 1) : rows_per_proc;
            int offset = (i < extra_rows) ? i * (rows_per_proc + 1) : (extra_rows * (rows_per_proc + 1)) + ((i - extra_rows) * rows_per_proc);
            MPI_Send(A + offset * cols_a, dest_rows * cols_a, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
        local_A = (double *)malloc(local_rows * cols_a * sizeof(double));
        local_C = (double *)malloc(local_rows * cols_b * sizeof(double));
        memcpy(local_A, A, local_rows * cols_a * sizeof(double));
    } else {
        local_A = (double *)malloc(local_rows * cols_a * sizeof(double));
        local_C = (double *)malloc(local_rows * cols_b * sizeof(double));
        MPI_Recv(local_A, local_rows * cols_a, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Transmitir la matriz B a todos los procesos
    if (rank == 0) {
        MPI_Bcast(B, cols_a * cols_b, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        B = (double *)malloc(cols_a * cols_b * sizeof(double));
        MPI_Bcast(B, cols_a * cols_b, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    thread_data_t *thread_data = (thread_data_t *)malloc(num_threads * sizeof(thread_data_t));
    for (int t = 0; t < num_threads; t++) {
        thread_data[t].thread_id = t;
        thread_data[t].num_threads = num_threads;
        thread_data[t].rows_a = local_rows;
        thread_data[t].cols_a = cols_a;
        thread_data[t].cols_b = cols_b;
        thread_data[t].A = local_A;
        thread_data[t].B = B;
        thread_data[t].C = local_C;
        pthread_create(&threads[t], NULL, thread_matrix_multiply, (void *)&thread_data[t]);
    }

    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }

    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            int source_rows = (i < extra_rows) ? (rows_per_proc + 1) : rows_per_proc;
            int offset = (i < extra_rows) ? i * (rows_per_proc + 1) : (extra_rows * (rows_per_proc + 1)) + ((i - extra_rows) * rows_per_proc);
            MPI_Recv(C + offset * cols_b, source_rows * cols_b, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        memcpy(C, local_C, local_rows * cols_b * sizeof(double));
    } else {
        MPI_Send(local_C, local_rows * cols_b, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        end_time = MPI_Wtime();
        printf("Tiempo total: %f segundos\n", end_time - start_time);

        if (rows_a <= MIN_SIZE_TO_DISPLAY && cols_a <= MIN_SIZE_TO_DISPLAY && cols_b <= MIN_SIZE_TO_DISPLAY) {
            printf("Matriz A:\n");
            print_matrix(A, rows_a, cols_a);
            printf("Matriz B:\n");
            print_matrix(B, cols_a, cols_b);
            printf("Matriz C (Resultado):\n");
            print_matrix(C, rows_a, cols_b);
        }

        free(A);
        free(C);
    }

    free(local_A);
    free(local_C);
    free(B);
    free(threads);
    free(thread_data);

    MPI_Finalize();
    return 0;
}
