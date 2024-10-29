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

void toASCII(unsigned char** Original, int rows, int cols, char** Salida) {
    printf("Entrando a toASCII\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Salida[i][j] = grayscale_to_ascii(Original[i][j]);
        }
    }
    printf("Saliendo de toASCII\n");
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
    int rank, size, rows, cols;
    unsigned char** Original = NULL;
    char** Salida = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) {
            printf("Uso: %s <archivo.pgm>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        Original = pgmread(argv[1], &rows, &cols);
        printf("Proceso %d: Imagen leída: %d filas, %d columnas\n", rank, rows, cols);
        Salida = (char**)GetMem2D(rows, cols, sizeof(char));
        printf("Proceso %d: Memoria asignada para Salida\n", rank);
        toASCII(Original, rows, cols, Salida);
        printf("Proceso %d: Imagen convertida a ASCII\n", rank);
    }

    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Proceso %d: Broadcast de filas\n", rank);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Proceso %d: Broadcast de columnas\n", rank);

    int local_rows = rows / size;
    unsigned char** local_Original = (unsigned char**)GetMem2D(local_rows, cols, sizeof(unsigned char));
    printf("Proceso %d: Memoria asignada para local_Original\n", rank);
    MPI_Scatter(&Original[0][0], local_rows * cols, MPI_UNSIGNED_CHAR, &local_Original[0][0], local_rows * cols, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    printf("Proceso %d: Datos dispersados\n", rank);

    char** local_Salida = (char**)GetMem2D(local_rows, cols, sizeof(char));
    printf("Proceso %d: Memoria asignada para local_Salida\n", rank);
    toASCII(local_Original, local_rows, cols, local_Salida);
    printf("Proceso %d: Conversión local completada\n", rank);

    char* gathered_data = (char*)malloc(rows * cols * sizeof(char)); // Buffer para almacenar todos los datos de ASCII
    printf("Proceso %d: Memoria asignada para gathered_data\n", rank);
    MPI_Gather(&local_Salida[0][0], local_rows * cols, MPI_CHAR, gathered_data, local_rows * cols, MPI_CHAR, 0, MPI_COMM_WORLD);
    printf("Proceso %d: Datos recolectados\n", rank);

    if (rank == 0) {
        // Reconstruir matriz ASCII
        Salida = (char**)GetMem2D(rows, cols, sizeof(char));
        printf("Proceso %d: Memoria asignada para Salida (reconstrucción)\n", rank);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                Salida[i][j] = gathered_data[i * cols + j];
            }
        }
        printf("Proceso %d: Matriz ASCII reconstruida\n", rank);
        toDisk(Salida, rows, cols);
        printf("Proceso %d: Datos escritos en el disco\n", rank);
        Free2D((void**)Original);
        Free2D((void**)Salida);
        printf("Proceso %d: Memoria liberada\n", rank);
    }

    free(gathered_data);
    MPI_Finalize();
    return 0;
}

