#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <ctype.h>

#define COMANDO_SALIR 0
#define COMANDO_GENERAR_VECTOR_ALEATORIO 1
#define COMANDO_SUMA_ASCII 2
#define COMANDO_CONVERTIR_MAYUSCULAS 3
#define COMANDO_TODAS_ACCIONES 4

void proceso0(int rank, int numProcs) {
    int comando;
    int longitudVector;
    char palabra[100];
    int finalizar = 0; // Bandera para controlar si se debe finalizar

    while (!finalizar) {
        printf("\nIngrese el comando (0 para salir, 1 para vector aleatorio, 2 para suma de ASCII, 3 para convertir a mayúsculas, 4 para todas las acciones): ");
        fflush(stdout);
        scanf("%d", &comando);

        switch (comando) {
            case COMANDO_SALIR:
                MPI_Finalize();
                return 0;
            case COMANDO_GENERAR_VECTOR_ALEATORIO:
                printf("Ingrese la longitud del vector: ");
                fflush(stdout);
                scanf("%d", &longitudVector);
                MPI_Send(&longitudVector, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
                break;
            case COMANDO_SUMA_ASCII:
                printf("Ingrese una palabra: ");
                fflush(stdout);
                scanf("%s", palabra);
                MPI_Send(palabra, strlen(palabra) + 1, MPI_CHAR, 2, 0, MPI_COMM_WORLD);
                break;
            case COMANDO_CONVERTIR_MAYUSCULAS:
                printf("Ingrese una palabra: ");
                fflush(stdout);
                scanf("%s", palabra);
                MPI_Send(palabra, strlen(palabra) + 1, MPI_CHAR, 3, 0, MPI_COMM_WORLD);
                break;
            case COMANDO_TODAS_ACCIONES:
                printf("Ingrese la longitud del vector: ");
                fflush(stdout);
                scanf("%d", &longitudVector);
                MPI_Send(&longitudVector, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
                
                printf("Ingrese una palabra: ");
                fflush(stdout);
                scanf("%s", palabra);
                MPI_Send(palabra, strlen(palabra) + 1, MPI_CHAR, 2, 0, MPI_COMM_WORLD);
                MPI_Send(palabra, strlen(palabra) + 1, MPI_CHAR, 3, 0, MPI_COMM_WORLD);
                break;
            default:
                printf("Comando no reconocido.\n");
                break;
        }
    }

    // Informa a los otros procesos que deben finalizar
    MPI_Send(&comando, 0, MPI_INT, 1, 0, MPI_COMM_WORLD);
    MPI_Send(&comando, 0, MPI_INT, 2, 0, MPI_COMM_WORLD);
    MPI_Send(&comando, 0, MPI_INT, 3, 0, MPI_COMM_WORLD);
    MPI_Finalize(); // Cierra MPI
    exit(0); // Sale del programa
}

void proceso1(int rank) {
    int longitudVector;
    int i;
    int finalizar = 0; // Bandera para controlar si se debe finalizar

    while (!finalizar) {
        MPI_Status status;
        MPI_Recv(&longitudVector, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        if (longitudVector == COMANDO_SALIR) finalizar = 1; // Verifica si se debe finalizar
        else {
            double *vectorAleatorio = (double *)malloc(longitudVector * sizeof(double));
            for (i = 0; i < longitudVector; i++) {
                vectorAleatorio[i] = (double)rand() / RAND_MAX;
                printf("%.2f ", vectorAleatorio[i]);
            }
            printf("\n");
            free(vectorAleatorio);
        }
    }
}

void proceso2(int rank) {
    char palabra[100];
    int suma = 0;
    int i;
    int finalizar = 0; // Bandera para controlar si se debe finalizar

    while (!finalizar) {
        MPI_Status status;
        MPI_Recv(palabra, 100, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
        if (strcmp(palabra, "0") == 0) finalizar = 1; // Verifica si se debe finalizar
        else {
            for (i = 0; i < strlen(palabra); i++) {
                suma += (int)palabra[i];
            }
            printf("Suma de valores ASCII: %d\n", suma);
        }
    }
}

void proceso3(int rank) {
    char palabra[100];
    int finalizar = 0; // Bandera para indicar si se debe finalizar

    while (!finalizar) {
        MPI_Status status;
        MPI_Recv(palabra, 100, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);

        // Verifica si la palabra recibida es el comando de salida
        if (strcmp(palabra, "0") == 0) {
            finalizar = 1; // Establece la bandera de finalización en verdadero
        } else {
            // Si no es el comando de salida, convierte la palabra a mayúsculas
            int i;
            for (i = 0; i < strlen(palabra); i++) {
                palabra[i] = toupper(palabra[i]);
            }
            printf("Palabra en mayúsculas: %s\n", palabra);
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, numProcs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    if (numProcs != 4) {
        if (rank == 0) printf("Error: Este programa requiere exactamente 4 procesos.\n");
        MPI_Finalize();
        exit(1);
    }

    switch (rank) {
        case 0:
            proceso0(rank, numProcs);
            break;
        case 1:
            proceso1(rank);
            break;
        case 2:
            proceso2(rank);
            break;
        case 3:
            proceso3(rank);
            break;
    }

    MPI_Finalize();
    return 0;
}
