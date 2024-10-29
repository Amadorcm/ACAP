
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

// Función para calcular el coeficiente de Tanimoto
double calcularTanimoto(int* conjuntoA, int tamA, int* conjuntoB, int tamB) {
    int interseccion = 0;
    int union_set = 0;

    // Calcular intersección
    for (int i = 0; i < tamA; i++) {
        for (int j = 0; j < tamB; j++) {
            if (conjuntoA[i] == conjuntoB[j]) {
                interseccion++;
                break;
            }
        }
    }

    // Calcular unión
    union_set = tamA + tamB - interseccion;

    // Calcular Tanimoto
    double tanimoto = (double)interseccion / (double)union_set;
    return tanimoto;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Uso: %s tamA tamB\n", argv[0]);
        return 1;
    }

    int tamA = atoi(argv[1]);
    int tamB = atoi(argv[2]);

    // Generar conjuntos aleatorios (supondremos que los conjuntos son conjuntos de enteros distintos)
    int* conjuntoA = (int*)malloc(tamA * sizeof(int));
    int* conjuntoB = (int*)malloc(tamB * sizeof(int));

    for (int i = 0; i < tamA; i++) {
        conjuntoA[i] = rand() % 100;  // Números aleatorios entre 0 y 99
    }

    for (int i = 0; i < tamB; i++) {
        conjuntoB[i] = rand() % 100;  // Números aleatorios entre 0 y 99
    }

    // Medir tiempo de ejecución
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Calcular coeficiente de Tanimoto
    double tanimoto = calcularTanimoto(conjuntoA, tamA, conjuntoB, tamB);

    gettimeofday(&end, NULL);
    double elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0; // Convertir segundos a milisegundos
    elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;    // Convertir microsegundos a milisegundos

    // Mostrar resultados
    printf("Coeficiente de Tanimoto: %lf\n", tanimoto);
    printf("Tiempo de ejecución: %f ms\n", elapsedTime);

    // Liberar memoria
    free(conjuntoA);
    free(conjuntoB);

    return 0;
}
