#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

typedef struct {
    int idHilo;
    int numHilos;
    int tipoReparto;
    int* vector1;
    int* vector2;
    int* resultado;
    int tamVector;
    int ultimo;
    pthread_mutex_t* mutex;
} tarea;

void* cuerpoHilo(void* arg) {
    tarea misDeberes = *((tarea*)arg);
    int inicio, fin;
    int sumaLocal = 0;

    if (misDeberes.tipoReparto == 0) { // Ciclico
        printf("Hilo %d accede a {", misDeberes.idHilo);
        for (int i = misDeberes.idHilo; i < misDeberes.tamVector; i += misDeberes.numHilos) {
            sumaLocal += misDeberes.vector1[i] + misDeberes.vector2[i];
            printf("%d", i);
            if (i + misDeberes.numHilos < misDeberes.tamVector)
                printf(", ");
        }
        printf("}\n");
    } else if (misDeberes.tipoReparto == 1) { // Por Bloques
        printf("Hilo %d accede a {", misDeberes.idHilo);
        int bloqueSize = misDeberes.tamVector / misDeberes.numHilos;
        int excedente = misDeberes.tamVector % misDeberes.numHilos;
        // Cálculo del inicio y fin del bloque para este hilo
        inicio = misDeberes.idHilo * bloqueSize;
        fin = inicio + bloqueSize;
        if(misDeberes.idHilo==misDeberes.ultimo-1){
            fin=fin+excedente;
            //printf("Hilo %d con excedente %d\n", misDeberes.idHilo,excedente);
        }
        for (int i = inicio; i < fin; i++) {
            sumaLocal += misDeberes.vector1[i] + misDeberes.vector2[i];
            printf("%d", i);
            if (i != fin - 1)
                printf(", ");
        }
        printf("}\n");
    } else if (misDeberes.tipoReparto == 2) { // Bloques balanceados
        printf("Hilo %d accede a {", misDeberes.idHilo);
        int bloqueSize = misDeberes.tamVector / misDeberes.numHilos;
        int excedente = misDeberes.tamVector % misDeberes.numHilos;
        // Cálculo del inicio y fin del bloque para este hilo
        inicio = misDeberes.idHilo * bloqueSize + MIN(misDeberes.idHilo, excedente);
        fin = inicio + bloqueSize + (misDeberes.idHilo < excedente ? 1 : 0);
        for (int i = inicio; i < fin; i++) {
            sumaLocal += misDeberes.vector1[i] + misDeberes.vector2[i];
            printf("%d", i);
            if (i != fin - 1)
                printf(", ");
            }
        printf("}\n");
    }

    pthread_mutex_lock(misDeberes.mutex);
    *(misDeberes.resultado) += sumaLocal;
    pthread_mutex_unlock(misDeberes.mutex);

    return NULL;
}


void inicializarVector(int** pointToVec, int tam) {
    *pointToVec = malloc(sizeof(int) * tam);
    int* vec = *pointToVec;

    srand(time(NULL));
    for (int i = 0; i < tam; i++) {
        vec[i] = rand() % 100;
    }
}

void mostrarVector(int* vector, int tam) {
    printf("[");
    for (int i = 0; i < tam; i++) {
        printf("%d", vector[i]);
        if (i != tam - 1) printf(", ");
    }
    printf("]\n");
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Uso: ./programa <tamVector> <numHilos> <tipoReparto>\n");
    } else {
        int tamVector = atoi(argv[1]);
        int nHilos = atoi(argv[2]);
        int tipoReparto = atoi(argv[3]);

        int* vector1 = NULL;
        int* vector2 = NULL;
        int* resultado = malloc(sizeof(int));
        *resultado = 0;

        inicializarVector(&vector1, tamVector);
        inicializarVector(&vector2, tamVector);

        pthread_t* hilos = malloc(sizeof(pthread_t) * nHilos);
        pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

        tarea* deberes = malloc(sizeof(tarea) * nHilos);
        for (int i = 0; i < nHilos; i++) {
            deberes[i].idHilo = i;
            deberes[i].numHilos = nHilos;
            deberes[i].tipoReparto = tipoReparto;
            deberes[i].vector1 = vector1;
            deberes[i].vector2 = vector2;
            deberes[i].resultado = resultado;
            deberes[i].tamVector = tamVector;
            deberes[i].mutex = &mutex;
            deberes[i].ultimo=nHilos;
            //printf("nHilos:%d",nHilos);
            pthread_create(&hilos[i], NULL, cuerpoHilo, &deberes[i]);
        }

        for (int i = 0; i < nHilos; i++) {
            pthread_join(hilos[i], NULL);
        }

        if (tamVector <= 10) {
            printf("Vector 1: ");
            mostrarVector(vector1, tamVector);
            printf("Vector 2: ");
            mostrarVector(vector2, tamVector);
        }

        printf("Resultado: %d\n", *resultado);

        free(vector1);
        free(vector2);
        free(resultado);
        free(hilos);
        free(deberes);
    }
    return 0;
}
