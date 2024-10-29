
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MASTER 0
#define DESTROY_TAG 666
#define NORMAL_TAG 1

//Funcion que realizan las hebras que no son la master y realiza la multiplicacion y la suma de las posiciones de los vectores que reciban de la master y envia dicho calculo a la master de nuevo
void worker(int rank, int numProcs){
    MPI_Status status;
    MPI_Probe(MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    if(status.MPI_TAG == NORMAL_TAG){
        int receivedSize;
        MPI_Recv(&receivedSize, 1, MPI_INT, MASTER, NORMAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Recibir el tamaño de vectorU
        int* receivedVectorU = malloc(sizeof(int) * receivedSize);
        int* receivedVectorV = malloc(sizeof(int) * receivedSize);
        MPI_Recv(receivedVectorU, receivedSize, MPI_INT, MASTER, NORMAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Recibir vectorU
        MPI_Recv(receivedVectorV, receivedSize, MPI_INT, MASTER, NORMAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Recibir vectorV
        
        int partialResult = 0;
        for(int i = 0; i < receivedSize; i++){
            partialResult += receivedVectorU[i] * receivedVectorV[i]; // Realizar el producto escalar parcial
        }
        MPI_Send(&partialResult, 1, MPI_INT, MASTER, NORMAL_TAG, MPI_COMM_WORLD); // Enviar el resultado parcial al maestro
        
        free(receivedVectorU);
        free(receivedVectorV);
    }
}
//Funcion de la hebra master que divide los vectores y se lo envia a las hebras trabajadores, luego recive los resultados de estas hebras y lo suma en el total
void masterTask(int rank, int numProcs, int dataSize){
    int* vectorU = malloc(sizeof(int) * dataSize);
    int* vectorV = malloc(sizeof(int) * dataSize);

    // Llenar los vectores U y V con datos
    for(int i = 0; i < dataSize; i++){
        vectorU[i] = i;
        vectorV[i] = i * 2; // Un ejemplo arbitrario para el vector V
    }

    // Dividir los datos entre los trabajadores
    int numWorkers = numProcs - 1;
    int pack = dataSize / numWorkers;
    int offset = dataSize % numWorkers;
    int focus = 0, sentSize = 0;
    for(int i = 0; i < numWorkers; i++){
        sentSize = (pack + (i < offset));
        MPI_Send(&sentSize, 1, MPI_INT, (i + 1), NORMAL_TAG, MPI_COMM_WORLD); // Enviar el tamaño de vectorU
        MPI_Send(&(vectorU[focus]), sentSize, MPI_INT, (i + 1), NORMAL_TAG, MPI_COMM_WORLD); // Enviar vectorU
        MPI_Send(&(vectorV[focus]), sentSize, MPI_INT, (i + 1), NORMAL_TAG, MPI_COMM_WORLD); // Enviar vectorV
        focus += sentSize;
    }

    // Recibir los resultados parciales y calcular el resultado final
    MPI_Status status;
    int resultado = 0, buffer;
    for(int i = 0; i < numWorkers; i++){
        MPI_Recv(&buffer, 1, MPI_INT, (i + 1), NORMAL_TAG, MPI_COMM_WORLD, &status);
        resultado += buffer;
    }
    printf("El producto escalar de los vectores U y V es: %d\n", resultado);

    // Liberar memoria
    free(vectorU);
    free(vectorV);
}

//funcion para cerrar el proceso de la hebra
void shutDown(int numProcs){
    for(int i = 1; i < numProcs; i++){
        MPI_Send(0, 0, MPI_INT, i, DESTROY_TAG, MPI_COMM_WORLD);
    }
}
//funcion main que crea las hebras y le asigna su función correspondiente y controla los errores
int main(int argc, char* argv[]){
    int rank, numProcs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    if(rank == MASTER){
        if(numProcs < 2){
            printf("Error: Se requieren al menos 2 procesos!\n");
            shutDown(numProcs);
        }else{
            if(argc != 2){
                printf("Error: Se espera el tamaño del vector como argumento\n");
                shutDown(numProcs);
            }else{
                int dataSize = atoi(argv[1]);
                if(dataSize <= 0){
                    printf("Error: Tamaño de datos inválido\n");
                    shutDown(numProcs);
                }else{
                    masterTask(rank, numProcs, dataSize);
                }
            }
        }
    }else{
        worker(rank, numProcs);
    }
    MPI_Finalize();
    return 0;
}
