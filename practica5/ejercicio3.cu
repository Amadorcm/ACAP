//Ejercicio 3: Multiplicacion Matrices CUDA y tiempo
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

// Función para generar una matriz aleatoria
void generateRandomMatrix(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (double)(rand() % 100);
    }
}

// Función para imprimir una matriz
void printMatrix(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

// Función para multiplicar matrices en CPU
void matrixMultiplyCPU(double *A, double *B, double *C, int rowsA, int colsA, int colsB) {
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            double sum = 0.0;
            for (int k = 0; k < colsA; k++) {
                sum += A[i * colsA + k] * B[k * colsB + j];
            }
            C[i * colsB + j] = sum;
        }
    }
}

// Kernel para multiplicar matrices en GPU
__global__ void matrixMultiplyGPU(double *A, double *B, double *C, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        double sum = 0.0;
        for (int k = 0; k < colsA; k++) {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <rowsA> <colsA> <colsB> <blockSize>\n", argv[0]);
        return 1;
    }

    int rowsA = atoi(argv[1]);
    int colsA = atoi(argv[2]);
    int colsB = atoi(argv[3]);
    int blockSize = atoi(argv[4]);

    size_t sizeA = rowsA * colsA * sizeof(double);
    size_t sizeB = colsA * colsB * sizeof(double);
    size_t sizeC = rowsA * colsB * sizeof(double);

    double *h_A = (double *)malloc(sizeA);
    double *h_B = (double *)malloc(sizeB);
    double *h_C = (double *)malloc(sizeC);
    double *h_C_GPU = (double *)malloc(sizeC);

    generateRandomMatrix(h_A, rowsA, colsA);
    generateRandomMatrix(h_B, colsA, colsB);

    if (rowsA <= 7 && colsA <= 7 && colsB <= 7) {
        printf("Matrix A:\n");
        printMatrix(h_A, rowsA, colsA);
        printf("Matrix B:\n");
        printMatrix(h_B, colsA, colsB);
    }

    // Medir tiempo en CPU
    clock_t startCPU = clock();
    matrixMultiplyCPU(h_A, h_B, h_C, rowsA, colsA, colsB);
    clock_t endCPU = clock();
    double cpuTime = (double)(endCPU - startCPU) / CLOCKS_PER_SEC;

    double *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, sizeA);
    cudaMalloc((void **)&d_B, sizeB);
    cudaMalloc((void **)&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 block(blockSize, blockSize);
    dim3 grid((colsB + block.x - 1) / block.x, (rowsA + block.y - 1) / block.y);

    // Medir tiempo en GPU
    cudaEvent_t startGPU, stopGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);
    cudaEventRecord(startGPU, 0);

    matrixMultiplyGPU<<<grid, block>>>(d_A, d_B, d_C, rowsA, colsA, colsB);

    cudaEventRecord(stopGPU, 0);
    cudaEventSynchronize(stopGPU);

    float gpuTime;
    cudaEventElapsedTime(&gpuTime, startGPU, stopGPU);
    gpuTime /= 1000.0; // Convertir a segundos

    cudaMemcpy(h_C_GPU, d_C, sizeC, cudaMemcpyDeviceToHost);

    if (rowsA <= 7 && colsA <= 7 && colsB <= 7) {
        printf("Result Matrix (CPU):\n");
        printMatrix(h_C, rowsA, colsB);
        printf("Result Matrix (GPU):\n");
        printMatrix(h_C_GPU, rowsA, colsB);
    }

    // Verificar que los resultados coincidan
    int correct = 1;
    for (int i = 0; i < rowsA * colsB; i++) {
        if (fabs(h_C[i] - h_C_GPU[i]) > 1e-5) {
            correct = 0;
            break;
        }
    }

    if (correct) {
        printf("Cálculo correcto\n");
    } else {
        printf("Cálculo incorrecto\n");
    }

    printf("La operación en CPU (secuencial) ha tardado: %f segundos\n", cpuTime);
    printf("La operación en GPU ha tardado: %f segundos\n", gpuTime);

    // Liberar memoria
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_GPU);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
