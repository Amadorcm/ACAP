// Ejercicio 2: Coeficiente tanimoto implementando la interseccion en el kernel y el resto en la cpu
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define CUDA_CHECK_ERROR(msg) \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "[%s] CUDA error: %s: %s. Exiting.\n", \
                msg, cudaGetErrorString(err), __FILE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void intersectionKernel(int *a, int *b, int *inter, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size) {
        if (a[tid] == 1 && b[tid] == 1) {
            atomicAdd(inter, 1);
        }
    }
}

float calculateTanimoto(int *a, int *b, int size) {
    int inter = 0;

    // Allocate device memory
    int *d_a, *d_b, *d_inter;
    cudaMalloc((void **)&d_a, size * sizeof(int));
    cudaMalloc((void **)&d_b, size * sizeof(int));
    cudaMalloc((void **)&d_inter, sizeof(int));
    CUDA_CHECK_ERROR("CUDA malloc error");

    // Copy data to device
    cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_inter, 0, sizeof(int));
    CUDA_CHECK_ERROR("CUDA memcpy H2D error");

    // Calculate grid size
    int block_size = 512;
    int grid_size = (size + block_size - 1) / block_size;

    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel
    intersectionKernel<<<grid_size, block_size>>>(d_a, d_b, d_inter, size);
    CUDA_CHECK_ERROR("CUDA kernel launch error");

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back to host
    cudaMemcpy(&inter, d_inter, sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR("CUDA memcpy D2H error");

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_inter);

    // Calculate Tanimoto coefficient
    float tanimoto_coefficient = (float)inter / (float)(size - inter + inter);
    printf("Tiempo Tanimoto: %.6f ms\n", milliseconds);

    return tanimoto_coefficient;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Uso: %s <longitud de los vectores> <tamaño del bloque> <semilla>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int size = atoi(argv[1]);
    int block_size = atoi(argv[2]);
    int seed = atoi(argv[3]);

    // Allocate host memory
    int *h_a, *h_b;
    h_a = (int *)malloc(size * sizeof(int));
    h_b = (int *)malloc(size * sizeof(int));

    // Initialize vectors randomly
    srand(seed);
    for (int i = 0; i < size; i++) {
        h_a[i] = rand() % 2; // 0 or 1
        h_b[i] = rand() % 2; // 0 or 1
    }

    // Calculate Tanimoto coefficient
    float tanimoto = calculateTanimoto(h_a, h_b, size);

    // Print result
    printf("Tanimoto: %.6f\n", tanimoto);

    // Free memory
    free(h_a);
    free(h_b);

    return 0;
}
