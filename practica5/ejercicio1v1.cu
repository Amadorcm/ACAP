//Ejercicio 1 Version 1 Valor dinámico del grid
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CUDA_CHECK_ERROR(msg) \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "[%s] CUDA error: %s: %s. Exiting.\n", \
                msg, cudaGetErrorString(err), __FILE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void vectorMultiply(float *a, float *b, float *c, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size) {
        c[tid] = a[tid] * b[tid];
    }
}

void vectorMultiplyCPU(float *a, float *b, float *c, int size) {
    for (int i = 0; i < size; i++) {
        c[i] = a[i] * b[i];
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Uso: %s <longitud del vector> <tamaño de bloque>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int size = atoi(argv[1]);
    int block_size = atoi(argv[2]);

    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    int grid_size;

    // Allocate host memory
    h_a = (float *)malloc(size * sizeof(float));
    h_b = (float *)malloc(size * sizeof(float));
    h_c_cpu = (float *)malloc(size * sizeof(float));
    h_c_gpu = (float *)malloc(size * sizeof(float));

    // Initialize vectors randomly
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        h_a[i] = (float)(rand() % 60);
        h_b[i] = (float)(rand() % 60);
    }

    // Allocate device memory
    cudaMalloc((void **)&d_a, size * sizeof(float));
    cudaMalloc((void **)&d_b, size * sizeof(float));
    cudaMalloc((void **)&d_c, size * sizeof(float));
    CUDA_CHECK_ERROR("CUDA malloc error");

    // Copy data to device
    cudaMemcpy(d_a, h_a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR("CUDA memcpy H2D error");

    // Calculate grid size
    grid_size = (size + block_size - 1) / block_size;

    // Launch kernel
    vectorMultiply<<<grid_size, block_size>>>(d_a, d_b, d_c, size);
    CUDA_CHECK_ERROR("CUDA kernel launch error");

    // Copy result back to host
    cudaMemcpy(h_c_gpu, d_c, size * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR("CUDA memcpy D2H error");

    // Execute CPU version for comparison if size is less than 7
    if (size < 7) {
        vectorMultiplyCPU(h_a, h_b, h_c_cpu, size);

        printf("Vector A:\n");
        for (int i = 0; i < size; i++) {
            printf("%.2f ", h_a[i]);
        }
        printf("\n");

        printf("Vector B:\n");
        for (int i = 0; i < size; i++) {
            printf("%.2f ", h_b[i]);
        }
        printf("\n");

        printf("Resultado CPU:\n");
        for (int i = 0; i < size; i++) {
            printf("%.2f ", h_c_cpu[i]);
        }
        printf("\n");

        printf("Resultado GPU:\n");
        for (int i = 0; i < size; i++) {
            printf("%.2f ", h_c_gpu[i]);
        }
        printf("\n");
    }else{
        printf("Operacion realizada correctamente!\n");
    }

    // Free memory
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

