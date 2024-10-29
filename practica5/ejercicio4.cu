//Ejercicio 4: Tanimoto GPU sin atomic
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define CUDA_CHECK_ERROR(msg) \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "[%s] CUDA error: %s: %s. Exiting.\n", \
                msg, cudaGetErrorString(err), __FILE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void intersectionKernel(int *a, int *b, int *result, int size) {
    extern __shared__ int shared_inter[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;

    // Initialize shared memory
    shared_inter[local_tid] = 0;
    __syncthreads();

    // Calculate local intersections
    if (tid < size) {
        if (a[tid] == 1 && b[tid] == 1) {
            shared_inter[local_tid] = 1;
        }
    }
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (local_tid < stride) {
            shared_inter[local_tid] += shared_inter[local_tid + stride];
        }
        __syncthreads();
    }

    // Store result of this block's reduction
    if (local_tid == 0) {
        atomicAdd(result, shared_inter[0]);
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

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);

    // Launch kernel with shared memory size
    intersectionKernel<<<grid_size, block_size, block_size * sizeof(int)>>>(d_a, d_b, d_inter, size);
    CUDA_CHECK_ERROR("CUDA kernel launch error");

    // Record the stop event
    cudaEventRecord(stop, 0);

    // Wait for the stop event to complete
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back to host
    cudaMemcpy(&inter, d_inter, sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR("CUDA memcpy D2H error");

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_inter);

    // Destroy the events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Calculate Tanimoto coefficient
    float tanimoto_coefficient = (float)inter / (float)(size - inter + inter);

    // Print the time taken
    printf("Tiempo Tanimoto: %f ms\n", milliseconds);

    return tanimoto_coefficient;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Uso: %s <longitud de los vectores> <tamaño del bloque> <semilla>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int size = atoi(argv[1]);
    int seed_a = atoi(argv[2]);
    int seed_b = atoi(argv[3]);

    int *a = (int *)malloc(size * sizeof(int));
    int *b = (int *)malloc(size * sizeof(int));

    srand(seed_a);
    for (int i = 0; i < size; i++) {
        a[i] = rand() % 2;
    }

    srand(seed_b);
    for (int i = 0; i < size; i++) {
        b[i] = rand() % 2;
    }

    float tanimoto = calculateTanimoto(a, b, size);
    printf("Tanimoto: %f\n", tanimoto);

    free(a);
    free(b);

    return 0;
}
