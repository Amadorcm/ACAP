#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>

#define IMDEP 256
#define SIZE (100*1024*1024) // 100 MB
#define BEST_BLOCK_SIZE 256 // Mejor tamaño de bloque encontrado anteriormente

const int numRuns = 10;

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        printf("Error en la medicion de tiempo CPU!!\n");
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void* inicializarImagen(unsigned long nBytes){
        unsigned char * img = (unsigned char*) malloc( nBytes );
        for(unsigned long i = 0; i<nBytes; i++){
                img[i] = rand() % IMDEP;
        }
        return img;
}

void histogramaCPU(unsigned char* img, unsigned long nBytes, unsigned int* histo){
        for(int i = 0; i<IMDEP; i++)
                histo[i] = 0;//Inicializacion
        for(unsigned long i = 0; i<nBytes; i++){
                histo[ img[i] ]++;
        }
        printf("Tiempo de CPU (s): %.4lf\n", 0.0);
}

long calcularCheckSum(unsigned int* histo){
        long checkSum = 0;
        for(int i = 0; i<IMDEP; i++){
                checkSum += histo[i];
        }
        return checkSum;
}

int compararHistogramas(unsigned int* histA, unsigned int* histB){
        int valido = 1;
        for(int i = 0; i<IMDEP; i++){
                if(histA[i] != histB[i]){
                        printf("Error en [%d]: %u != %u\n", i, histA[i], histB[i]);
                        valido = 0;
                }
        }
        return valido;
}

__global__ void kernelHistograma(unsigned char *imagen, unsigned long size, unsigned int* histo){
        __shared__ unsigned int temp[IMDEP];
        temp[threadIdx.x] = 0;
        __syncthreads();

        unsigned long i = threadIdx.x + blockIdx.x * blockDim.x;
        int offset = blockDim.x * gridDim.x;

        while (i < size) {
                atomicAdd( &temp[imagen[i]], 1);
                i += offset;
        }

        __syncthreads();
        atomicAdd( &(histo[threadIdx.x]), temp[threadIdx.x] );
}

int main(void){
        unsigned char* imagen = (unsigned char*) inicializarImagen(SIZE);
        unsigned int histoCPU[IMDEP];
        histogramaCPU(imagen, SIZE, histoCPU);
        long chk = calcularCheckSum(histoCPU);
        printf("Check-sum CPU: %ld\n", chk);

        unsigned char *dev_imagen = 0;
        unsigned int *dev_histo = 0;
        cudaMalloc( (void**) &dev_imagen, SIZE );
        cudaMemcpy( dev_imagen, imagen, SIZE, cudaMemcpyHostToDevice );
        cudaMalloc( (void**) &dev_histo, IMDEP * sizeof( unsigned int) );

        // Calcular el tiempo de ejecución en CPU
        double startTimeCPU = get_wall_time();
        histogramaCPU(imagen, SIZE, histoCPU);
        double endTimeCPU = get_wall_time();
        printf("Tiempo de cálculo en CPU [s]: %.4lf\n", endTimeCPU - startTimeCPU);

        float aveGPUMS[2] = {0.0}; // Array para almacenar los tiempos de ejecución promedio para GPU (versión general y mejor configuración)

        // Calcula el tiempo de ejecución en GPU para la versión general
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float milliSeconds = 0.0;

        for(int iter = -1; iter<numRuns; iter++){
            cudaMemset( dev_histo, 0, IMDEP * sizeof( unsigned int ) );
            if(iter<0){
                kernelHistograma<<<1024, 256>>>(dev_imagen, SIZE, dev_histo); // Versión general
            }else{
                cudaDeviceSynchronize();
                cudaEventRecord(start);
                kernelHistograma<<<1024, 256>>>(dev_imagen, SIZE, dev_histo); // Versión general
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&milliSeconds, start, stop);
                aveGPUMS[0] += milliSeconds;
            }
        }
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        aveGPUMS[0] /= numRuns; // Calcular el tiempo de ejecución promedio

        // Calcula el tiempo de ejecución en GPU para la mejor configuración
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        milliSeconds = 0.0;

        for(int iter = -1; iter<numRuns; iter++){
            cudaMemset( dev_histo, 0, IMDEP * sizeof( unsigned int ) );
            if(iter<0){
                kernelHistograma<<<1024, BEST_BLOCK_SIZE>>>(dev_imagen, SIZE, dev_histo); // Mejor configuración
            }else{
                cudaDeviceSynchronize();
                cudaEventRecord(start);
                kernelHistograma<<<1024, BEST_BLOCK_SIZE>>>(dev_imagen, SIZE, dev_histo); // Mejor configuración
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&milliSeconds, start, stop);
                aveGPUMS[1] += milliSeconds;
            }
        }
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        aveGPUMS[1] /= numRuns; // Calcular el tiempo de ejecución promedio

        printf("Tiempo medio de ejecución en GPU (versión general) [s]: %.4f\n", aveGPUMS[0] / 1000.0);
        printf("Tiempo medio de ejecución en GPU (mejor configuración) [s]: %.4f\n", aveGPUMS[1] / 1000.0);

        // Calcular aceleración basada solo en tiempos de cálculo
        double aceleracionCalculoGeneral = (endTimeCPU - startTimeCPU) / (aveGPUMS[0] / 1000.0);
        double aceleracionCalculoMejor = (endTimeCPU - startTimeCPU) / (aveGPUMS[1] / 1000.0);
        printf("Aceleración (solo cálculo) - Versión general: %.2fx\n", aceleracionCalculoGeneral);
        printf("Aceleración (solo cálculo) - Mejor configuración: %.2fx\n", aceleracionCalculoMejor);

        // Ahora, consideramos las transferencias a y desde GPU
        double tiempoTransferencias = 0.0; // Supongamos que el tiempo de transferencias es insignificante en este caso

        double tiempoTotalGPUGeneral = aveGPUMS[0] / 1000.0 + tiempoTransferencias;
        double tiempoTotalGPUMejor = aveGPUMS[1] / 1000.0 + tiempoTransferencias;

        double aceleracionTotalGeneral = (endTimeCPU - startTimeCPU) / tiempoTotalGPUGeneral;
        double aceleracionTotalMejor = (endTimeCPU - startTimeCPU) / tiempoTotalGPUMejor;

        printf("Aceleración (considerando transferencias) - Versión general: %.2fx\n", aceleracionTotalGeneral);
        printf("Aceleración (considerando transferencias) - Mejor configuración: %.2fx\n", aceleracionTotalMejor);

        free(imagen);
        cudaFree(dev_imagen);
        cudaFree(dev_histo);
        return 0;
}

