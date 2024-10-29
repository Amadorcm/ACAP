#include <stdio.h>
#include <mpi.h>

#define MASTER 0

int main(int argc, char* argv[]){
	int rank, len, numProcs;
	char procName[MPI_MAX_PROCESSOR_NAME];
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Get_processor_name(procName, &len);
	printf("Soy el proceso %d en el procesador %s\n", rank, procName);
	if(rank==MASTER){
		MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
		printf("MASTER: Hay un total de %d procesos!\n", numProcs);
	}
	MPI_Finalize();
	return 0;
}
