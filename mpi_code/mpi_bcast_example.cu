#include <C:\Program Files\MPI\MS_MPI\SDK\Include\mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int rank, size;
    

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int sendbuf[10];
    int recvbuf[5];
    int value = 0;
    int value_1 = 0;
    int value_2 = 0;
    if (rank == 0) {
        // Broadcast the value from the root process (rank 0)
        value = 42;
        value_1 = 32;
        value_2 = 44;
        printf("Process %d broadcasted the value %d to all other processes\n", rank, value);
        MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&value_1, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&value_2, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
    } else {
        // Receive the broadcasted value in all other processes
        MPI_Bcast(&value_1, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&value_2, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
    }
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Process %d received the broadcasted value %d %d %d\n", rank, value,value_1,value_2);

    MPI_Finalize();
    return 0;
}