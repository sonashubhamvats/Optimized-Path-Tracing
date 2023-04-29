#include <C:\Program Files\MPI\MS_MPI\SDK\Include\mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int n = 10;
    const int chunk_size = 5;
    int sendbuf[n];
    int recvbuf[chunk_size];

    if (rank == 0) {
        // Initialize send buffer
        for (int i = 0; i < n; i++) {
            sendbuf[i] = i;
        }
    }

    // Scatter data from root process to all other processes
    MPI_Scatter(sendbuf, chunk_size, MPI_INT, recvbuf, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    printf("Process %d received:", rank);
    for (int i = 0; i < n / size; i++) {
        printf(" %d", recvbuf[i]);
    }
    printf("\n");

    MPI_Finalize();
    return 0;
}