#include <stdio.h>
#include <C:\Program Files\MPI\MS_MPI\SDK\Include\mpi.h>
#include <cuda.h>
#include <iostream>

#define N 24

__global__ void add(int *a, int *b, int *c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char *argv[]) {
    
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int *a, *b, *c;
    int *c_a,*c_b,*c_c;
    int *d_a, *d_b, *d_c;
    int i;
    c_a = (int *) malloc((N/size) * sizeof(int));
    c_b = (int *) malloc((N/size) * sizeof(int));

    if (rank == 0) {
        // Allocate memory on the parent process
        a = (int *) malloc(N * sizeof(int));
        b = (int *) malloc(N * sizeof(int));
        c = (int *) malloc(N * sizeof(int));
        
        // Initialize arrays a and b
        for (i = 0; i < N; i++) {
            a[i] = i;
            b[i] = 2 * i;
        }
        std::cout<<size<<" h "<<N<<std::endl;
    }

    // Scatter arrays a and b to other processes
    MPI_Scatter(a, N/size, MPI_INT, c_a, N/size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(b, N/size, MPI_INT, c_b, N/size, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory on each process for arrays a, b, and c
    cudaMalloc(&d_a, N/size * sizeof(int));
    cudaMalloc(&d_b, N/size * sizeof(int));
    cudaMalloc(&d_c, N/size * sizeof(int));

    // Copy arrays a and b from host to device memory
    cudaMemcpy(d_a, c_a, N/size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, c_b, N/size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel on each process
    add<<<N, 8>>>(d_a, d_b, d_c, N/size);
    cudaDeviceSynchronize();

    c_c = (int *) malloc(N/size * sizeof(int));
    // // Copy array c from device to host memory
    cudaMemcpy(c_c, d_c, N/size * sizeof(int), cudaMemcpyDeviceToHost);

    // Gather array c from all processes to the parent process
    MPI_Gather(c_c, N/size, MPI_INT, c, N/size, MPI_INT, 0, MPI_COMM_WORLD);

    // Print array c on the parent process
    if (rank == 0) {
        for (i = 0; i < N; i++) {
            printf("%d ", c[i]);
        }
        printf("\n");
    }

    // // Free memory on each process
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    if (rank == 0) {
        free(a);
        free(b);
        free(c);
    }

    MPI_Finalize();

    return 0;
}