#include <C:\Program Files\MPI\MS_MPI\SDK\Include\mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "../vec3.cuh"

struct  my_struct{
    vec3 xyz;
};

struct nested_my_struct{
    my_struct m;
};

int main(int argc, char** argv) {
    int rank, size, tag = 0;
    

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    my_struct send_struct = {vec3(22,22,23)};
    nested_my_struct nested_my_struct_send = {send_struct};
    nested_my_struct nested_my_struct_recv;
    // Define MPI data type for my_struct
    // int block_lengths[2] = { 1, 1 };
    // MPI_Datatype types[2] = { MPI_INT, MPI_DOUBLE };
    // MPI_Aint displacements[2];
    // displacements[0] = offsetof(my_struct, id);
    // displacements[1] = offsetof(my_struct, values);
    // MPI_Datatype my_struct_mpi_type;
    // MPI_Type_create_struct(2, block_lengths, displacements, types, &my_struct_mpi_type);
    // MPI_Type_commit(&my_struct_mpi_type);

    if (rank == 0) {
        // Allocate memory for values and initialize
        // send_struct.values = (double*) malloc(3 * sizeof(double));
        // send_struct.values[0] = 1.0;
        // send_struct.values[1] = 2.0;
        // send_struct.values[2] = 3.0;

        // Send struct to process 1
        MPI_Send(&nested_my_struct_send, sizeof(nested_my_struct), MPI_BYTE, 1, tag, MPI_COMM_WORLD);

        // printf("Process %d sent (%d, %f, %f, %f) to process 1\n", rank, send_struct.id,
        //         send_struct.values[0], send_struct.values[1], send_struct.values[2]);

        // // Free memory
        // free(send_struct.values);
    } else if (rank == 1) {
        // Receive struct from process 0
        MPI_Recv(&nested_my_struct_recv, sizeof(nested_my_struct), MPI_BYTE, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("%d %f %f %f\n",rank,nested_my_struct_recv.m.xyz.x(),
        nested_my_struct_recv.m.xyz.y(),nested_my_struct_recv.m.xyz.z());
        // printf("Process %d received (%d, %f, %f, %f) from process 0\n", rank, recv_struct.id,
        //         recv_struct.values[0], recv_struct.values[1], recv_struct.values[2]);

        // // Free memory
        // free(recv_struct.values);
    }

    // Free MPI data type
    //MPI_Type_free(&my_struct_mpi_type);

    MPI_Finalize();
    return 0;
}