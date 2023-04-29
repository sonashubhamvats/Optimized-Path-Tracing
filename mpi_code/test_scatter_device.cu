#include <stdio.h>
#include <C:\Program Files\MPI\MS_MPI\SDK\Include\mpi.h>
#include <cuda.h>
#include <iostream>
#include "../cuda_common.cuh"
#include "../camera.cuh"

__global__ void initialize_camera(camera *d_camera){
    vec3 lookfrom(-5.0f, 0.0f, 25.0f);
    vec3 lookat(0,0,0);
    float dist_to_focus = (lookfrom-lookat).length();
    float aperture = 0.0;
    *d_camera   = camera(lookfrom,
                    lookat,
                    vec3(0,1,0),
                    60.0,
                    float(400)/float(200),
                    aperture,
                    dist_to_focus,0.0,1.0);
}

__global__ void print_camera_details_D(camera *c)
{
    printf("%f h %f",(*c).lens_radius,(*c).origin.x()); 
}

void print_camera_details(camera *c)
{
    std::cout<<(*c).lens_radius<<" h "<<(*c).origin.x()<<std::endl; 
}

int main(int argc, char *argv[]){
    int rank=0, size;
    int tag=0;

    camera *s_d_camera;

    cudaMalloc(&s_d_camera,sizeof(camera));
    
    initialize_camera<<<1,1>>>(s_d_camera);
    cudaDeviceSynchronize();

    
    camera *s_h_camera = (camera*)malloc(sizeof(camera));
    //printf("\nHell");
    cudaMemcpy(s_h_camera,s_d_camera,sizeof(camera),cudaMemcpyDeviceToHost);
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    print_camera_details(s_h_camera);
    print_camera_details_D<<<1,1>>>(s_d_camera);
    cudaDeviceSynchronize();
    // if(rank==0){
        
        
    //     //gpuErrchk(cudaMemcpy(s_h_camera[0],s_d_camera[0],sizeof(camera),cudaMemcpyDeviceToHost));
    //     MPI_Bcast(s_h_camera, sizeof(camera), MPI_BYTE, 0,MPI_COMM_WORLD);


    // }
    // else
    // {
    //     camera *r_h_camera;
    //     r_h_camera = (camera*)malloc(sizeof(camera));
    //     MPI_Bcast(r_h_camera, sizeof(camera), MPI_BYTE, 0, MPI_COMM_WORLD);

    //     print_camera_details(s_h_camera);
    //     camera *r_d_camera;
    //     cudaMalloc(&s_d_camera,sizeof(camera));
    //     cudaMemcpy(r_d_camera,r_h_camera,sizeof(camera),cudaMemcpyHostToDevice);

    //     print_camera_details_D<<<1,1>>>(r_d_camera);
    //     cudaDeviceSynchronize();

    // }

    MPI_Finalize();

    //MPI_Send();
    

    return 0;
}