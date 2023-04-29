#include "../cuda_common.cuh"
#include "../constants.cuh"
#include <float.h>
#include <time.h>
#include "../camera.cuh"
#include "../texture.cuh"
#include "../sphere.cuh"
#include "../collider_list.cuh"
#include "../materials.cuh"
#include "../moving_spheres.cuh"
#include "../rt_stb_image.h"
#include "../bvh_gpu.cuh"
#include "../bvh_cpu.cuh"
#include "../aarect.cuh"
#include "../box.cuh"
#include "../constant_medium.cuh"

#include <fstream>
#include <string>
#include <cmath>
#include <sstream>
#include <C:\Program Files\MPI\MS_MPI\SDK\Include\mpi.h>
__device__ vec3 optimized_color_tree_streamlined_sampling(const ray &r, bvh_gpu_streamlined_tree *bvh_tree,
curandState *temp_state,vec3 &background, triangle *d_list,int s[])
{
    ray cur_ray = r;
    vec3 curr_atten(1.0,1.0,1.0);
    ray temp_scattered_ray;
    vec3 temp_attentuation,emitted;
    collider_record rec;
    int itr = 0;
    float t;
    vec3 unit_direction;
    
    for(;itr<51;itr++)
    {
        
        if(bvh_tree->hit(cur_ray,0.001,FLT_MAX,rec,0,s))
        {
            vec3 scatter_direction = rec.normal 
            + random_in_unit_sphere(temp_state);
            if (scatter_direction.near_zero())
                scatter_direction = rec.normal;
            cur_ray = ray(rec.p, scatter_direction,cur_ray.time());
            curr_atten = vec3(0.98f,0.360f,0.360f)*curr_atten;
            
        }
        else
        {
            //printf("\n%d",itr);
            return background*curr_atten;
        }
    }
    return vec3(0,0,0);
}

__global__ void optimized_render_streamlined_tree_sampling_mpi(vec3 *anti_alias_pixel_arr,int nx,int ny,camera *d_camera,
bvh_gpu_streamlined_tree *bvh_tree,triangle*d_list,int process_rank,int height_of_each_chunk)
{
    if(blockIdx.x>=nx||blockIdx.y>=ny)
        return;

    int total_no_threads_in_a_block = blockDim.x*blockDim.y;
    int total_no_threads_in_a_row = total_no_threads_in_a_block*gridDim.x;
    int anti_alias_pixel_index = threadIdx.x + threadIdx.y*blockDim.x + total_no_threads_in_a_block*blockIdx.x+
    total_no_threads_in_a_row*blockIdx.y;

    curandState local_state;
    curand_init(1984+anti_alias_pixel_index, 0, 0, &local_state);


    vec3 col(0,0,0);
    float u,v;
    vec3 background(0.70, 0.80, 1.00);


    u = float(blockIdx.x+curand_uniform(&local_state))/float(nx);
    v = float(blockIdx.y+process_rank*height_of_each_chunk+curand_uniform(&local_state))/float(ny);
    
    ray r(vec3(0,0,0),vec3(0,0,0));
    (*d_camera).get_ray(u,v,&local_state,r);

    //__shared__ int s[15];
    int s[25];

    col=optimized_color_tree_streamlined_sampling(r,bvh_tree,&local_state,background,d_list,s);
    

    anti_alias_pixel_arr[anti_alias_pixel_index] = col;
}

__global__ void optimized_world_renderer_streamlined_mpi(triangle *d_list,vec3 *a,vec3 *b,vec3 *c,
camera *d_camera,aabb *list_of_bounding_boxes,int nx,int ny,int size)
{
    curandState localState;

    curand_init(1984,0,0,&localState);

   

    for(int i = 0; i<size;i++)
    {
        vec3 rand_point = random_vector_in_range(&localState,-10,10);

        vec3 a = rand_point - random_vector_in_range(&localState,-0.5,0.5);
        vec3 b = rand_point - random_vector_in_range(&localState,-0.5,0.5);
        vec3 c = rand_point - random_vector_in_range(&localState,-0.5,0.5);
        //triangle t = triangle(a[i],b[i],c[i],i);
        triangle t = triangle(a,b,c,i);
        d_list[i] =t;
        t.bounding_box(0,0,
        list_of_bounding_boxes[i],i);

    }
        
    vec3 lookfrom(-5.0f, 0.0f, 25.0f);
    vec3 lookat(0,0,0);
    float dist_to_focus = (lookfrom-lookat).length();
    float aperture = 0.0;
    *d_camera = camera(lookfrom,
                    lookat,
                    vec3(0,1,0),
                    60.0,
                    float(nx)/float(ny),
                    aperture,
                    dist_to_focus,0.0,1.0);
    
}

void host_bvh_tree_creation(aabb *list_of_bounding_boxes,
bvh_gpu_node *b_arr,int size)
{
    bvh_cpu parent = 
    bvh_cpu(0,size,0);
    //printf("\nHere");
    form_gpu_bvh(list_of_bounding_boxes,b_arr,parent);
    // int counter=0;
    // bvh_tree_traversal(b_arr,0,2*size-1,counter);
    // printf("\n%d",counter);

}

__global__ void device_memory_check(camera *d_camera,bvh_gpu_streamlined_tree *bvh_tree,
triangle *d_list,int total_no_of_objects_in_the_scene){
    printf("\nPrinting out the triangles -");
    for(int i=0;i<total_no_of_objects_in_the_scene;i++)
    {
        printf("\nTriangle- %d",i);
        printf("\n%f %f %f",d_list[i].a.x(),d_list[i].a.y(),d_list[i].a.z());
        printf("\n%f %f %f",d_list[i].b.x(),d_list[i].b.y(),d_list[i].b.z());
        printf("\n%f %f %f",d_list[i].c.x(),d_list[i].c.y(),d_list[i].c.z());
    }

    printf("\n\nPrinting out the triangles from bvh_tree -");
    for(int i=0;i<total_no_of_objects_in_the_scene;i++)
    {
        printf("\nTriangle- %d",i);
        printf("\n%f %f %f",bvh_tree->list_of_colliders[i].a.x()
        ,bvh_tree->list_of_colliders[i].a.y(),bvh_tree->list_of_colliders[i].a.z());
        printf("\n%f %f %f",bvh_tree->list_of_colliders[i].b.x()
        ,bvh_tree->list_of_colliders[i].b.y(),bvh_tree->list_of_colliders[i].b.z());
        printf("\n%f %f %f",bvh_tree->list_of_colliders[i].c.x()
        ,bvh_tree->list_of_colliders[i].c.y(),bvh_tree->list_of_colliders[i].c.z());
    }
    
    printf("\n\nSize of bvh_arr %d",bvh_tree->bvh_node_arr_size);
    printf("\nPrinting out the bvh_arr from some -");
    for(int i=0;i<total_no_of_objects_in_the_scene;i++)
    {
        printf("\nbvh node - %d with index %d",i,bvh_tree->bvh_node_arr[i].box.index);
        printf("\n%f %f %f",bvh_tree->bvh_node_arr[i].box.min().x()
        ,bvh_tree->bvh_node_arr[i].box.min().y(),bvh_tree->bvh_node_arr[i].box.min().z());
        printf("\n%f %f %f",bvh_tree->bvh_node_arr[i].box.max().x()
        ,bvh_tree->bvh_node_arr[i].box.max().y(),bvh_tree->bvh_node_arr[i].box.max().z());
        
    }

    printf("\n\nCamera details - %f %f %f",d_camera->origin.x(),d_camera->origin.z(),
    d_camera->lower_left_corner.x());
    
}

__global__ void shared_reduction_interleaved_approach_complete_unrolling_2d_vec3(vec3 *input,vec3 *temp,
int size){
    
    int tid = threadIdx.x;

    vec3 *i_data = input + blockIdx.x*blockDim.x;

    __syncthreads();

    if(blockDim.x>=1024&&tid<512)
        i_data[tid]+=i_data[tid+512];
    __syncthreads();

    if(blockDim.x>=512&&tid<256)
        i_data[tid]+=i_data[tid+256];
    __syncthreads();

    if(blockDim.x>=256&&tid<128)
        i_data[tid]+=i_data[tid+128];
    __syncthreads();

    if(blockDim.x>=128&&tid<64)
        i_data[tid]+=i_data[tid+64];
    __syncthreads();


    if(tid<32)
    {
        i_data[tid]+=i_data[tid+32];
        __syncthreads();
        i_data[tid]+=i_data[tid+16];
        __syncthreads();
        i_data[tid]+=i_data[tid+8];
        __syncthreads();
        i_data[tid]+=i_data[tid+4];
        __syncthreads();
        i_data[tid]+=i_data[tid+2];
        __syncthreads();
        i_data[tid]+=i_data[tid+1];
        __syncthreads();
    }
    
    if(tid==0)
    {
        vec3 final_col=i_data[tid];
        final_col /= float(blockDim.x*blockDim.y);
        final_col[0] = sqrt(final_col[0]);
        final_col[1] = sqrt(final_col[1]);
        final_col[2] = sqrt(final_col[2]);
        
        temp[blockIdx.x] = final_col;
    }
        
}

__global__ void initialize_bvh_tree_streamlined(bvh_gpu_node *b_arr,triangle *d_list,
bvh_gpu_streamlined_tree *bvh_tree,int size)
{
    bvh_tree[0] = bvh_gpu_streamlined_tree(b_arr,d_list,size);
}

int main(int argc, char** argv){
    
    
    // size_t limit = 1024*4;
    // cudaDeviceSetLimit(cudaLimitStackSize,limit);
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int nx = 736;
    int ny = 640;
    int ns_x = 8;
    int ns_y = 8;
    int ns = ns_x*ns_y;
    double start_time = MPI_Wtime();
    int num_pixels = nx*ny;
    int no_of_objects_in_the_scene =0;
    if(rank==0)
    {
        std::vector<vec3> vertices,face_values;
        //read_from_obj("./models/budhha.obj",vertices,face_values);
        //no_of_objects_in_the_scene = face_values.size();
        no_of_objects_in_the_scene = 10000;
        MPI_Bcast(&no_of_objects_in_the_scene,1,MPI_INT,0,MPI_COMM_WORLD);
    }
    else
    {
        MPI_Bcast(&no_of_objects_in_the_scene,1,MPI_INT,0,MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int total_size_bvh= (2*(2*no_of_objects_in_the_scene-1)+1)*sizeof(bvh_gpu_node);
    int bvh_tree_size= sizeof(bvh_gpu_streamlined_tree);
    int total_no_of_tree_nodes=(2*(2*no_of_objects_in_the_scene-1)+1);

    //size bvh_arr list of triangles
    triangle *h_list = (triangle*)malloc(sizeof(triangle)*no_of_objects_in_the_scene);
    bvh_gpu_node *h_bvh_arr = (bvh_gpu_node*)malloc(total_size_bvh);
    camera *h_camera = (camera*)malloc(sizeof(camera));

    bvh_gpu_streamlined_tree *bvh_tree;

    triangle *d_list;
    camera *d_camera;
    bvh_gpu_node *bvh_arr;
    
    gpuErrchk(cudaMalloc(&bvh_tree,bvh_tree_size));
    
    gpuErrchk(cudaMalloc(&d_list,
        no_of_objects_in_the_scene*sizeof(triangle)));
    gpuErrchk(cudaMallocManaged(&bvh_arr,total_size_bvh));
    gpuErrchk(cudaMalloc(&d_camera,sizeof(camera)));

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank==0)
    {
        vec3 *a,*b,*c;
        aabb *list_of_bounding_boxes;
        int total_aabb_size = no_of_objects_in_the_scene*sizeof(aabb);
        
        
        gpuErrchk(cudaMallocManaged(&a,sizeof(vec3)*no_of_objects_in_the_scene));
        gpuErrchk(cudaMallocManaged(&b,sizeof(vec3)*no_of_objects_in_the_scene));
        gpuErrchk(cudaMallocManaged(&c,sizeof(vec3)*no_of_objects_in_the_scene));
        gpuErrchk(cudaMallocManaged(&list_of_bounding_boxes,total_aabb_size));

        //read_from_model("./models/machine_1.txt",a,b,c,vertices,face_values);
        
        optimized_world_renderer_streamlined_mpi<<<1,1>>>(d_list,a,b,c,d_camera,list_of_bounding_boxes,
        nx,ny,no_of_objects_in_the_scene);
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());
        std::cerr<<"\nEnded the creation of the world";
        
        host_bvh_tree_creation(list_of_bounding_boxes,
        bvh_arr,no_of_objects_in_the_scene);

        initialize_bvh_tree_streamlined<<<1,1>>>(bvh_arr,d_list,bvh_tree
        ,total_no_of_tree_nodes);

        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());

        std::cerr<<"\nTree creation ends";


        gpuErrchk(cudaMemcpy(h_bvh_arr,bvh_arr,total_size_bvh,cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(h_list,d_list,sizeof(triangle)*no_of_objects_in_the_scene,cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(h_camera,d_camera,sizeof(camera),cudaMemcpyDeviceToHost));

        MPI_Bcast(h_camera,sizeof(camera),MPI_BYTE,0,MPI_COMM_WORLD);
        MPI_Bcast(h_bvh_arr,total_size_bvh,MPI_BYTE,0,MPI_COMM_WORLD);
        MPI_Bcast(h_list,sizeof(triangle)*no_of_objects_in_the_scene,MPI_BYTE,0,MPI_COMM_WORLD);

        
    }
    else
    {
        MPI_Bcast(h_camera,sizeof(camera),MPI_BYTE,0,MPI_COMM_WORLD);
        MPI_Bcast(h_bvh_arr,total_size_bvh,MPI_BYTE,0,MPI_COMM_WORLD);
        MPI_Bcast(h_list,sizeof(triangle)*no_of_objects_in_the_scene,MPI_BYTE,0,MPI_COMM_WORLD);
        
        gpuErrchk(cudaMemcpy(d_list,h_list,sizeof(triangle)*no_of_objects_in_the_scene,cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(bvh_arr,h_bvh_arr,total_size_bvh,cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_camera,h_camera,sizeof(camera),cudaMemcpyHostToDevice));


        initialize_bvh_tree_streamlined<<<1,1>>>(bvh_arr,d_list,bvh_tree
        ,total_no_of_tree_nodes);

        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());

        //all the datatypes are now brodcasted to all the processes now we will start the rendering
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double elapsed_time_pre_processing = (end_time-start_time);

    double global_elapsed_time_pre_processing;
    MPI_Reduce(&elapsed_time_pre_processing, &global_elapsed_time_pre_processing, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); // Aggregate the elapsed times from all processes and get the maximum value
    //device memory check for d_list bvh_arr d_camera
    
    start_time = MPI_Wtime();

    int final_pixel_arr_size  = num_pixels*sizeof(vec3);
    int chunk_final_pixel_arr_size = (final_pixel_arr_size)/(size);
    //initializing the pixel arrays
    vec3 *device_chunk_anti_alias_pixel_arr;
    vec3 *host_final_pixel_arr,*device_chunk_final_pixel_arr;
    vec3 *host_chunk_final_pixel_arr;

    //host memory aLLocation
    host_final_pixel_arr = (vec3*)malloc(final_pixel_arr_size);
    host_chunk_final_pixel_arr = (vec3*)malloc(chunk_final_pixel_arr_size);
    
    MPI_Scatter(host_final_pixel_arr,chunk_final_pixel_arr_size,MPI_BYTE,
    host_chunk_final_pixel_arr, chunk_final_pixel_arr_size, MPI_BYTE, 0, MPI_COMM_WORLD);

    int chunk_anti_alias_size = ns*chunk_final_pixel_arr_size;

    //memory allocation
    gpuErrchk(cudaMalloc(&device_chunk_anti_alias_pixel_arr,chunk_anti_alias_size));
    gpuErrchk(cudaMalloc(&device_chunk_final_pixel_arr,chunk_final_pixel_arr_size));

    // //for cases when sampling not enabled
    // // dim3 blocks_render(8,8);
    // // dim3 grid_render(nx/8,ny/8);

    //sampling
    dim3 blocks_render(ns_x,ns_y);
    dim3 grid_render(nx,(ny/size));
    

    dim3 blocks_reduction = ns_x*ns_y;
    dim3 grid_reduction = nx*(ny/size);

    optimized_render_streamlined_tree_sampling_mpi<<<grid_render,blocks_render>>>(device_chunk_anti_alias_pixel_arr,
    nx,ny,d_camera,bvh_tree,d_list,rank,(ny/size));

    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    //starting our reduction kernel to find our final pixel array
    shared_reduction_interleaved_approach_complete_unrolling_2d_vec3<<<grid_reduction,blocks_reduction>>>(device_chunk_anti_alias_pixel_arr,
    device_chunk_final_pixel_arr,num_pixels*ns);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    gpuErrchk(cudaMemcpy(host_chunk_final_pixel_arr,device_chunk_final_pixel_arr,chunk_final_pixel_arr_size,
    cudaMemcpyDeviceToHost));
    
    MPI_Gather(host_chunk_final_pixel_arr, chunk_final_pixel_arr_size,MPI_BYTE, 
    host_final_pixel_arr,chunk_final_pixel_arr_size,MPI_BYTE,0, MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    double elapsed_time_rendering = (end_time-start_time);
    if(rank==0)
    {
        std::cerr<<"\nThe total time taken for pre processing was - "<<elapsed_time_pre_processing<<std::endl;
        std::cerr<<"The total time taken for rendering was - "<<elapsed_time_rendering<<std::endl;
        
        //for now 
        int *max_distance = new int(1);
        bool greyscale_visualization = false;
        std::cout<<"P3\n"<<nx<<" "<<ny<<"\n255\n";
        for(int j= ny-1;j>=0;j--){
            for(int i = 0;i<nx;i++){
                size_t pixel_index = j*nx + i;
                size_t host_file_index = pixel_index*3;
                float r = host_final_pixel_arr[pixel_index].r();
                float g = host_final_pixel_arr[pixel_index].g();
                float b = host_final_pixel_arr[pixel_index].b();
                if(greyscale_visualization)
                {
                    if(r>0.0f)
                        r/=*max_distance;
                    if(g>0.0f)
                        g/=*max_distance;
                    if(b>0.0f)
                        b/=*max_distance;
                }
                int ir = int(255.99*r);
                int ig = int(255.99*g);
                int ib = int(255.99*b);

                
                std::cout<<ir<<" "<<ig<<" "<<ib<<std::endl;
            }
        }
        
    }
    MPI_Finalize();
    
    return 0;
    // //for cases when sampling not enabled
    // // dim3 blocks_render(8,8);
    // // dim3 grid_render(nx/8,ny/8);

    // //sampling
    // dim3 blocks_render(ns_x,ns_y);
    // dim3 grid_render(nx,ny);
    

    // dim3 blocks_reduction = ns_x*ns_y;
    // dim3 grid_reduction = nx*ny;
    
    // size_t anti_alias_pixel_arr_size = num_pixels*ns*sizeof(vec3);
    // size_t final_pixel_arr_size  = num_pixels*sizeof(vec3);

    // vec3 *device_anti_alias_pixel_arr;
    // vec3 *host_final_pixel_arr,*device_final_pixel_arr;

    // //host memory aLLocation
    // host_final_pixel_arr = (vec3*)malloc(final_pixel_arr_size);

    // //device memory allocation
    // gpuErrchk(cudaMalloc(&device_final_pixel_arr,final_pixel_arr_size));
    // gpuErrchk(cudaMalloc(&device_anti_alias_pixel_arr,anti_alias_pixel_arr_size));
    // clock_t start,end_render,end;
    // start = clock();
    // std::cerr << "Rendering a " << nx << "x" << ny << " image \n";
    // std::cerr << "Rendering started \n";

    // // optimized_render_tree<<<grid_render,blocks_render>>>(device_anti_alias_pixel_arr,
    // // nx,ny,ns,d_camera,bvh_tree,d_materials,d_textures);

    // // optimized_render_list<<<grid_render,blocks_render>>>(device_anti_alias_pixel_arr,
    // // nx,ny,ns,d_camera,d_world,d_materials,d_textures);
    // //*max_distance = FLT_MIN;
    // // optimized_render_streamlined_tree<<<grid_render,blocks_render>>>(device_final_pixel_arr,
    // // nx,ny,d_camera,bvh_tree,d_list,max_distance);

    // optimized_render_streamlined_tree_sampling<<<grid_render,blocks_render>>>(device_anti_alias_pixel_arr,
    // nx,ny,d_camera,bvh_tree,d_list);

    // // // optimized_render_streamlined_tree_sampling_sah<<<grid_render,blocks_render>>>(device_anti_alias_pixel_arr,
    // // // nx,ny,d_camera,bvh_tree_sah,d_list);

    // gpuErrchk(cudaGetLastError());
    // gpuErrchk(cudaDeviceSynchronize());
    // // end_render = clock();
    // // timer_seconds = ((double)(end_render-start))/CLOCKS_PER_SEC;
    // // std::cerr << "Rendering complete took " << timer_seconds<<"\n";
    
    // //starting our reduction kernel to find our final pixel array
    // shared_reduction_interleaved_approach_complete_unrolling_2d<<<grid_reduction,blocks_reduction>>>(device_anti_alias_pixel_arr,
    // device_final_pixel_arr,num_pixels*ns);
    // gpuErrchk(cudaGetLastError());
    // gpuErrchk(cudaDeviceSynchronize());
    
    // gpuErrchk(cudaMemcpy(host_final_pixel_arr,device_final_pixel_arr,final_pixel_arr_size,
    // cudaMemcpyDeviceToHost));
    // // end = clock();

    // // timer_seconds = ((double)(end-start))/CLOCKS_PER_SEC;
    // // std::cerr<<"Reduction complete took "<<timer_seconds<<" seconds in the whole process\n";
    // // std::cerr<<"Rendering complete took "<<timer_seconds<<" seconds in the whole process\n";

    
        
        
    // // bool greyscale_visualization = false;
    // // std::cout<<"P3\n"<<nx<<" "<<ny<<"\n255\n";
    // // for(int j= ny-1;j>=0;j--){
    // //     for(int i = 0;i<nx;i++){
    // //         size_t pixel_index = j*nx + i;
    // //         float r = host_final_pixel_arr[pixel_index].r();
    // //         float g = host_final_pixel_arr[pixel_index].g();
    // //         float b = host_final_pixel_arr[pixel_index].b();
    // //         if(greyscale_visualization)
    // //         {
    // //             if(r>0.0f)
    // //                 r/=*max_distance;
    // //             if(g>0.0f)
    // //                 g/=*max_distance;
    // //             if(b>0.0f)
    // //                 b/=*max_distance;
    // //         }
    // //         int ir = int(255.99*r);
    // //         int ig = int(255.99*g);
    // //         int ib = int(255.99*b);

            
    // //         std::cout<<ir<<" "<<ig<<" "<<ib<<"\n";
    // //     }
    // // }

    

}