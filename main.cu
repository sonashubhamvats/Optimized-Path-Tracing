#define _USE_MATH_DEFINES
#include "cuda_common.cuh"
#include "constants.cuh"
#include <float.h>
#include <time.h>
#include "camera.cuh"
#include "texture.cuh"
#include "sphere.cuh"
#include "collider_list.cuh"
#include "materials.cuh"
#include "moving_spheres.cuh"
#include "rt_stb_image.h"
#include "bvh_gpu.cuh"
#include "bvh_cpu.cuh"
#include "aarect.cuh"
#include "box.cuh"
#include "constant_medium.cuh"
#include "./src/include/GLFW/glfw3.h"
#include "./glfw_code/show_output_glfw.cuh"
#include <fstream>
#include <string>
#include <cmath>
#include <sstream>

__device__ vec3 optimized_color_list(const ray &r, collider_list **d_world,
curandState *temp_state,vec3 &background, collider_material *d_material, collider_texture *d_texture)
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
        
        if((*d_world)->hit(cur_ray,0.001,FLT_MAX,rec))
        {
            int index_on_the_collider_list = rec.index_on_the_collider_list;
            if(d_material[index_on_the_collider_list].scatter(cur_ray,rec,
            temp_scattered_ray,temp_state)){
                temp_attentuation = 
                d_texture[index_on_the_collider_list].value(rec.u,rec.v,rec.p);
                curr_atten = curr_atten*temp_attentuation;
                cur_ray = temp_scattered_ray;
            }
            else
            {
                emitted = d_material[index_on_the_collider_list].emitted();
                return curr_atten*emitted;
            }
            
        }
        else
        {
            return background*curr_atten;
            
        }
    }
    return vec3(0,0,0);
}

__device__ vec3 optimized_color_tree(const ray &r, bvh_gpu *bvh_tree,
curandState *temp_state,vec3 &background, collider_material *d_material, collider_texture *d_texture)
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
        
        if(bvh_tree->hit(cur_ray,0.001,FLT_MAX,rec,0))
        {
            int index_on_the_collider_list = rec.index_on_the_collider_list;
            //printf("\n%d",index_on_the_collider_list);
            if(d_material[index_on_the_collider_list].scatter(cur_ray,rec,
            temp_scattered_ray,temp_state)){
                temp_attentuation = 
                d_texture[index_on_the_collider_list].value(rec.u,rec.v,rec.p);
                curr_atten = curr_atten*temp_attentuation;
                cur_ray = temp_scattered_ray;
            }
            else
            {
                
                emitted = d_material[index_on_the_collider_list].emitted();
                return curr_atten*emitted;
            }
            
        }
        else
        {
            return background*curr_atten;
            
        }
    }
    return vec3(0,0,0);
}

__device__ vec3 optimized_color_tree_streamlined_sampling(const ray &r, bvh_gpu_streamlined_tree *bvh_tree,
curandState *temp_state,vec3 &background,int s[])
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

__device__ vec3 optimized_color_tree_streamlined_sampling_sah(const ray &r, bvh_gpu_streamlined_tree_sah *bvh_tree,
curandState *temp_state,vec3 &background,int s[])
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
            curr_atten = vec3(0.721f,0.525f,0.043f)*curr_atten;
            
        }
        else
        {
            return background*curr_atten;
        }
    }
    return vec3(0,0,0);
}


__device__ vec3 optimized_color_tree_streamlined(const ray &r, bvh_gpu_streamlined_tree *bvh_tree,
curandState *temp_state,vec3 &background,int s[])
{
    ray cur_ray = r;
    vec3 curr_atten(1.0,1.0,1.0);
    ray temp_scattered_ray;
    vec3 temp_attentuation,emitted;
    collider_record rec;
    int itr = 0;
    float t;
    vec3 unit_direction;
    
    for(;itr<5;itr++)
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

__device__ vec3 optimized_color_tree_streamlined_zero_depth(const ray &r, bvh_gpu_streamlined_tree *bvh_tree,
vec3 &background,int s[])
{
    collider_record rec;
    if(bvh_tree->hit(r,0.001,FLT_MAX,rec,0,s))
    {
        return vec3(0.98f,0.360f,0.360f);
        
    }
    else
    {
        return background;
    }
}

__device__ vec3 optimized_color_tree_streamlined_variable_depth(const ray &r, bvh_gpu_streamlined_tree *bvh_tree,
vec3 &background,curandState *temp_state,int s[],int depth)
{
    ray cur_ray = r;
    vec3 curr_atten(1.0,1.0,1.0);
    ray temp_scattered_ray;
    vec3 temp_attentuation,emitted;
    collider_record rec;
    int itr = 0;
    float t;
    vec3 unit_direction;
    
    for(;itr<depth;itr++)
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
//non recursive function for streamlined rendering
__device__ vec3 optimized_color_tree_streamlined_sah(const ray &r, bvh_gpu_streamlined_tree_sah *bvh_tree,
curandState *temp_state,vec3 &background,int s[],vec3 camera_pos,float *max_distance)
{
    ray cur_ray = r;
    vec3 curr_atten(1.0,1.0,1.0);
    ray temp_scattered_ray;
    vec3 temp_attentuation,emitted;
    collider_record rec;
    int itr = 0;
    float t;
    vec3 unit_direction;
    if(bvh_tree->hit(cur_ray,0.001,FLT_MAX,rec,0,s))
    {
        
        float intersection_distance = sqrt((rec.p - camera_pos).squared_length());
        *max_distance = fmaxf(*max_distance,intersection_distance);
        return vec3(0.7f,0.7f,0.7f)*intersection_distance;
        //printf("\nHit");
    }
    else
    {
        return vec3(0.0f,0.0f,0.0f);
        //printf("\nNo Hit");
    }
    
    
}


__device__ vec3 optimized_color_tree_streamlined_recursive_sah(const ray &r, bvh_gpu_streamlined_tree_sah *bvh_tree,
curandState *temp_state,vec3 &background,int s[])
{
    ray cur_ray = r;
    vec3 curr_atten(1.0,1.0,1.0);
    ray temp_scattered_ray;
    vec3 temp_attentuation,emitted;
    collider_record rec;
    int itr = 0;
    float t;
    vec3 unit_direction;
    
    for(;itr<5;itr++)
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
__device__ vec3 optimized_color_tree_streamlined_recursive_sah_metal(const ray &r, bvh_gpu_streamlined_tree_sah *bvh_tree,
curandState *temp_state,vec3 &background,int s[])
{
    ray cur_ray = r;
    vec3 curr_atten(1.0,1.0,1.0);
    ray temp_scattered_ray;
    vec3 temp_attentuation,emitted;
    collider_record rec;
    int itr = 0;
    float t;
    vec3 unit_direction;
    for(;itr<5;itr++)
    {
        
        if(bvh_tree->hit(cur_ray,0.001,FLT_MAX,rec,0,s))
        {

            vec3 reflected = reflect(unit_vector(cur_ray.direction()), rec.normal);
            cur_ray = ray(rec.p, reflected,cur_ray.time());
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

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    // int i = threadIdx.x + blockIdx.x * blockDim.x;
    // int j = threadIdx.y + blockIdx.y * blockDim.y;
    // if((i >= max_x) || (j >= max_y)) return;
    // int pixel_index = j*max_x + i;
    int total_no_threads_in_a_block = blockDim.x*blockDim.y;
    int total_no_threads_in_a_row = total_no_threads_in_a_block*gridDim.x;
    int pixel_index = threadIdx.x + threadIdx.y*blockDim.x + total_no_threads_in_a_block*blockIdx.x+
    total_no_threads_in_a_row*blockIdx.y;
    curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);
}


__global__ void optimized_render_list(vec3 *anti_alias_pixel_arr,int nx,int ny,int ns,camera *d_camera
, collider_list **d_world, collider_material *d_material, collider_texture *d_texture)
{    

    if(blockIdx.x>=nx||blockIdx.y>=ny)
        return;

    int total_no_threads_in_a_block = blockDim.x*blockDim.y;
    int total_no_threads_in_a_row = total_no_threads_in_a_block*gridDim.x;
    int anti_alias_pixel_index = threadIdx.x + threadIdx.y*blockDim.x + total_no_threads_in_a_block*blockIdx.x+
    total_no_threads_in_a_row*blockIdx.y;
    int rand_state_index = anti_alias_pixel_index;

    curandState local_state;
    curand_init(1984+rand_state_index, 0, 0, &local_state);


    vec3 col(0,0,0);
    float u,v;

    
    u = float(blockIdx.x+curand_uniform(&local_state))/float(nx);
    v = float(blockIdx.y+curand_uniform(&local_state))/float(ny);
    
    //vec3 background(0.0,0.0,0.0);
    vec3 background(0.70, 0.80, 1.00);
    
    ray r(vec3(0,0,0),vec3(0,0,0));
    (d_camera)->get_ray(u,v,&local_state,r);
    col=optimized_color_list(r,d_world,&local_state,background,d_material,d_texture);
    


    anti_alias_pixel_arr[anti_alias_pixel_index] = col;

}

__global__ void optimized_render_tree(vec3 *anti_alias_pixel_arr,int nx,int ny,int ns,camera *d_camera
, bvh_gpu *bvh_tree, collider_material *d_material, collider_texture *d_texture)
{    
    
    if(blockIdx.x>=nx||blockIdx.y>=ny)
        return;

    int total_no_threads_in_a_block = blockDim.x*blockDim.y;
    int total_no_threads_in_a_row = total_no_threads_in_a_block*gridDim.x;
    int anti_alias_pixel_index = threadIdx.x + threadIdx.y*blockDim.x + total_no_threads_in_a_block*blockIdx.x+
    total_no_threads_in_a_row*blockIdx.y;
    int rand_state_index = anti_alias_pixel_index;

    curandState local_state;
    curand_init(1984+rand_state_index, 0, 0, &local_state);


    vec3 col(0,0,0);
    float u,v;

   
    u = float(blockIdx.x+curand_uniform(&local_state))/float(nx);
    v = float(blockIdx.y+curand_uniform(&local_state))/float(ny);
    
    //vec3 background(0.0,0.0,0.0);
    vec3 background(0.70, 0.80, 1.00);
    
    ray r(vec3(0,0,0),vec3(0,0,0));
    (d_camera)->get_ray(u,v,&local_state,r);
    col=optimized_color_tree(r,bvh_tree,&local_state,background,d_material,d_texture);
    
    anti_alias_pixel_arr[anti_alias_pixel_index] = col;

}

//same but sampling of a streamlined renderer
__global__ void optimized_render_streamlined_tree_sampling(vec3 *anti_alias_pixel_arr,int nx,int ny,camera *d_camera,
bvh_gpu_streamlined_tree *bvh_tree)
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
    v = float(blockIdx.y+curand_uniform(&local_state))/float(ny);
    
    ray r(vec3(0,0,0),vec3(0,0,0));
    (d_camera)->get_ray(u,v,&local_state,r);

    //__shared__ int s[15];
    int s[25];

    col=optimized_color_tree_streamlined_sampling(r,bvh_tree,&local_state,background,s);
    

    anti_alias_pixel_arr[anti_alias_pixel_index] = col;
}


__global__ void optimized_render_streamlined_tree_sampling_sah(vec3 *anti_alias_pixel_arr,int nx,int ny,camera *d_camera,
bvh_gpu_streamlined_tree_sah *bvh_tree)
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
    v = float(blockIdx.y+curand_uniform(&local_state))/float(ny);
    
    ray r(vec3(0,0,0),vec3(0,0,0));
    (d_camera)->get_ray(u,v,&local_state,r);

    
    int s[25];
    col=optimized_color_tree_streamlined_sampling_sah(r,bvh_tree,&local_state,background,s);
    

    anti_alias_pixel_arr[anti_alias_pixel_index] = col;
}

__global__ void optimized_render_streamlined_tree_large_loads_sampling(vec3 *anti_alias_pixel_arr,int nx,int ny,camera *d_camera
, bvh_gpu_streamlined_tree *bvh_tree,int depth)
{    
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
    v = float(blockIdx.y+curand_uniform(&local_state))/float(ny);
    
    ray r(vec3(0,0,0),vec3(0,0,0));
    (d_camera)->get_ray(u,v,&local_state,r);

    
    int s[25];
    if(depth==0)
        col+=optimized_color_tree_streamlined_zero_depth(r,bvh_tree,background,s);
    else
        col+=optimized_color_tree_streamlined_variable_depth(r,bvh_tree,background,&local_state,s,depth);

    anti_alias_pixel_arr[anti_alias_pixel_index] = col;

}

__global__ void optimized_render_streamlined_tree_large_loads(vec3 *final_pixel_arr,int nx,int ny,camera *d_camera
, bvh_gpu_streamlined_tree *bvh_tree)
{    
    int total_no_threads_in_single_row = nx;
    int final_pixel_index = threadIdx.x + blockDim.x*blockIdx.x + nx*(threadIdx.y+blockIdx.y*blockDim.y);

    int pixel_x = threadIdx.x + blockIdx.x*blockDim.x;
    int pixel_y = threadIdx.y + blockIdx.y*blockDim.y;

    curandState local_state;
    curand_init(1984+final_pixel_index, 0, 0, &local_state);


    vec3 col(0,0,0);
    float u,v;
    vec3 background(0.70, 0.80, 1.00);
    int s[25];
    for(int i=0;i<10;i++)
    {
        u = float(pixel_x+curand_uniform(&local_state))/float(nx);
        v = float(pixel_y+curand_uniform(&local_state))/float(ny);
    

        ray r(vec3(0,0,0),vec3(0,0,0));
        (d_camera)->get_ray(u,v,&local_state,r);
        
        col+=optimized_color_tree_streamlined_zero_depth(r,bvh_tree,background,s);
    }

    vec3 final_col=col;
    final_col /= 10.0f;
    final_col[0] = sqrt(final_col[0]);
    final_col[1] = sqrt(final_col[1]);
    final_col[2] = sqrt(final_col[2]);
    final_pixel_arr[final_pixel_index] = final_col;

}

__global__ void optimized_render_streamlined_tree_sah(vec3 *final_pixel_arr,int nx,int ny,camera *d_camera
, bvh_gpu_streamlined_tree_sah *bvh_tree,float *max_distance,int ns)
{    

    int total_no_threads_in_single_row = nx;
    int final_pixel_index = threadIdx.x + blockDim.x*blockIdx.x + nx*(threadIdx.y+blockIdx.y*blockDim.y);

    int pixel_x = threadIdx.x + blockIdx.x*blockDim.x;
    int pixel_y = threadIdx.y + blockIdx.y*blockDim.y;
    

    curandState local_state;
    curand_init(1984+final_pixel_index, 0, 0, &local_state);


    vec3 col(0,0,0);
    float u,v;
    vec3 background(0.70, 0.80, 1.00);
    int s[25];
    for(int i=0;i<ns;i++)
    {
        u = float(pixel_x+curand_uniform(&local_state))/float(nx);
        v = float(pixel_y+curand_uniform(&local_state))/float(ny);
    

        ray r(vec3(0,0,0),vec3(0,0,0));
        (d_camera)->get_ray(u,v,&local_state,r);
        col+=optimized_color_tree_streamlined_sah(r,bvh_tree,&local_state,background,s,d_camera->origin,max_distance);
    }

    vec3 final_col=col;
    final_col /= ns;
    // final_col[0] = sqrt(final_col[0]);
    // final_col[1] = sqrt(final_col[1]);
    // final_col[2] = sqrt(final_col[2]);
    final_pixel_arr[final_pixel_index] = final_col;

}

__global__ void optimized_render_streamlined_tree_recursive_sah(vec3 *anti_alias_pixel_arr,int nx,int ny,camera *d_camera
, bvh_gpu_streamlined_tree_sah *bvh_tree)
{    

    int total_no_threads_in_a_block = blockDim.x*blockDim.y;
    int total_no_threads_in_a_row = total_no_threads_in_a_block*gridDim.x;
    int anti_alias_pixel_index = threadIdx.x + threadIdx.y*blockDim.x + total_no_threads_in_a_block*blockIdx.x+
    total_no_threads_in_a_row*blockIdx.y;

    int pixel_x = blockDim.y*blockIdx.x + threadIdx.y;
    int pixel_y = blockIdx.y;
    

    curandState local_state;
    curand_init(1984+anti_alias_pixel_index, 0, 0, &local_state);


    vec3 col(0,0,0);
    float u,v;
    vec3 background(0.70, 0.80, 1.00);
    int s[15];
    
    u = float(pixel_x+curand_uniform(&local_state))/float(nx);
    v = float(pixel_y+curand_uniform(&local_state))/float(ny);


    ray r(vec3(0,0,0),vec3(0,0,0));
    (d_camera)->get_ray(u,v,&local_state,r);
    col+=optimized_color_tree_streamlined_recursive_sah(r,bvh_tree,&local_state,background,s);
    

   
    anti_alias_pixel_arr[anti_alias_pixel_index] = col;

}

__global__ void optimized_render_streamlined_tree_recursive_sah_metal(vec3 *anti_alias_pixel_arr,int nx,int ny,camera *d_camera
, bvh_gpu_streamlined_tree_sah *bvh_tree)
{    

    // int total_no_threads_in_single_row = nx;
    // int final_pixel_index = threadIdx.x + blockDim.x*blockIdx.x + nx*(threadIdx.y+blockIdx.y*blockDim.y);

    // int pixel_x = threadIdx.x + blockIdx.x*blockDim.x;
    // int pixel_y = threadIdx.y + blockIdx.y*blockDim.y;
    

    // curandState local_state;
    // curand_init(1984+final_pixel_index, 0, 0, &local_state);


    // vec3 col(0,0,0);
    // float u,v;
    // vec3 background(0.70, 0.80, 1.00);
    // int s[25];
    // for(int i=0;i<4;i++)
    // {
    //     u = float(pixel_x+curand_uniform(&local_state))/float(nx);
    //     v = float(pixel_y+curand_uniform(&local_state))/float(ny);
    

    //     ray r(vec3(0,0,0),vec3(0,0,0));
    //     (d_camera)->get_ray(u,v,&local_state,r);
    //     col+=optimized_color_tree_streamlined_recursive_sah_metal(r,bvh_tree,&local_state,background,s);
    // }

    // vec3 final_col=col;
    // final_col /= 4.0f;
    // // final_col[0] = sqrt(final_col[0]);
    // // final_col[1] = sqrt(final_col[1]);
    // // final_col[2] = sqrt(final_col[2]);
    // final_pixel_arr[final_pixel_index] = final_col;

    int total_no_threads_in_a_block = blockDim.x*blockDim.y;
    int total_no_threads_in_a_row = total_no_threads_in_a_block*gridDim.x;
    int anti_alias_pixel_index = threadIdx.x + threadIdx.y*blockDim.x + total_no_threads_in_a_block*blockIdx.x+
    total_no_threads_in_a_row*blockIdx.y;

    int pixel_x = blockDim.y*blockIdx.x + threadIdx.y;
    int pixel_y = blockIdx.y;
    

    curandState local_state;
    curand_init(1984+anti_alias_pixel_index, 0, 0, &local_state);


    vec3 col(0,0,0);
    float u,v;
    vec3 background(0.70, 0.80, 1.00);
    int s[15];
    
    u = float(pixel_x+curand_uniform(&local_state))/float(nx);
    v = float(pixel_y+curand_uniform(&local_state))/float(ny);


    ray r(vec3(0,0,0),vec3(0,0,0));
    (d_camera)->get_ray(u,v,&local_state,r);
    col+=optimized_color_tree_streamlined_recursive_sah_metal(r,bvh_tree,&local_state,background,s);
    

   
    anti_alias_pixel_arr[anti_alias_pixel_index] = col;

}

__global__ void cmd_master_renderer(colliders *d_colliders,collider_material *d_materials,
collider_texture *d_textures, collider_list **d_world, 
camera *d_camera,aabb *list_of_bounding_boxes,float nx,float ny,int size,
vec3 lookfrom,vec3 lookat,float aperture,vec3 vup,float fov,int option_number
,unsigned char *device_data,int width,int height)
{
    curandState localState;

    curand_init(1984,0,0,&localState);
    

    int first_initialization_texture = 1;
    int first_initialization_texture_dielectric_special = 1;
    int first_initialization_material = 1;
    int first_initialization_material_dielectric_special = 1;
    noise_texture *n;
    sphere *s;
    lambertian *l;
    metal *m;
    dielectric *d;
    solid_color *s_c;
    solid_color *s_c_white;
    constant_medium *c;
    checker_texture *c_t;
    triangle *t;
    isotropic *iso;
    image_texture *i_t;
    box *b;
    if(option_number==29)
    {
        for(int i=0;i<size-1;i++)
        {
            vec3 rand_point = random_vector_in_range(&localState,-12,12);
            if(i<3*(size/16))
            {
                //xz_plane positive spheres
                rand_point.e.y = 10;
                vec3 a_vec = rand_point - random_vector_in_range(&localState,-0.5,0.5);
                float rand_radius = random_double(&localState,0,0.8f);
                s = new sphere(a_vec,rand_radius,i);
                d_colliders[i] = colliders(s,sphere_type_index);
                l =new lambertian();
                d_materials[i] = collider_material(l,lambertian_type_index);
                

                float choice =random_double(&localState,0.0,3.0f);
                if(choice<1.0f)
                {
                    s_c = new solid_color(vec3(RND*1.0f,RND*1.0f,RND*1.0f));
                    d_textures[i] = collider_texture(s_c,solid_color_type_index);
                }
                else if(choice<2.0f)
                {
                    
                    c_t = new checker_texture(vec3(1.0f,0.2f,0.5f),vec3(0.2f,1.0f,0.5f));
                    d_textures[i] = collider_texture(c_t,checker_texture_type_index);
                }
                else
                {
                    
                    n = new noise_texture(localState,4.0f);
                    d_textures[i] = collider_texture(n,noise_texture_type_index);
                }


                d_colliders[i].bounding_box(0,0,
                list_of_bounding_boxes[i],i);
            }
            else if(i<6*(size/16))
            {
                //xz_plane negative spheres
                rand_point.e.y = -10;
                vec3 a_vec = rand_point - random_vector_in_range(&localState,-0.5,0.5);
                float rand_radius = random_double(&localState,0,0.5f);
                s = new sphere(a_vec,rand_radius,i);
                d_colliders[i] = colliders(s,sphere_type_index);
                float choice =random_double(&localState,0.0,1.0f);
                if(choice<0.5f)
                {
                    l =new lambertian();
                    d_materials[i] = collider_material(l,lambertian_type_index);
                    s_c = new solid_color(vec3(RND*1.0f,RND*1.0f,RND*1.0f));
                    d_textures[i] = collider_texture(s_c,solid_color_type_index);
                }   
                else
                {
                    if(first_initialization_material_dielectric_special-->0)
                        d = new dielectric(1.5f);
                    d_materials[i] = collider_material(d,dielectric_type_index);
                    if(first_initialization_texture_dielectric_special-->0)
                        s_c_white = new solid_color(vec3(1.0f,1.0f,1.0f));
                    d_textures[i] = collider_texture(s_c_white,solid_color_type_index);
                }
               
                d_colliders[i].bounding_box(0,0,
                list_of_bounding_boxes[i],i);
            }
            else if(i<9*(size/16))
            {
                //yz_plane positive spheres
                rand_point.e.x = 10;
                vec3 a_vec = rand_point - random_vector_in_range(&localState,-0.5,0.5);
                float rand_radius = random_double(&localState,0,0.8f);
                s = new sphere(a_vec,rand_radius,i);
                d_colliders[i] = colliders(s,sphere_type_index);
                
                m = new metal(0.0f);
                d_materials[i] = collider_material(m,metal_type_index);
                
                s_c = new solid_color(vec3(RND*1.0f,RND*1.0f,RND*1.0f));
                d_textures[i] = collider_texture(s_c,solid_color_type_index);

                d_colliders[i].bounding_box(0,0,
                list_of_bounding_boxes[i],i);
            }
            else if(i<12*(size/16))
            {
                //yz_plane negative spheres
                rand_point.e.x = -10;
                vec3 a_vec = rand_point - random_vector_in_range(&localState,-0.5,0.5);
                float rand_radius = random_double(&localState,0,0.8f);
                s = new sphere(a_vec,rand_radius,i);
                d_colliders[i] = colliders(s,sphere_type_index);
                l =new lambertian();
                d_materials[i] = collider_material(l,lambertian_type_index);
                float choice =random_double(&localState,0.0,3.0f);
                if(choice<1.0f)
                {
                    s_c = new solid_color(vec3(RND*1.0f,RND*1.0f,RND*1.0f));
                    d_textures[i] = collider_texture(s_c,solid_color_type_index);
                }  
                else
                {
                    i_t = new image_texture(&device_data,width,height);
                    d_textures[i] = collider_texture(i_t,image_texture_type_index);
                }

                d_colliders[i].bounding_box(0,0,
                list_of_bounding_boxes[i],i);
            }
            else if(i<13*(size/16))
            {
                //xz_plane positive triangles
                rand_point = random_vector_in_range(&localState,-4,4);
                rand_point.e.y = 4.0f;
                vec3 a_vec = rand_point - random_vector_in_range(&localState,-0.5,0.5);
                vec3 b_vec = rand_point - random_vector_in_range(&localState,-0.5,0.5);
                vec3 c_vec = rand_point - random_vector_in_range(&localState,-0.5,0.5);
                

                t = new triangle(a_vec,b_vec,c_vec,i);
                d_colliders[i] = colliders(t,triangle_type_index);
                l =new lambertian();
                d_materials[i] = collider_material(l,lambertian_type_index);
                s_c = new solid_color(vec3(RND*1.0f,RND*1.0f,RND*1.0f));
                d_textures[i] = collider_texture(s_c,solid_color_type_index);

                d_colliders[i].bounding_box(0,0,
                list_of_bounding_boxes[i],i);
            }
            else if(i<14*(size/16))
            {
                //xz_plane negative triangles
                rand_point = random_vector_in_range(&localState,-4,4);
                rand_point.e.y = -4.0f;
                vec3 a_vec = rand_point - random_vector_in_range(&localState,-0.5,0.5);
                vec3 b_vec = rand_point - random_vector_in_range(&localState,-0.5,0.5);
                vec3 c_vec = rand_point - random_vector_in_range(&localState,-0.5,0.5);

                t = new triangle(a_vec,b_vec,c_vec,i);
                d_colliders[i] = colliders(t,triangle_type_index);
                l =new lambertian();
                d_materials[i] = collider_material(l,lambertian_type_index);
                s_c = new solid_color(vec3(RND*1.0f,RND*1.0f,RND*1.0f));
                d_textures[i] = collider_texture(s_c,solid_color_type_index);
                

                d_colliders[i].bounding_box(0,0,
                list_of_bounding_boxes[i],i);
            }
            else if(i<15*(size/16))
            {
                //yz_plane positive triangles
                rand_point = random_vector_in_range(&localState,-4,4);
                rand_point.e.x = 4;
                vec3 a_vec = rand_point - random_vector_in_range(&localState,-0.5,0.5);
                vec3 b_vec = rand_point - random_vector_in_range(&localState,-0.5,0.5);
                vec3 c_vec = rand_point - random_vector_in_range(&localState,-0.5,0.5);

                t = new triangle(a_vec,b_vec,c_vec,i);
                d_colliders[i] = colliders(t,triangle_type_index);
                t = new triangle(a_vec,b_vec,c_vec,i);
                d_colliders[i] = colliders(t,triangle_type_index);

                m = new metal(0.0f);
                d_materials[i] = collider_material(m,metal_type_index);
                
                s_c = new solid_color(vec3(RND*1.0f,RND*1.0f,RND*1.0f));
                d_textures[i] = collider_texture(s_c,solid_color_type_index);

                d_colliders[i].bounding_box(0,0,
                list_of_bounding_boxes[i],i);
            }
            else if(i<size)
            {
                //yz_plane negative triangles
                rand_point = random_vector_in_range(&localState,-4,4);
                rand_point.e.x = -4;
                vec3 a_vec = rand_point - random_vector_in_range(&localState,-0.5,0.5);
                vec3 b_vec = rand_point - random_vector_in_range(&localState,-0.5,0.5);
                vec3 c_vec = rand_point - random_vector_in_range(&localState,-0.5,0.5);

                t = new triangle(a_vec,b_vec,c_vec,i);
                d_colliders[i] = colliders(t,triangle_type_index);

                float choice = random_double(&localState,0.0f,1.0f);
                
                l =new lambertian();
                d_materials[i] = collider_material(l,lambertian_type_index);
                s_c = new solid_color(vec3(RND*1.0f,RND*1.0f,RND*1.0f));
                d_textures[i] = collider_texture(s_c,solid_color_type_index);
            

                d_colliders[i].bounding_box(0,0,
                list_of_bounding_boxes[i],i);
            }
            

        }
        b = new box(vec3(-2,-2,-2),vec3(2,2,2),size-1);
        d_colliders[size-1] = colliders(b,box_index);
        // l =new lambertian();
        // d_materials[size-1] = collider_material(l,lambertian_type_index);
        // n = new noise_texture(localState,4.0f);
        // d_textures[size-1] = collider_texture(n,noise_texture_type_index);
        diffuse_light *light = new diffuse_light(vec3(17,17,17));
        d_materials[size-1] = collider_material(light,diffuse_type_index);
        d_colliders[size-1].bounding_box(0,0,
                list_of_bounding_boxes[size-1],size-1);

        float dist_to_focus = (lookfrom-lookat).length();
        *d_camera = camera(lookfrom,
                        lookat,
                        vec3(0,1,0),
                        fov,
                        float(nx)/float(ny),
                        aperture,
                        dist_to_focus,0.0,1.0);

        return;
    }
    for(int i = 0; i<size;i++)
    {
        vec3 rand_point = random_vector_in_range(&localState,-10,10);

        vec3 a_vec = rand_point - random_vector_in_range(&localState,-0.5,0.5);
        vec3 b_vec = rand_point - random_vector_in_range(&localState,-0.5,0.5);
        vec3 c_vec = rand_point - random_vector_in_range(&localState,-0.5,0.5);

        float rand_no_dielectric = random_double(&localState,0,1.0f);
        float rand_radius = random_double(&localState,0,0.6f);
        
        if(option_number<=4||option_number>=13&&option_number<=15)
        {
            s = new sphere(a_vec,rand_radius,i);
            d_colliders[i] = colliders(s,sphere_type_index);
        }
        else if(option_number>=5&&option_number<=8||option_number>=16&&option_number<=17)
        {
            t = new triangle(a_vec,b_vec,c_vec,i);
            d_colliders[i] = colliders(t,triangle_type_index);
        }
        else
        {
            b = new box(a_vec,a_vec+vec3(RND*1.0f,RND*1.0f,RND*1.0f),i);
            d_colliders[i] = colliders(b,box_index);
        }

        if(option_number==15||option_number==21)
        {
            if(option_number==15)
            {
                c = new constant_medium(s,0.5f,i,&localState);
                d_colliders[i] = colliders(c,constant_medium_index);
            }
            else
            {
                c = new constant_medium(b,0.5f,i,&localState);
                d_colliders[i] = colliders(c,constant_medium_index);
            }
        }

        d_colliders[i].bounding_box(0,0,
        list_of_bounding_boxes[i],i);

        if(option_number<=12)
        {
            if(first_initialization_material-->0)
                l =new lambertian();
            d_materials[i] = collider_material(l,lambertian_type_index);
        }
        else if(option_number==13||option_number==16||option_number==19)
        {
            if(first_initialization_material-->0)
                m = new metal(0.0f);
            d_materials[i] = collider_material(m,metal_type_index);
        }
        else if(option_number==14||option_number==17||option_number==20)
        {
            if(rand_no_dielectric<0.5f)
            {
                if(first_initialization_material_dielectric_special-->0)
                    d = new dielectric(1.5f);
                d_materials[i] = collider_material(d,dielectric_type_index);
                if(first_initialization_texture_dielectric_special-->0)
                    s_c_white = new solid_color(vec3(1.0f,1.0f,1.0f));
                d_textures[i] = collider_texture(s_c_white,solid_color_type_index);
            }
            else
            {
                if(first_initialization_material-->0)
                    l =new lambertian();
                d_materials[i] = collider_material(l,lambertian_type_index);
                s_c = new solid_color(vec3(RND*1.0f,RND*1.0f,RND*1.0f));
                d_textures[i] = collider_texture(s_c,solid_color_type_index);
            }
            
        }
        else if(option_number==15||option_number==21)
        {
            if(first_initialization_material-->0)
                iso = new isotropic();
            d_materials[i] = collider_material(iso,isotropic_type_index);
        }
        

        if(option_number==1||option_number==13||option_number==15||option_number==16
        ||option_number==19||option_number==21||option_number==9||option_number==5)
        {   
            s_c = new solid_color(vec3(RND*1.0f,RND*1.0f,RND*1.0f));
            d_textures[i] = collider_texture(s_c,solid_color_type_index);
        }
        else if(option_number==2||option_number==6||option_number==10)
        {
            if(first_initialization_texture-->0)
                n = new noise_texture(localState,4.0f);
            d_textures[i] = collider_texture(n,noise_texture_type_index);
    
        }
        else if(option_number==3||option_number==7||option_number==11)
        {
            if(first_initialization_texture-->0)
                c_t = new checker_texture(vec3(1.0f,0.2f,0.5f),vec3(0.2f,1.0f,0.5f));
            d_textures[i] = collider_texture(c_t,checker_texture_type_index);
        }
        else if(option_number==4||option_number==8||option_number==12)
        {
            if(first_initialization_texture-->0)
                i_t = new image_texture(&device_data,width,height);
            d_textures[i] = collider_texture(i_t,image_texture_type_index);
        }
        
        

    }
    d_world[0] = new collider_list(d_colliders,size);
    float dist_to_focus = (lookfrom-lookat).length();
    *d_camera = camera(lookfrom,
                    lookat,
                    vec3(0,1,0),
                    fov,
                    float(nx)/float(ny),
                    aperture,
                    dist_to_focus,0.0,1.0);
}
__global__ void cmd_master_renderer_streamlined(triangle *d_list,vec3 *a,vec3 *b,vec3 *c,
camera *d_camera,aabb *list_of_bounding_boxes,int nx,int ny,int size,
vec3 lookfrom,vec3 lookat,float aperture,vec3 vup,float fov,bool random_or_not,
bool metal_or_not = false)
{
    curandState localState;
    curand_init(1984,0,0,&localState);
    
    triangle t;
    if(metal_or_not==true&&random_or_not==false)
    {
        for(int i=0;i<size;i++)
        {
            if(i<(size-50))
            {
                t = triangle(a[i],b[i],c[i],i);
            }
            else
            {
                vec3 rand_point=random_vector_in_range(&localState,-1.5,1.5);
                vec3 a_vec = rand_point - random_vector_in_range(&localState,-0.2,0.2);
                vec3 b_vec = rand_point - random_vector_in_range(&localState,-0.2,0.2);
                vec3 c_vec = rand_point - random_vector_in_range(&localState,-0.2,0.2);
                t = triangle(a_vec,b_vec,c_vec,i);
            }
            d_list[i] =t;
            t.bounding_box(0,0,
            list_of_bounding_boxes[i],i);
        }
        float dist_to_focus = (lookfrom-lookat).length();
        *d_camera = camera(lookfrom,
                    lookat,
                    vec3(0,1,0),
                    fov,
                    float(nx)/float(ny),
                    aperture,
                    dist_to_focus,0.0,1.0);
        return;
    }
    if(metal_or_not==true&&random_or_not==true)
    {
        for(int i=0;i<size;i++)
        {
            
            vec3 rand_point=random_vector_in_range(&localState,-10,10);
            vec3 a_vec = rand_point - random_vector_in_range(&localState,-0.5,0.5);
            vec3 b_vec = rand_point - random_vector_in_range(&localState,-0.5,0.5);
            vec3 c_vec = rand_point - random_vector_in_range(&localState,-0.5,0.5);
            t = triangle(a_vec,b_vec,c_vec,i);
            
            d_list[i] =t;
            t.bounding_box(0,0,
            list_of_bounding_boxes[i],i);
        }
        float dist_to_focus = (lookfrom-lookat).length();
        *d_camera = camera(lookfrom,
                    lookat,
                    vec3(0,1,0),
                    fov,
                    float(nx)/float(ny),
                    aperture,
                    dist_to_focus,0.0,1.0);
        return;
    }
    for(int i = 0; i<size;i++)
    {
        if(random_or_not)
        {
            vec3 rand_point;
            if(size>=50000)
                rand_point = random_vector_in_range(&localState,-40,40);
            else
                rand_point = random_vector_in_range(&localState,-10,10);

            vec3 a_vec = rand_point - random_vector_in_range(&localState,-0.5,0.5);
            vec3 b_vec = rand_point - random_vector_in_range(&localState,-0.5,0.5);
            vec3 c_vec = rand_point - random_vector_in_range(&localState,-0.5,0.5);
            t = triangle(a_vec,b_vec,c_vec,i);
        }
        else
        {
            t = triangle(a[i],b[i],c[i],i);
        }
        d_list[i] =t;
        t.bounding_box(0,0,
        list_of_bounding_boxes[i],i);

    }
    float dist_to_focus = (lookfrom-lookat).length();
    *d_camera = camera(lookfrom,
                lookat,
                vec3(0,1,0),
                fov,
                float(nx)/float(ny),
                aperture,
                dist_to_focus,0.0,1.0);

    
}
__global__ void shared_reduction_interleaved_approach_complete_unrolling_small(vec3 *input,unsigned char *temp,
int size){
    
    int tid = threadIdx.x;

    vec3 *i_data = input + blockIdx.x*blockDim.x;

    __syncthreads();

    if(tid<32)
    {
        if(blockDim.x>=64)
            i_data[tid]+=i_data[tid+32];
        __syncthreads();
        if(blockDim.x>=32)
            i_data[tid]+=i_data[tid+16];
        __syncthreads();
        if(blockDim.x>=16)
            i_data[tid]+=i_data[tid+8];
        __syncthreads();
        if(blockDim.x>=8)
            i_data[tid]+=i_data[tid+4];
        __syncthreads();
        if(blockDim.x>=4)
            i_data[tid]+=i_data[tid+2];
        __syncthreads();
        if(blockDim.x>=2)
            i_data[tid]+=i_data[tid+1];
        __syncthreads();
    }
    if(tid==0)
    {
        vec3 final_col=i_data[tid];

        final_col /= float(blockDim.x);
        // final_col[0] = sqrt(final_col[0]);
        // final_col[1] = sqrt(final_col[1]);
        // final_col[2] = sqrt(final_col[2]);
        
        int temp_index = blockIdx.x*3;
        
        temp[temp_index] = (int)clamp((final_col[0]*255.99),0.0,255.99);
        temp[temp_index+1] = (int)clamp((final_col[1]*255.99),0.0,255.99);
        temp[temp_index+2] = (int)clamp((final_col[2]*255.99),0.0,255.99);

        
        
    }
        
}


__global__ void shared_reduction_interleaved_approach_complete_unrolling_2d(vec3 *input,unsigned char *temp,
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
        if(blockDim.x>=64)
            i_data[tid]+=i_data[tid+32];
        __syncthreads();
        if(blockDim.x>=32)
            i_data[tid]+=i_data[tid+16];
        __syncthreads();
        if(blockDim.x>=16)
            i_data[tid]+=i_data[tid+8];
        __syncthreads();
        if(blockDim.x>=8)
            i_data[tid]+=i_data[tid+4];
        __syncthreads();
        if(blockDim.x>=4)
            i_data[tid]+=i_data[tid+2];
        __syncthreads();
        if(blockDim.x>=2)
            i_data[tid]+=i_data[tid+1];
        __syncthreads();
    }
    
    if(tid==0)
    {
        vec3 final_col=i_data[tid];

        final_col /= float(blockDim.x);
        final_col[0] = sqrt(final_col[0]);
        final_col[1] = sqrt(final_col[1]);
        final_col[2] = sqrt(final_col[2]);
        
        int temp_index = blockIdx.x*3;
        
        temp[temp_index] = (int)clamp((final_col[0]*255.99),0.0,255.99);
        temp[temp_index+1] = (int)clamp((final_col[1]*255.99),0.0,255.99);
        temp[temp_index+2] = (int)clamp((final_col[2]*255.99),0.0,255.99);

        
        
    }
        
}



__global__ void depth_visualization_kernel(vec3 *input,unsigned char *temp,
float max_distance){
    
    int pixel_index = threadIdx.x + blockIdx.x*blockDim.x;
    
    float r = input[pixel_index].r();
    float g = input[pixel_index].g();
    float b = input[pixel_index].b();
    
    if(r>0.0f)
        r/=max_distance;
    if(g>0.0f)
        g/=max_distance;
    if(b>0.0f)
        b/=max_distance;
    
    int ir = int(255.99*r);
    int ig = int(255.99*g);
    int ib = int(255.99*b);

    int temp_index = pixel_index*3;
    temp[temp_index] = ir;
    temp[temp_index+1] = ig;
    temp[temp_index+2] = ib;
    
        
}

__global__ void host_assign_kernel(vec3 *input,unsigned char *temp){
    
    int pixel_index = threadIdx.x + blockIdx.x*blockDim.x;
    
    float r = input[pixel_index].r();
    float g = input[pixel_index].g();
    float b = input[pixel_index].b();
    
    int ir = int(255.99*r);
    int ig = int(255.99*g);
    int ib = int(255.99*b);

    int temp_index = pixel_index*3;
    temp[temp_index] = ir;
    temp[temp_index+1] = ig;
    temp[temp_index+2] = ib;
    
        
}

__global__ void initialize_bvh_tree(bvh_gpu_node *b_arr,colliders *d_list,
bvh_gpu *bvh_tree,int size)
{
    bvh_tree[0] = bvh_gpu(b_arr,d_list,2*(2*size-1)+1);
}

__global__ void initialize_bvh_tree_streamlined(bvh_gpu_node *b_arr,triangle *d_list,
bvh_gpu_streamlined_tree *bvh_tree,int size)
{
    bvh_tree[0] = bvh_gpu_streamlined_tree(b_arr,d_list,size);
}

__global__ void initialize_bvh_tree_streamlined_sah(bvh_gpu_node_sah *b_arr,triangle *d_list,
bvh_gpu_streamlined_tree_sah *bvh_tree,int size_of_the_tree)
{
    bvh_tree[0] = bvh_gpu_streamlined_tree_sah(b_arr,d_list,size_of_the_tree);
}

// __global__ void free_world(hittable **d_list, hittable **d_world, camera **d_camera,int size) {
//     for(int i=0; i < size; i++) {
//         //delete ((sphere *)d_list[i])->mat_ptr;
//         delete d_list[i];
//     }
//     delete *d_world;
//     delete *d_camera;
// }

void read_from_image_stb(const char* filename,unsigned char **data,
int &width,int &height){
    int components_per_pixel = 3;
    *data = stbi_load(
        filename, &width, &height, 
        &components_per_pixel, components_per_pixel);

    if (!data) {
        std::cerr << "ERROR: Could not load texture image file '" << filename << "'.\n";
        width = height = 0;
    }
}
void bvh_tree_traversal(bvh_gpu_node *b_arr,int index,int size,int &counter){
    if(index>=size)
        return;
    // printf("\nIndex on the bvh_arr %d %d",b_arr[index].index_on_bvh_array,index);
    // printf("\nFor index- %d",b_arr[index].index_collider_list);
    // printf("\nx_min- %f  y_min- %f  z_min- %f\n",
    // b_arr[index].box.min().x(),
    // b_arr[index].box.min().y(),
    // b_arr[index].box.min().z());

    // printf("x_max- %f  y_max- %f  z_max- %f\n",
    // b_arr[index].box.max().x(),
    // b_arr[index].box.max().y(),
    // b_arr[index].box.max().z());
    counter++;
    bvh_tree_traversal(b_arr,2*index+1,size,counter);
    bvh_tree_traversal(b_arr,2*index+2,size,counter);
    
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

void host_bvh_tree_creation_sah(aabb *list_of_bounding_boxes,
bvh_gpu_node_sah *b_arr,int size)
{
    bvh_cpu_sah parent = 
    bvh_cpu_sah(0,size,0);

    form_gpu_bvh_sah(list_of_bounding_boxes,b_arr,parent);
    

}

void read_from_obj(const char *filename,std::vector<vec3> &vertices,
std::vector<vec3> &face_values,bool three_or_not/*,float &max_x,float &max_y,
float &max_z,float &min_x,float &min_y,float &min_z*/)
{
    std::ifstream in(filename, std::ios::in);
    if (!in)
    {
        std::cerr << "Cannot open " << filename << std::endl;
        exit(1);

    }
    std::string line;
    while (std::getline(in, line))
    {
    //check v for vertices
        if (line.substr(0,2)=="v "){
            std::istringstream v(line.substr(2));
            
            float x,y,z;
            v>>x;v>>y;v>>z;

            // std::cout<<x<<" "<<y<<" "<<z<<"\n";

            // min_x = min(min_x,x);
            // min_y = min(min_y,y);
            // min_z = min(min_z,z);

            // max_x = max(max_x,x);
            // max_y = max(max_y,y);
            // max_z = max(max_z,z);

            vertices.push_back(vec3(x,y,z));
        }
        else if(line.substr(0,2)=="f "){
            int a,b,c; 
            int A,B,C;
            int _a,_b,_c;
            const char* chh=line.c_str();
            if(three_or_not)
                sscanf (chh, "f %i/%i/%i %i/%i/%i %i/%i/%i",&a,&A,&_a,&b,&B,&_b,&c,&C,&_c); 
            else
                sscanf (chh, "f %i/%i %i/%i %i/%i",&a,&A,&b,&B,&c,&C);
            a--;b--;c--;
            face_values.push_back(vec3(a,b,c));

        
        }

    }
}
//machine size 12582
void read_from_model(vec3 *a,vec3 *b,vec3 *c,std::vector<vec3> vertices,
std::vector<vec3> face_values)
{
    for(int i=0;i<face_values.size();i++)
    {
        
        a[i] = vec3(vertices[face_values[i].x()].x(),
        vertices[face_values[i].x()].y(),vertices[face_values[i].x()].z());

        b[i] = vec3(vertices[face_values[i].y()].x(),
        vertices[face_values[i].y()].y(),vertices[face_values[i].y()].z());

        c[i] = vec3(vertices[face_values[i].z()].x(),
        vertices[face_values[i].z()].y(),vertices[face_values[i].z()].z());

        //printf("\nHere %f %d %d",face_values[i].y(),vertices.size(),i);

        
    }
}


// int main(){
//     std::vector<vec3> vertices,face_values;
//     float min_x=FLT_MAX,min_y=FLT_MAX,min_z=FLT_MAX;
//     float max_x=FLT_MIN,max_y=FLT_MIN,max_z=FLT_MIN;
//     read_from_obj("./models/fighterplane.obj",vertices,face_values,max_x,
//     max_y,max_z,min_x,min_y,min_z);
//     std::cout<<max_x<<" "<<
//     max_y<<" "<<max_z<<"\n"<<min_x<<" "<<min_y<<" "<<min_z;
// }

void trace_the_circle_points(float a,float b,float c,float r
,std::vector<std::vector<float>> &circlePoints,float angle){

    // Define the number of points to use to approximate the circle
    int numPoints = 1000;

    // Loop over the values of t and u to generate the circle points
    for (int i = 0; i < numPoints; i++) {
        float t = (float)2 * (M_PI) * i / (float)numPoints;
        float u = M_PI * angle / (float)numPoints;
        float x = a + r * cos(t);
        float y = b + r * sin(t) * cos(u);
        float z = c + r * sin(t) * sin(u);
        circlePoints.push_back({x, y, z});
        
    }
}

int main(int argc ,char *argv[])
{
    if(argc>1)
    {
        std::string option_number_str = argv[1];
        int option_number = stoi(std::string(option_number_str.begin()+1,option_number_str.end()));
        if(option_number>=1&&option_number<22)
        {
            //whitted ray tracer at different resolutions and different nos
            if(argc<6)
            {
                std::cerr<<"\nInvalid arguments";
                return;
            }
            std::string no_of_objects_string = argv[2];
            std::string nx_string = argv[3];
            std::string ny_string = argv[4];
            std::string bvh_or_linear = argv[5];
            
            if(bvh_or_linear=="-bvh")
            {
                std::cerr<<"\nChose a bvh tree architecture!"<<std::endl;
            }
            else
            {
                std::cerr<<"\nChose a linear architecture!"<<std::endl;
            }

            int nx = std::stoi(std::string(nx_string.begin()+1,nx_string.end()));
            int ny = std::stoi(std::string(ny_string.begin()+1,ny_string.end()));
            int ns_x = 8;
            int ns_y = 8;
            int ns = ns_x*ns_y;
            int num_pixels = nx*ny;

            size_t anti_alias_pixel_arr_size = num_pixels*ns*sizeof(vec3);
            size_t final_pixel_arr_size  = num_pixels*sizeof(unsigned char)*3;

            //initializing the pixel arrays
            vec3 *device_anti_alias_pixel_arr;
            vec3 *device_final_pixel_arr_vec3;
            unsigned char *host_final_pixel_arr,*device_final_pixel_arr;
            
            //image texturing process
            unsigned char *host_data,*device_data;
            int width,height;
            char *filename = "./images/earthmap.jpg";

            //host memory aLLocation
            host_final_pixel_arr = (unsigned char*)malloc(final_pixel_arr_size);


            //device memory allocation
            gpuErrchk(cudaMalloc(&device_final_pixel_arr,final_pixel_arr_size));
            gpuErrchk(cudaMalloc(&device_anti_alias_pixel_arr,anti_alias_pixel_arr_size));


            if(option_number==4||option_number==8||option_number==12)
            {
                //host image texturing process
                read_from_image_stb(filename,&host_data,width,height);
                int channel_colors = 3;
                int size_of_the_image_texture_in_bytes 
                =sizeof(unsigned char)*width*height*channel_colors;

                gpuErrchk(cudaMalloc((void **)&device_data,size_of_the_image_texture_in_bytes));
                gpuErrchk(cudaMemcpy(device_data,host_data,size_of_the_image_texture_in_bytes,
                cudaMemcpyHostToDevice));
            }

            
            colliders *d_colliders;
            collider_list **d_world;
            collider_material *d_materials;
            collider_texture *d_textures;
            camera *d_camera;
            
            bvh_gpu_node *bvh_arr;
            bvh_gpu *bvh_tree;
            aabb *list_of_bounding_boxes;
            
            int total_size_bvh;
            int bvh_tree_size;
            int no_of_objects_in_the_scene = std::stoi(std::string(no_of_objects_string.begin()+1,no_of_objects_string.end()));
            int total_aabb_size = no_of_objects_in_the_scene*sizeof(aabb);
            
            vec3 lookfrom,lookat,vup;
            float fov,aperture;

            
            lookfrom = vec3(-5.0f,0.0f,-25.0f);
            lookat = vec3(0.0f,0.0f,0.0f);
            vup = vec3(0.0f,1.0f,0.0f);
            fov = 65.0f;
            aperture = 0.0f;
            
           
            std::cerr<<"Rendering "<<no_of_objects_in_the_scene<<" objects"<<std::endl;

            if(bvh_or_linear=="-bvh")
            {
                total_size_bvh  = (2*(2*no_of_objects_in_the_scene-1)+1)*sizeof(bvh_gpu_node);
                bvh_tree_size = sizeof(bvh_gpu);
                gpuErrchk(cudaMallocManaged(&bvh_arr,total_size_bvh));
                gpuErrchk(cudaMalloc(&bvh_tree,bvh_tree_size));
                
            }
            
            //device memory allocation
            gpuErrchk(cudaMallocManaged(&list_of_bounding_boxes,total_aabb_size));
            
            gpuErrchk(cudaMalloc(&d_colliders,
            no_of_objects_in_the_scene*sizeof(colliders)));

            gpuErrchk(cudaMalloc(&d_materials,
            no_of_objects_in_the_scene*sizeof(collider_material)));

            gpuErrchk(cudaMalloc(&d_textures,
            no_of_objects_in_the_scene*sizeof(collider_texture)));

            gpuErrchk(cudaMalloc(&d_world,sizeof(collider_list*)));
            
            gpuErrchk(cudaMalloc(&d_camera,sizeof(camera)));
            
            clock_t start_creation,end_creation;
            std::cerr<<"Started creation and pre processing of data "<<std::endl;
            start_creation = clock();
            
            cmd_master_renderer<<<1,1>>>(d_colliders,d_materials,d_textures,d_world
            ,d_camera,list_of_bounding_boxes,
            nx,ny,no_of_objects_in_the_scene
            ,lookfrom,lookat,aperture,vup,fov,option_number,
            device_data,width,height);
            gpuErrchk(cudaGetLastError());
            gpuErrchk(cudaDeviceSynchronize());
            
            std::cerr<<"\nEnded the creation of the world";
            if(bvh_or_linear=="-bvh")
            {
                //bvh_gpu arr creation
                host_bvh_tree_creation(list_of_bounding_boxes,
                bvh_arr,no_of_objects_in_the_scene);
                //bvh tree initialization
                initialize_bvh_tree<<<1,1>>>(bvh_arr,d_colliders,bvh_tree,
                no_of_objects_in_the_scene);
                gpuErrchk(cudaGetLastError());
                gpuErrchk(cudaDeviceSynchronize());

                std::cerr<<"\nTree creation ends";  
            }
            
            end_creation = clock();
            float timer_seconds = ((double)(end_creation-start_creation))/CLOCKS_PER_SEC;
            std::cerr << "\nPre processing and creation of the world took " << timer_seconds<<"\n";

            clock_t start,end_render,end;
            start = clock();

            dim3 blocks_render(ns_x,ns_y);
            dim3 grid_render(nx,ny);

            dim3 blocks_reduction = ns_x*ns_y;
            dim3 grid_reduction = nx*ny;

            std::cerr << "Rendering a " << nx << "x" << ny << " image \n";

            std::cerr << "Rendering started \n";
            GLFWwindow *window;
            
            if(bvh_or_linear=="-bvh")
            {
                optimized_render_tree<<<grid_render,blocks_render>>>(device_anti_alias_pixel_arr,
                nx,ny,ns,d_camera,bvh_tree,d_materials,d_textures);
                gpuErrchk(cudaGetLastError());
                gpuErrchk(cudaDeviceSynchronize());
            }
            else
            {
                optimized_render_list<<<grid_render,blocks_render>>>(device_anti_alias_pixel_arr,
                nx,ny,ns,d_camera,d_world,d_materials,d_textures);
                gpuErrchk(cudaGetLastError());
                gpuErrchk(cudaDeviceSynchronize());
            }
            
            end_render = clock();
            timer_seconds = ((double)(end_render-start))/CLOCKS_PER_SEC;
            std::cerr << "Rendering complete took " << timer_seconds<<"\n";
            
            //starting our reduction kernel to find our final pixel array
            shared_reduction_interleaved_approach_complete_unrolling_2d<<<grid_reduction,blocks_reduction>>>(device_anti_alias_pixel_arr,
            device_final_pixel_arr,num_pixels*ns);
            gpuErrchk(cudaGetLastError());
            gpuErrchk(cudaDeviceSynchronize());
            
            gpuErrchk(cudaMemcpy(host_final_pixel_arr,device_final_pixel_arr,final_pixel_arr_size,
            cudaMemcpyDeviceToHost));
            
            end = clock();
            timer_seconds = ((double)(end-start))/CLOCKS_PER_SEC;
            std::cout<<"Reduction complete took "<<timer_seconds<<" seconds in the whole process "<<
            argv[1]<<" "<<argv[2]<<" "<<argv[3]<<" "<<argv[4]<<" "<<argv[5]<<std::endl;

            show_windows(host_final_pixel_arr,nx,ny,window);

        }
        else if(option_number>=22&&option_number<27)
        {
            //whitted ray tracer at different resolutions and different nos
            if(argc<7)
            {
                std::cerr<<"\nInvalid arguments";
                return;
            }
            std::string no_of_objects_string = argv[2];
            std::string nx_string = argv[3];
            std::string ny_string = argv[4];
            std::string normal_bvh_or_bvh_sah = argv[5];
            std::string model_name = argv[6];
            model_name = std::string(model_name.begin()+1,model_name.end());
            
            const int length = model_name.length();
            char* model_name_char = new char[length + 1];
            strcpy(model_name_char, model_name.c_str());
            
            if(normal_bvh_or_bvh_sah=="-bvh")
            {
                std::cerr<<"\nChose a bvh tree architecture!"<<std::endl;
            }
            else
            {
                std::cerr<<"\nChose a bvh_sah tree architecture!"<<std::endl;
            }

            

            int nx = std::stoi(std::string(nx_string.begin()+1,nx_string.end()));
            int ny = std::stoi(std::string(ny_string.begin()+1,ny_string.end()));
            int ns_x = 8;
            int ns_y = 8;
            int ns = ns_x*ns_y;
            int num_pixels = nx*ny;

            size_t anti_alias_pixel_arr_size = num_pixels*ns*sizeof(vec3);
            size_t final_pixel_arr_size  = num_pixels*sizeof(unsigned char)*3;

            //initializing the pixel arrays
            vec3 *device_anti_alias_pixel_arr;
            vec3 *device_final_pixel_arr_vec3;
            unsigned char *host_final_pixel_arr,*device_final_pixel_arr;

            //host memory aLLocation
            host_final_pixel_arr = (unsigned char*)malloc(final_pixel_arr_size);


            //device memory allocation
            gpuErrchk(cudaMalloc(&device_final_pixel_arr,final_pixel_arr_size));
            gpuErrchk(cudaMalloc(&device_anti_alias_pixel_arr,anti_alias_pixel_arr_size));

            triangle *d_list;
            camera *d_camera;
            
            bvh_gpu_node *bvh_arr;
            bvh_gpu_streamlined_tree *bvh_tree;
            bvh_gpu_node_sah *bvh_arr_sah;
            bvh_gpu_streamlined_tree_sah *bvh_tree_sah;
            vec3 *a,*b,*c;

            aabb *list_of_bounding_boxes;
            
            int total_size_bvh;
            int bvh_tree_size;
            int no_of_objects_in_the_scene;
            int total_aabb_size;

            if(model_name=="random")
                no_of_objects_in_the_scene = std::stoi(std::string(no_of_objects_string.begin()+1,no_of_objects_string.end()));
            else
            {
                std::vector<vec3> vertices,face_values;
                bool three_or_not = false;
                if(model_name=="./models/satellite.obj")
                    three_or_not = true;
                read_from_obj(model_name_char,vertices,face_values,three_or_not);
                no_of_objects_in_the_scene = face_values.size();
                gpuErrchk(cudaMallocManaged(&a,sizeof(vec3)*no_of_objects_in_the_scene));
                gpuErrchk(cudaMallocManaged(&b,sizeof(vec3)*no_of_objects_in_the_scene));
                gpuErrchk(cudaMallocManaged(&c,sizeof(vec3)*no_of_objects_in_the_scene));
                read_from_model(a,b,c,vertices,face_values);
            }

            total_aabb_size = no_of_objects_in_the_scene*sizeof(aabb);
            
            vec3 lookfrom,lookat,vup;
            float fov,aperture;

            if(option_number==22)
            {
                lookfrom = vec3(-5.0f,0.0f,-25.0f);
                lookat = vec3(0.0f,0.0f,0.0f);
                vup = vec3(0.0f,1.0f,0.0f);
                fov = 65.0f;
                aperture = 0.0f;
            }
            else if(option_number==24)
            {
                lookfrom = vec3(0.0f, 0.0f, -1.0f);
                lookat = vec3(0,0,0);
                aperture = 0.0;
                vup = vec3(0.0f,1.0f,0.0f);
                fov = 35.0f;
            }
            else if(option_number==25)
            {
                lookfrom = vec3(8.0f, 3.0f, -20.0f);
                lookat = vec3(0,0,0);
                aperture = 0.0;
                vup = vec3(0.0f,1.0f,0.0f);
                fov = 35.0f;
            }
            else if(option_number==26)
            {
                lookfrom = vec3(-5.0f, 3.0f, -1.0f);
                lookat = vec3(0,0,0);
                aperture = 0.0;
                vup = vec3(0.0f,1.0f,0.0f);
                fov = 20.0f;
            }
            
            std::cerr<<"Rendering "<<no_of_objects_in_the_scene<<" objects"<<std::endl;

            if(normal_bvh_or_bvh_sah=="-bvh")
            {
                total_size_bvh  = (2*(2*no_of_objects_in_the_scene-1)+1)*sizeof(bvh_gpu_node);
                bvh_tree_size = sizeof(bvh_gpu_streamlined_tree);
                gpuErrchk(cudaMallocManaged(&bvh_arr,total_size_bvh));
                gpuErrchk(cudaMalloc(&bvh_tree,bvh_tree_size));
                
            }
            else
            {
                total_size_bvh  = (2*no_of_objects_in_the_scene-1)*sizeof(bvh_gpu_node_sah);
                bvh_tree_size = sizeof(bvh_gpu_streamlined_tree_sah);
                gpuErrchk(cudaMallocManaged(&bvh_arr_sah,total_size_bvh));
                gpuErrchk(cudaMalloc(&bvh_tree_sah,bvh_tree_size));
            }
            
            //device memory allocation
            gpuErrchk(cudaMallocManaged(&list_of_bounding_boxes,total_aabb_size));
            gpuErrchk(cudaMalloc(&d_list,sizeof(triangle)*no_of_objects_in_the_scene))
            gpuErrchk(cudaMalloc(&d_camera,sizeof(camera)));

            clock_t start_creation,end_creation;
            std::cerr<<"Started creation and pre processing of data "<<std::endl;
            start_creation = clock();
            
            cmd_master_renderer_streamlined<<<1,1>>>(d_list,a,b,c
            ,d_camera,list_of_bounding_boxes,
            nx,ny,no_of_objects_in_the_scene
            ,lookfrom,lookat,aperture,vup,fov,model_name=="random"?true:false);
            gpuErrchk(cudaGetLastError());
            gpuErrchk(cudaDeviceSynchronize());
            
            if(normal_bvh_or_bvh_sah=="-bvh")
            {
                //bvh_gpu arr creation
                host_bvh_tree_creation(list_of_bounding_boxes,
                bvh_arr,no_of_objects_in_the_scene);
                
                //bvh tree initialization
                initialize_bvh_tree_streamlined<<<1,1>>>(bvh_arr,d_list,bvh_tree
                ,2*(2*no_of_objects_in_the_scene-1));
                gpuErrchk(cudaGetLastError());
                gpuErrchk(cudaDeviceSynchronize());

                std::cerr<<"\nTree creation ends";
            }
            else
            {
                //bvh_gpu arr creation
                host_bvh_tree_creation_sah(list_of_bounding_boxes,
                bvh_arr_sah,no_of_objects_in_the_scene);

                //bvh tree initialization
                initialize_bvh_tree_streamlined_sah<<<1,1>>>(bvh_arr_sah,d_list,bvh_tree_sah
                ,(2*no_of_objects_in_the_scene-1));
                gpuErrchk(cudaGetLastError());
                gpuErrchk(cudaDeviceSynchronize());

                std::cerr<<"\nTree creation ends";
            }
            std::cerr<<"\nEnded the creation of the world";

            end_creation = clock();
            float timer_seconds = ((double)(end_creation-start_creation))/CLOCKS_PER_SEC;
            std::cerr << "\nPre processing and creation of the world took " << timer_seconds<<"\n";

            clock_t start,end_render,end;
            start = clock();

            dim3 blocks_render(ns_x,ns_y);
            dim3 grid_render(nx,ny);

            dim3 blocks_reduction = ns_x*ns_y;
            dim3 grid_reduction = nx*ny;

            std::cerr << "Rendering a " << nx << "x" << ny << " image \n";

            std::cerr << "Rendering started \n";
            GLFWwindow *window;
            
            if(normal_bvh_or_bvh_sah=="-bvh")
            {
                optimized_render_streamlined_tree_sampling<<<grid_render,blocks_render>>>(device_anti_alias_pixel_arr,
                nx,ny,d_camera,bvh_tree);
                gpuErrchk(cudaGetLastError());
                gpuErrchk(cudaDeviceSynchronize());
            }
            else
            {
                optimized_render_streamlined_tree_sampling_sah<<<grid_render,blocks_render>>>(device_anti_alias_pixel_arr,
                nx,ny,d_camera,bvh_tree_sah);
                gpuErrchk(cudaGetLastError());
                gpuErrchk(cudaDeviceSynchronize());
            }
            
            end_render = clock();
            timer_seconds = ((double)(end_render-start))/CLOCKS_PER_SEC;
            std::cerr << "Rendering complete took " << timer_seconds<<"\n";
            
            //starting our reduction kernel to find our final pixel array
            shared_reduction_interleaved_approach_complete_unrolling_2d<<<grid_reduction,blocks_reduction>>>(device_anti_alias_pixel_arr,
            device_final_pixel_arr,num_pixels*ns);
            gpuErrchk(cudaGetLastError());
            gpuErrchk(cudaDeviceSynchronize());
            
            gpuErrchk(cudaMemcpy(host_final_pixel_arr,device_final_pixel_arr,final_pixel_arr_size,
            cudaMemcpyDeviceToHost));
            
            end = clock();
            timer_seconds = ((double)(end-start))/CLOCKS_PER_SEC;
            std::cout<<"Reduction complete took "<<timer_seconds<<" seconds in the whole process "<<
            argv[1]<<" "<<argv[2]<<" "<<argv[3]<<" "<<argv[4]<<" "<<argv[5]<<std::endl;

            show_windows(host_final_pixel_arr,nx,ny,window);
        }
        else if(option_number==31)
        {
            
            if(argc<8)
            {
                std::cerr<<"\nInvalid arguments";
                return;
            }
            std::string no_of_objects_string = argv[2];
            std::string nx_string = argv[3];
            std::string ny_string = argv[4];
            std::string interleaved_or_not = argv[5];
            std::string depth_string = argv[6];
            std::string sample_1D_string = argv[7];


            int sample_1D = std::stoi(std::string(sample_1D_string.begin()+1,sample_1D_string.end()));
            int depth = std::stoi(std::string(depth_string.begin()+1,depth_string.end()));

            if(interleaved_or_not=="-interleaved")
            {
                printf("Using the interleaved kernel with %d depth\n",depth);
            }
                
            
            int nx = std::stoi(std::string(nx_string.begin()+1,nx_string.end()));
            int ny = std::stoi(std::string(ny_string.begin()+1,ny_string.end()));
            int ns_x = sample_1D;
            int ns_y = sample_1D;
            int ns = ns_x*ns_y;
            int num_pixels = nx*ny;

            size_t anti_alias_pixel_arr_size = num_pixels*ns*sizeof(vec3);
            size_t final_pixel_arr_size  = num_pixels*sizeof(unsigned char)*3;

            //initializing the pixel arrays
            vec3 *device_anti_alias_pixel_arr;
            vec3 *device_final_pixel_arr_vec3;
            unsigned char *host_final_pixel_arr,*device_final_pixel_arr;

            //host memory aLLocation
            host_final_pixel_arr = (unsigned char*)malloc(final_pixel_arr_size);


            //device memory allocation
            gpuErrchk(cudaMalloc(&device_final_pixel_arr,final_pixel_arr_size));
            gpuErrchk(cudaMalloc(&device_anti_alias_pixel_arr,anti_alias_pixel_arr_size));
            gpuErrchk(cudaMalloc(&device_final_pixel_arr_vec3,sizeof(vec3)*(nx*ny)));

            triangle *d_list;
            camera *d_camera;
            
            bvh_gpu_node *bvh_arr;
            bvh_gpu_streamlined_tree *bvh_tree;
            bvh_gpu_node_sah *bvh_arr_sah;
            bvh_gpu_streamlined_tree_sah *bvh_tree_sah;
            vec3 *a,*b,*c;

            aabb *list_of_bounding_boxes;
            
            int total_size_bvh;
            int bvh_tree_size;
            int no_of_objects_in_the_scene;
            int total_aabb_size;

            
            no_of_objects_in_the_scene = std::stoi(std::string(no_of_objects_string.begin()+1,no_of_objects_string.end()));
            

            total_aabb_size = no_of_objects_in_the_scene*sizeof(aabb);
            
            vec3 lookfrom,lookat,vup;
            float fov,aperture;

            if(option_number==31)
            {
                lookfrom = vec3(-5.0f,0.0f,-100.0f);
                lookat = vec3(0.0f,0.0f,0.0f);
                vup = vec3(0.0f,1.0f,0.0f);
                fov = 60.0f;
                aperture = 0.0f;
            }
            
            std::cerr<<"Rendering "<<no_of_objects_in_the_scene<<" objects"<<std::endl;

            
            total_size_bvh  = (2*(2*no_of_objects_in_the_scene-1)+1)*sizeof(bvh_gpu_node);
            bvh_tree_size = sizeof(bvh_gpu_streamlined_tree);
            gpuErrchk(cudaMallocManaged(&bvh_arr,total_size_bvh));
            gpuErrchk(cudaMalloc(&bvh_tree,bvh_tree_size));
            
            
                
            //device memory allocation
            gpuErrchk(cudaMallocManaged(&list_of_bounding_boxes,total_aabb_size));
            gpuErrchk(cudaMalloc(&d_list,sizeof(triangle)*no_of_objects_in_the_scene))
            gpuErrchk(cudaMalloc(&d_camera,sizeof(camera)));

            clock_t start_creation,end_creation;
            std::cerr<<"Started creation and pre processing of data "<<std::endl;
            start_creation = clock();
            
            cmd_master_renderer_streamlined<<<1,1>>>(d_list,a,b,c
            ,d_camera,list_of_bounding_boxes,
            nx,ny,no_of_objects_in_the_scene
            ,lookfrom,lookat,aperture,vup,fov,true);
            gpuErrchk(cudaGetLastError());
            gpuErrchk(cudaDeviceSynchronize());
            
            
            //bvh_gpu arr creation
            host_bvh_tree_creation(list_of_bounding_boxes,
            bvh_arr,no_of_objects_in_the_scene);
            
            //bvh tree initialization
            initialize_bvh_tree_streamlined<<<1,1>>>(bvh_arr,d_list,bvh_tree
            ,2*(2*no_of_objects_in_the_scene-1));
            gpuErrchk(cudaGetLastError());
            gpuErrchk(cudaDeviceSynchronize());

            std::cerr<<"\nTree creation ends";
        
            std::cerr<<"\nEnded the creation of the world";

            end_creation = clock();
            float timer_seconds = ((double)(end_creation-start_creation))/CLOCKS_PER_SEC;
            std::cerr << "\nPre processing and creation of the world took " << timer_seconds<<"\n";

            clock_t start,end_render,end;
            start = clock();

            dim3 blocks_render(ns_x,ns_y);
            dim3 grid_render(nx,ny);

            dim3 blocks_render_linear(8,8);
            dim3 grid_render_linear(nx/8,ny/8);

            dim3 blocks_reduction = ns_x*ns_y;
            dim3 grid_reduction = nx*ny;

            std::cerr << "Rendering a " << nx << "x" << ny << " image \n";

            std::cerr << "Rendering started \n";
            GLFWwindow *window;
            
            if(interleaved_or_not=="-interleaved")
            {
                optimized_render_streamlined_tree_large_loads_sampling<<<grid_render,blocks_render>>>(device_anti_alias_pixel_arr,
                nx,ny,d_camera,bvh_tree,depth);
                gpuErrchk(cudaGetLastError());
                gpuErrchk(cudaDeviceSynchronize());
            }
            else
            {

                optimized_render_streamlined_tree_large_loads<<<grid_render_linear,blocks_render_linear>>>(device_final_pixel_arr_vec3,
                nx,ny,d_camera,bvh_tree);
                gpuErrchk(cudaGetLastError());
                gpuErrchk(cudaDeviceSynchronize());
            }
            
            
            
            end_render = clock();
            timer_seconds = ((double)(end_render-start))/CLOCKS_PER_SEC;
            std::cerr << "Rendering complete took " << timer_seconds<<"\n";
            if(interleaved_or_not=="-interleaved")
            {
                //starting our reduction kernel to find our final pixel array
                shared_reduction_interleaved_approach_complete_unrolling_2d<<<grid_reduction,blocks_reduction>>>(device_anti_alias_pixel_arr,
                device_final_pixel_arr,num_pixels*ns);
                gpuErrchk(cudaGetLastError());
                gpuErrchk(cudaDeviceSynchronize());
                
                gpuErrchk(cudaMemcpy(host_final_pixel_arr,device_final_pixel_arr,final_pixel_arr_size,
                cudaMemcpyDeviceToHost));
            }
            else
            {
                host_assign_kernel<<<(num_pixels)/64,64>>>(device_final_pixel_arr_vec3,
                device_final_pixel_arr);
                gpuErrchk(cudaGetLastError());
                gpuErrchk(cudaDeviceSynchronize());
                gpuErrchk(cudaMemcpy(host_final_pixel_arr,device_final_pixel_arr,final_pixel_arr_size,
                cudaMemcpyDeviceToHost));
            }
            
            end = clock();
            timer_seconds = ((double)(end-start))/CLOCKS_PER_SEC;
            std::cout<<"Reduction complete took "<<timer_seconds<<" seconds in the whole process "<<
            argv[1]<<" "<<argv[2]<<" "<<argv[3]<<" "<<argv[4]<<std::endl;

            show_windows(host_final_pixel_arr,nx,ny,window);
        }
        else if(option_number==27)
        {
            //whitted ray tracer at different resolutions and different nos
            if(argc<8)
            {
                std::cerr<<"\nInvalid arguments";
                return;
            }
            std::string no_of_objects_string = argv[2];
            std::string nx_string = argv[3];
            std::string ny_string = argv[4];
            std::string model_name = argv[5];
            std::string depth_or_not = argv[6];
            std::string ns_string = argv[7];
            model_name = std::string(model_name.begin()+1,model_name.end());
            
            const int length = model_name.length();
            char* model_name_char = new char[length + 1];
            strcpy(model_name_char, model_name.c_str());
        

            int nx = std::stoi(std::string(nx_string.begin()+1,nx_string.end()));
            int ny = std::stoi(std::string(ny_string.begin()+1,ny_string.end()));
            int num_pixels = nx*ny;
            int ns = std::stoi(std::string(ns_string.begin()+1,ns_string.end()));
            std::cerr<<"\n Samples per pixel "<<ns<<std::endl;

            size_t anti_alias_pixel_arr_size = num_pixels*ns*sizeof(vec3);
            size_t final_pixel_arr_size_vec3 = num_pixels*sizeof(vec3);
            size_t final_pixel_arr_size  = num_pixels*sizeof(unsigned char)*3;

            //initializing the pixel arrays
            vec3 *device_final_pixel_arr_vec3,*device_anti_alias_pixel_arr;
            unsigned char *host_final_pixel_arr,*device_final_pixel_arr;
            float *max_distance;


            //host memory aLLocation
            host_final_pixel_arr = (unsigned char*)malloc(final_pixel_arr_size);


            //device memory allocation
            gpuErrchk(cudaMalloc(&device_final_pixel_arr,final_pixel_arr_size));
            gpuErrchk(cudaMalloc(&device_final_pixel_arr_vec3,final_pixel_arr_size_vec3));
            gpuErrchk(cudaMalloc(&device_anti_alias_pixel_arr,anti_alias_pixel_arr_size));
            gpuErrchk(cudaMallocManaged(&max_distance,sizeof(float)));

            triangle *d_list;
            camera *d_camera;
            
            bvh_gpu_node_sah *bvh_arr_sah;
            bvh_gpu_streamlined_tree_sah *bvh_tree_sah;
            vec3 *a,*b,*c;

            aabb *list_of_bounding_boxes;
            
            int total_size_bvh;
            int bvh_tree_size;
            int no_of_objects_in_the_scene;
            int total_aabb_size;
            std::vector<std::vector<float>> circlePoints;

            if(model_name=="random")
                no_of_objects_in_the_scene = std::stoi(std::string(no_of_objects_string.begin()+1,no_of_objects_string.end()));
            else
            {
                std::vector<vec3> vertices,face_values;
                bool three_or_not = false;
                if(model_name=="./models/satellite.obj")
                    three_or_not = true;
                read_from_obj(model_name_char,vertices,face_values,three_or_not);
                no_of_objects_in_the_scene = face_values.size();
                gpuErrchk(cudaMallocManaged(&a,sizeof(vec3)*no_of_objects_in_the_scene));
                gpuErrchk(cudaMallocManaged(&b,sizeof(vec3)*no_of_objects_in_the_scene));
                gpuErrchk(cudaMallocManaged(&c,sizeof(vec3)*no_of_objects_in_the_scene));
                read_from_model(a,b,c,vertices,face_values);
            }
            if(depth_or_not=="-metal"&&model_name=="./models/satellite.obj")
                no_of_objects_in_the_scene+=50;
            total_aabb_size = no_of_objects_in_the_scene*sizeof(aabb);
            
            vec3 lookfrom,lookat,vup;
            float fov,aperture;

            
            if(option_number==27&&depth_or_not=="-depth"&&model_name!="random")
            {
                lookfrom = vec3(-5.0f, 0.0f, -1.0f);
                lookat = vec3(0,0,0);
                aperture = 0.0;
                vup = vec3(0.0f,1.0f,0.0f);
                fov = 18.0f;
            }
            else if(option_number==27&&depth_or_not=="-recursive"&&model_name!="random")
            {
                lookfrom = vec3(-5.0f, 0.0f, -1.0f);
                lookat = vec3(0,0,0);
                aperture = 0.0;
                vup = vec3(0.0f,1.0f,0.0f);
                fov = 20.0f;
            }
            else if(option_number==27&&depth_or_not=="-metal"&&model_name!="random")
            {
                lookfrom = vec3(-5.0f, 0.0f, -1.0f);
                lookat = vec3(0,0,0);
                aperture = 0.0;
                vup = vec3(0.0f,1.0f,0.0f);
                fov = 20.0f;
            }
            else
            {
                lookfrom = vec3(-5.0f, 0.0f, -25.0f);
                lookat = vec3(0,0,0);
                aperture = 0.0;
                vup = vec3(0.0f,1.0f,0.0f);
                fov = 60.0f;
            }
            
            std::cerr<<"Rendering "<<no_of_objects_in_the_scene<<" objects"<<std::endl;

            
            total_size_bvh  = (2*no_of_objects_in_the_scene-1)*sizeof(bvh_gpu_node_sah);
            bvh_tree_size = sizeof(bvh_gpu_streamlined_tree_sah);
            gpuErrchk(cudaMallocManaged(&bvh_arr_sah,total_size_bvh));
            gpuErrchk(cudaMalloc(&bvh_tree_sah,bvh_tree_size));
                
           
            //device memory allocation
            gpuErrchk(cudaMallocManaged(&list_of_bounding_boxes,total_aabb_size));
            gpuErrchk(cudaMalloc(&d_list,sizeof(triangle)*no_of_objects_in_the_scene))
            gpuErrchk(cudaMallocManaged(&d_camera,sizeof(camera)));

            clock_t start_creation,end_creation;
            std::cerr<<"Started creation and pre processing of data "<<std::endl;
            start_creation = clock();
            
            cmd_master_renderer_streamlined<<<1,1>>>(d_list,a,b,c
            ,d_camera,list_of_bounding_boxes,
            nx,ny,no_of_objects_in_the_scene
            ,lookfrom,lookat,aperture,vup,fov,model_name=="random"?true:false,
            depth_or_not=="-metal"?true:false);
            gpuErrchk(cudaGetLastError());
            gpuErrchk(cudaDeviceSynchronize());
            
            //bvh_gpu arr creation
            host_bvh_tree_creation_sah(list_of_bounding_boxes,
            bvh_arr_sah,no_of_objects_in_the_scene);
            
            //bvh tree initialization
            initialize_bvh_tree_streamlined_sah<<<1,1>>>(bvh_arr_sah,d_list,bvh_tree_sah
            ,2*(2*no_of_objects_in_the_scene-1));
            gpuErrchk(cudaGetLastError());
            gpuErrchk(cudaDeviceSynchronize());

            std::cerr<<"\nTree creation ends";
            
            std::cerr<<"\nEnded the creation of the world";

            end_creation = clock();
            float timer_seconds = ((double)(end_creation-start_creation))/CLOCKS_PER_SEC;
            std::cerr << "\nPre processing and creation of the world took " << timer_seconds<<"\n";

            clock_t start,end_render,end;
            

            dim3 blocks_render(8,8);
            dim3 grid_render(nx/8,ny/8);

            dim3 blocks_render_sampling(ns,8);
            dim3 grid_render_sampling(nx/8,ny);

            dim3 blocks_reduction(ns);
            dim3 grid_reduction(num_pixels);

            trace_the_circle_points(0,0,0,sqrt((lookfrom-lookat).squared_length()),circlePoints,25);
            std::cerr << "Rendering a " << nx << "x" << ny << " image \n";
            GLFWwindow *window;
            bool first_time = true;
            for(int i=0;i<circlePoints.size();i++)
            {
                start = clock();
                lookfrom = vec3(circlePoints[i][0],circlePoints[i][1],
                circlePoints[i][2]);
                d_camera->recalculate_look_from(lookfrom,lookat,vup,fov,float(nx)/float(ny),
                (lookfrom-lookat).length());
                *max_distance = FLT_MIN;
                if(depth_or_not=="-depth")
                {
                    optimized_render_streamlined_tree_sah<<<grid_render,blocks_render>>>(device_final_pixel_arr_vec3,
                    nx,ny,d_camera,bvh_tree_sah,max_distance,ns);
                    gpuErrchk(cudaGetLastError());
                    gpuErrchk(cudaDeviceSynchronize());
                }
                else if(depth_or_not=="-recursive")
                {
                    optimized_render_streamlined_tree_recursive_sah<<<grid_render_sampling,blocks_render_sampling>>>(device_anti_alias_pixel_arr,
                    nx,ny,d_camera,bvh_tree_sah);
                    gpuErrchk(cudaGetLastError());
                    gpuErrchk(cudaDeviceSynchronize());
                }
                else
                {
                    optimized_render_streamlined_tree_recursive_sah_metal<<<grid_render_sampling,blocks_render_sampling>>>(device_anti_alias_pixel_arr,
                    nx,ny,d_camera,bvh_tree_sah);
                    gpuErrchk(cudaGetLastError());
                    gpuErrchk(cudaDeviceSynchronize());
                }
                
                if(depth_or_not=="-depth")
                {
                    depth_visualization_kernel<<<(num_pixels)/64,64>>>(device_final_pixel_arr_vec3,
                    device_final_pixel_arr,*max_distance);
                    gpuErrchk(cudaGetLastError());
                    gpuErrchk(cudaDeviceSynchronize());
                }
                else
                {
                    shared_reduction_interleaved_approach_complete_unrolling_small<<<grid_reduction,blocks_reduction>>>(device_anti_alias_pixel_arr,
                    device_final_pixel_arr,num_pixels*ns);
                    gpuErrchk(cudaGetLastError());
                    gpuErrchk(cudaDeviceSynchronize());
                    
                    gpuErrchk(cudaMemcpy(host_final_pixel_arr,device_final_pixel_arr,final_pixel_arr_size,
                    cudaMemcpyDeviceToHost));
                }
                gpuErrchk(cudaMemcpy(host_final_pixel_arr,device_final_pixel_arr,final_pixel_arr_size,
                cudaMemcpyDeviceToHost));
                end = clock();
                timer_seconds = ((double)(end-start))/CLOCKS_PER_SEC;
                std::cerr << "\nRendering took " << timer_seconds<<"\n";
                show_windows_animated(host_final_pixel_arr,nx,ny,first_time,window);
            }

            
        }
        else if(option_number==29)
        {
            //whitted ray tracer at different resolutions and different nos
            if(argc<6)
            {
                std::cerr<<"\nInvalid arguments";
                return;
            }
            std::string no_of_objects_string = argv[2];
            std::string nx_string = argv[3];
            std::string ny_string = argv[4];
            std::string bvh_or_linear = argv[5];
            if(bvh_or_linear=="-bvh")
            {
                std::cerr<<"\nChose a bvh tree architecture!"<<std::endl;
            }
            else
            {
                std::cerr<<"\nChose a linear architecture!"<<std::endl;
            }

            int nx = std::stoi(std::string(nx_string.begin()+1,nx_string.end()));
            int ny = std::stoi(std::string(ny_string.begin()+1,ny_string.end()));
            int ns_x = 16;
            int ns_y = 32;
            int ns = ns_x*ns_y;
            int num_pixels = nx*ny;

            size_t anti_alias_pixel_arr_size = num_pixels*ns*sizeof(vec3);
            size_t final_pixel_arr_size  = num_pixels*sizeof(unsigned char)*3;

            //initializing the pixel arrays
            vec3 *device_anti_alias_pixel_arr;
            vec3 *device_final_pixel_arr_vec3;
            unsigned char *host_final_pixel_arr,*device_final_pixel_arr;
            
            //image texturing process
            unsigned char *host_data,*device_data;
            int width,height;
            char *filename = "./images/pattern.jpg";

            //host memory aLLocation
            host_final_pixel_arr = (unsigned char*)malloc(final_pixel_arr_size);


            //device memory allocation
            gpuErrchk(cudaMalloc(&device_final_pixel_arr,final_pixel_arr_size));
            gpuErrchk(cudaMalloc(&device_anti_alias_pixel_arr,anti_alias_pixel_arr_size));


            if(option_number==29)
            {
                //host image texturing process
                read_from_image_stb(filename,&host_data,width,height);
                int channel_colors = 3;
                int size_of_the_image_texture_in_bytes 
                =sizeof(unsigned char)*width*height*channel_colors;

                gpuErrchk(cudaMalloc((void **)&device_data,size_of_the_image_texture_in_bytes));
                gpuErrchk(cudaMemcpy(device_data,host_data,size_of_the_image_texture_in_bytes,
                cudaMemcpyHostToDevice));
            }

            
            colliders *d_colliders;
            collider_list **d_world;
            collider_material *d_materials;
            collider_texture *d_textures;
            camera *d_camera;
            
            bvh_gpu_node *bvh_arr;
            bvh_gpu *bvh_tree;
            aabb *list_of_bounding_boxes;
            
            int total_size_bvh;
            int bvh_tree_size;
            int no_of_objects_in_the_scene = std::stoi(std::string(no_of_objects_string.begin()+1,no_of_objects_string.end()));
            int total_aabb_size = no_of_objects_in_the_scene*sizeof(aabb);
            
            vec3 lookfrom,lookat,vup;
            float fov,aperture;

            
            lookfrom = vec3(-5.0f,0.0f,-25.0f);
            lookat = vec3(0.0f,0.0f,0.0f);
            vup = vec3(0.0f,1.0f,0.0f);
            fov = 65.0f;
            aperture = 0.0f;
            
           
            std::cerr<<"Rendering "<<no_of_objects_in_the_scene<<" objects"<<std::endl;

            if(bvh_or_linear=="-bvh")
            {
                total_size_bvh  = (2*(2*no_of_objects_in_the_scene-1)+1)*sizeof(bvh_gpu_node);
                bvh_tree_size = sizeof(bvh_gpu);
                gpuErrchk(cudaMallocManaged(&bvh_arr,total_size_bvh));
                gpuErrchk(cudaMalloc(&bvh_tree,bvh_tree_size));
                
            }
            
            //device memory allocation
            gpuErrchk(cudaMallocManaged(&list_of_bounding_boxes,total_aabb_size));
            
            gpuErrchk(cudaMalloc(&d_colliders,
            no_of_objects_in_the_scene*sizeof(colliders)));

            gpuErrchk(cudaMalloc(&d_materials,
            no_of_objects_in_the_scene*sizeof(collider_material)));

            gpuErrchk(cudaMalloc(&d_textures,
            no_of_objects_in_the_scene*sizeof(collider_texture)));

            gpuErrchk(cudaMalloc(&d_world,sizeof(collider_list*)));
            
            gpuErrchk(cudaMalloc(&d_camera,sizeof(camera)));
            
            clock_t start_creation,end_creation;
            std::cerr<<"Started creation and pre processing of data "<<std::endl;
            start_creation = clock();
            
            cmd_master_renderer<<<1,1>>>(d_colliders,d_materials,d_textures,d_world
            ,d_camera,list_of_bounding_boxes,
            nx,ny,no_of_objects_in_the_scene
            ,lookfrom,lookat,aperture,vup,fov,option_number,
            device_data,width,height);
            gpuErrchk(cudaGetLastError());
            gpuErrchk(cudaDeviceSynchronize());
            
            std::cerr<<"\nEnded the creation of the world";
            if(bvh_or_linear=="-bvh")
            {
                //bvh_gpu arr creation
                host_bvh_tree_creation(list_of_bounding_boxes,
                bvh_arr,no_of_objects_in_the_scene);
                //bvh tree initialization
                initialize_bvh_tree<<<1,1>>>(bvh_arr,d_colliders,bvh_tree,
                no_of_objects_in_the_scene);
                gpuErrchk(cudaGetLastError());
                gpuErrchk(cudaDeviceSynchronize());

                std::cerr<<"\nTree creation ends";  
            }
            
            end_creation = clock();
            float timer_seconds = ((double)(end_creation-start_creation))/CLOCKS_PER_SEC;
            std::cerr << "\nPre processing and creation of the world took " << timer_seconds<<"\n";

            clock_t start,end_render,end;
            start = clock();

            dim3 blocks_render(ns_x,ns_y);
            dim3 grid_render(nx,ny);

            dim3 blocks_reduction = ns_x*ns_y;
            dim3 grid_reduction = nx*ny;

            std::cerr << "Rendering a " << nx << "x" << ny << " image \n";

            std::cerr << "Rendering started \n";
            GLFWwindow *window;
            
            if(bvh_or_linear=="-bvh")
            {
                optimized_render_tree<<<grid_render,blocks_render>>>(device_anti_alias_pixel_arr,
                nx,ny,ns,d_camera,bvh_tree,d_materials,d_textures);
                gpuErrchk(cudaGetLastError());
                gpuErrchk(cudaDeviceSynchronize());
            }
            else
            {
                optimized_render_list<<<grid_render,blocks_render>>>(device_anti_alias_pixel_arr,
                nx,ny,ns,d_camera,d_world,d_materials,d_textures);
                gpuErrchk(cudaGetLastError());
                gpuErrchk(cudaDeviceSynchronize());
            }
            
            end_render = clock();
            timer_seconds = ((double)(end_render-start))/CLOCKS_PER_SEC;
            std::cerr << "Rendering complete took " << timer_seconds<<"\n";
            
            //starting our reduction kernel to find our final pixel array
            shared_reduction_interleaved_approach_complete_unrolling_2d<<<grid_reduction,blocks_reduction>>>(device_anti_alias_pixel_arr,
            device_final_pixel_arr,num_pixels*ns);
            gpuErrchk(cudaGetLastError());
            gpuErrchk(cudaDeviceSynchronize());
            
            gpuErrchk(cudaMemcpy(host_final_pixel_arr,device_final_pixel_arr,final_pixel_arr_size,
            cudaMemcpyDeviceToHost));
            
            end = clock();
            timer_seconds = ((double)(end-start))/CLOCKS_PER_SEC;
            std::cerr<<"Reduction complete took "<<timer_seconds<<" seconds in the whole process\n";

            show_windows(host_final_pixel_arr,nx,ny,window);
        }
    }
    else
    {
        std::cerr<<"Invalid inputs!!";
    }
    
    return 0;
}

/*int main()
{
    
    int nx = 736;
    int ny = 640;
    int ns_x = 8;
    int ns_y = 8;
    int ns = ns_x*ns_y;
    int num_pixels = nx*ny;
    
    size_t limit = 1024*4;
    cudaDeviceSetLimit(cudaLimitStackSize,limit);
    //cudaSetLimit(cudaLimitStackSize, 32768ULL);
    size_t anti_alias_pixel_arr_size = num_pixels*ns*sizeof(vec3);
    size_t final_pixel_arr_size  = num_pixels*sizeof(unsigned char)*3;

    //initializing the pixel arrays
    vec3 *device_anti_alias_pixel_arr;
    vec3 *device_final_pixel_arr_vec3;
    unsigned char *host_final_pixel_arr,*device_final_pixel_arr;

    //host memory aLLocation
    host_final_pixel_arr = (unsigned char*)malloc(final_pixel_arr_size);
    
    //host image texturing process
    unsigned char *host_data,*device_data;
    int width,height;
    char *filename = "./images/earthmap.jpg";
    read_from_image_stb(filename,&host_data,width,height);
    int channel_colors = 3;
    int size_of_the_image_texture_in_bytes 
    =sizeof(unsigned char)*width*height*channel_colors;

    gpuErrchk(cudaMalloc((void **)&device_data,size_of_the_image_texture_in_bytes));
    gpuErrchk(cudaMemcpy(device_data,host_data,size_of_the_image_texture_in_bytes,
    cudaMemcpyHostToDevice));


    //device memory allocation
    gpuErrchk(cudaMalloc(&device_final_pixel_arr,final_pixel_arr_size));
    gpuErrchk(cudaMalloc(&device_anti_alias_pixel_arr,anti_alias_pixel_arr_size));
    gpuErrchk(cudaMalloc(&device_final_pixel_arr_vec3,sizeof(vec3)*num_pixels));

    // allocate memory random states arrays
    // curandState *d_rand_state;
    // gpuErrchk(cudaMalloc((void **)&d_rand_state, num_pixels*ns*sizeof(curandState)));
    // curandState *d_rand_state2;
    // gpuErrchk(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));
    
    //running the rand_init for a single rand state kernel
    // rand_init<<<1,1>>>(d_rand_state2);
    // gpuErrchk(cudaGetLastError());
    // gpuErrchk(cudaDeviceSynchronize());
    //for gray scale depth visualization
    
    float *max_distance;
    
    bool streamlined = true;
    //making up our world with pointers for non steamlined system
    if(!streamlined)
    {
        colliders *d_colliders;
        collider_list **d_world;
        collider_material *d_materials;
        collider_texture *d_textures;
        camera *d_camera;
        bvh_gpu_node *bvh_arr;
        bvh_gpu *bvh_tree;
        int no_of_objects_in_the_scene = 10000;
        int total_size_bvh  = (2*(2*no_of_objects_in_the_scene-1)+1)*sizeof(bvh_gpu_node);
        int bvh_tree_size = sizeof(bvh_gpu);
        aabb *list_of_bounding_boxes;
        int total_aabb_size = no_of_objects_in_the_scene*sizeof(aabb);
        
        gpuErrchk(cudaMallocManaged(&list_of_bounding_boxes,total_aabb_size));
        gpuErrchk(cudaMallocManaged(&bvh_arr,total_size_bvh));

        gpuErrchk(cudaMalloc(&d_colliders,
        no_of_objects_in_the_scene*sizeof(colliders)));

        gpuErrchk(cudaMalloc(&bvh_tree,bvh_tree_size));

        gpuErrchk(cudaMalloc(&d_materials,
        no_of_objects_in_the_scene*sizeof(collider_material)));

        gpuErrchk(cudaMalloc(&d_textures,
        no_of_objects_in_the_scene*sizeof(collider_texture)));

        gpuErrchk(cudaMalloc(&d_world,sizeof(collider_list*)));
        gpuErrchk(cudaMalloc(&d_camera,sizeof(camera)));
        clock_t start_creation,end_creation;
        std::cerr<<"Started creation and pre processing of data "<<std::endl;
        start_creation = clock();
        optimized_world_renderer<<<1,1>>>(d_colliders,d_materials,d_textures,d_world
        ,d_camera,list_of_bounding_boxes,
        nx,ny,no_of_objects_in_the_scene
        ,device_data,width,height);
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());
        std::cerr<<"\nEnded the creation of the world";
        //bvh_gpu arr creation
        host_bvh_tree_creation(list_of_bounding_boxes,
        bvh_arr,no_of_objects_in_the_scene);

        //bvh tree initialization
        initialize_bvh_tree<<<1,1>>>(bvh_arr,d_colliders,bvh_tree,
        no_of_objects_in_the_scene);
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());

        std::cerr<<"\nTree creation ends";


        end_creation = clock();
        float timer_seconds = ((double)(end_creation-start_creation))/CLOCKS_PER_SEC;
        std::cerr << "\nPre processing and creation of the world took " << timer_seconds<<"\n";

        clock_t start,end_render,end;
        start = clock();

        dim3 blocks_render(ns_x,ns_y);
        dim3 grid_render(nx,ny);
        // dim3 blocks_reduction = blocks_render;
        // dim3 grid_reduction = grid_render;

        dim3 blocks_reduction = ns_x*ns_y;
        dim3 grid_reduction = nx*ny;

        std::cerr << "Rendering a " << nx << "x" << ny << " image \n";
    
        
        // //allocating our rand states array
        // //printf("render_init started");
        // // render_init<<<grid_render,blocks_render>>>(nx, ny, d_rand_state);
        // // gpuErrchk(cudaGetLastError());
        // // gpuErrchk(cudaDeviceSynchronize());
        // //printf("\nrender_init ended");

        //starting the rendering process
        std::cerr << "Rendering started \n";
        optimized_render_tree<<<grid_render,blocks_render>>>(device_anti_alias_pixel_arr,
        nx,ny,ns,d_camera,bvh_tree,d_materials,d_textures);

        // optimized_render_list<<<grid_render,blocks_render>>>(device_anti_alias_pixel_arr,
        // nx,ny,ns,d_camera,d_world,d_materials,d_textures);
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());
        std::cerr << "Rendering ended \n";
        end_render = clock();
        timer_seconds = ((double)(end_render-start))/CLOCKS_PER_SEC;
        std::cerr << "Rendering complete took " << timer_seconds<<"\n";
        
        //starting our reduction kernel to find our final pixel array
        shared_reduction_interleaved_approach_complete_unrolling_2d<<<grid_reduction,blocks_reduction>>>(device_anti_alias_pixel_arr,
        device_final_pixel_arr,num_pixels*ns);
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());



        gpuErrchk(cudaMemcpy(host_final_pixel_arr,device_final_pixel_arr,final_pixel_arr_size,
        cudaMemcpyDeviceToHost));
        end = clock();

        timer_seconds = ((double)(end-start))/CLOCKS_PER_SEC;
        std::cerr<<"Reduction complete took "<<timer_seconds<<" seconds in the whole process\n";

    }
    else
    {
        triangle *d_list;
        camera *d_camera;
        bvh_gpu_node *bvh_arr;
        bvh_gpu_streamlined_tree *bvh_tree;
        bvh_gpu_node_sah *bvh_arr_sah;
        bvh_gpu_streamlined_tree_sah *bvh_tree_sah;
        vec3 *a,*b,*c;
        
        //budhha model preparation
        std::vector<vec3> vertices,face_values;
        //read_from_obj("./models/budhha.obj",vertices,face_values);
        int no_of_objects_in_the_scene = 12582;
        //for sah
        // int total_size_bvh_sah  = ((2*no_of_objects_in_the_scene-1))*sizeof(bvh_gpu_node_sah);
        // int bvh_tree_size_sah = sizeof(bvh_gpu_streamlined_tree_sah);
        //for normal
        int total_size_bvh  = (2*(2*no_of_objects_in_the_scene-1)+1)*sizeof(bvh_gpu_node);
        int bvh_tree_size = sizeof(bvh_gpu_streamlined_tree);
        
        aabb *list_of_bounding_boxes;
        int total_aabb_size = no_of_objects_in_the_scene*sizeof(aabb);
        
        
        gpuErrchk(cudaMallocManaged(&a,sizeof(vec3)*no_of_objects_in_the_scene));

        gpuErrchk(cudaMallocManaged(&max_distance,sizeof(float)));

        gpuErrchk(cudaMallocManaged(&b,sizeof(vec3)*no_of_objects_in_the_scene));
        gpuErrchk(cudaMallocManaged(&c,sizeof(vec3)*no_of_objects_in_the_scene));

        gpuErrchk(cudaMallocManaged(&list_of_bounding_boxes,total_aabb_size));
        
        gpuErrchk(cudaMallocManaged(&bvh_arr,total_size_bvh));
        //gpuErrchk(cudaMallocManaged(&bvh_arr_sah,total_size_bvh_sah));

        gpuErrchk(cudaMalloc(&d_list,
        no_of_objects_in_the_scene*sizeof(triangle)));

        gpuErrchk(cudaMalloc(&bvh_tree,bvh_tree_size));
        //gpuErrchk(cudaMalloc(&bvh_tree_sah,bvh_tree_size_sah));

        gpuErrchk(cudaMallocManaged(&d_camera,sizeof(camera)));

        clock_t start_creation,end_creation;
        std::cerr<<"Started creation and pre processing of data "<<std::endl;
        start_creation = clock();
        
        read_from_model("./models/machine_1.txt",a,b,c,vertices,face_values);
        
        optimized_world_renderer_streamlined_mpi<<<1,1>>>(d_list,a,b,c,d_camera,list_of_bounding_boxes,
        nx,ny,no_of_objects_in_the_scene);
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());


        std::vector<std::vector<float>> circle_points;
        trace_the_circle_points(0,0,0,5.89f,circle_points);
        std::cerr<<"\nEnded the creation of the world";
        
        //bvh_gpu arr creation
        // host_bvh_tree_creation_sah(list_of_bounding_boxes,
        // bvh_arr_sah,no_of_objects_in_the_scene);

        host_bvh_tree_creation(list_of_bounding_boxes,
        bvh_arr,no_of_objects_in_the_scene);

        //return;

        //bvh tree initialization
        // initialize_bvh_tree_streamlined_sah<<<1,1>>>(bvh_arr_sah,d_list,bvh_tree_sah
        // ,(2*no_of_objects_in_the_scene-1));

        initialize_bvh_tree_streamlined<<<1,1>>>(bvh_arr,d_list,bvh_tree
        ,2*(2*no_of_objects_in_the_scene-1));

        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());

        std::cerr<<"\nTree creation ends";


        end_creation = clock();
        float timer_seconds = ((double)(end_creation-start_creation))/CLOCKS_PER_SEC;
        std::cerr << "\nPre processing and creation of the world took " << timer_seconds<<"\n";

        clock_t start,end_render,end;
        start = clock();

        //for cases when sampling not enabled
        dim3 blocks_render(8,8);
        dim3 grid_render(nx/8,ny/8);

        //sampling
        // dim3 blocks_render(ns_x,ns_y);
        // dim3 grid_render(nx,ny);
        

        dim3 blocks_reduction = ns_x*ns_y;
        dim3 grid_reduction = nx*ny;
        
        std::cerr << "Rendering a " << nx << "x" << ny << " image \n";
    

        std::cerr << "Rendering started \n";
        GLFWwindow *window;
        bool first_time = true;
        for(int i=0;i<circle_points.size();i++)
        {
            //vec3 lookfrom(-5.0f, 0.0f, 2.0f);
            vec3 lookfrom(circle_points[i][0],circle_points[i][1],
            circle_points[i][2]);
            vec3 lookat(0,0,0);
            d_camera->recalculate_look_from(lookfrom,lookat,vec3(0,1,0),35.0,float(nx)/float(ny),
            (lookfrom-lookat).length());
            //start the loop here
            // optimized_render_tree<<<grid_render,blocks_render>>>(device_anti_alias_pixel_arr,
            // nx,ny,ns,d_camera,bvh_tree,d_materials,d_textures);

            // optimized_render_list<<<grid_render,blocks_render>>>(device_anti_alias_pixel_arr,
            // nx,ny,ns,d_camera,d_world,d_materials,d_textures);
            *max_distance = FLT_MIN;
            optimized_render_streamlined_tree<<<grid_render,blocks_render>>>(device_final_pixel_arr_vec3,
            nx,ny,d_camera,bvh_tree,d_list,max_distance);

            // optimized_render_streamlined_tree_sampling<<<grid_render,blocks_render>>>(device_anti_alias_pixel_arr,
            // nx,ny,d_camera,bvh_tree,d_list);

            // optimized_render_streamlined_tree_sampling_sah<<<grid_render,blocks_render>>>(device_anti_alias_pixel_arr,
            // nx,ny,d_camera,bvh_tree_sah,d_list);

            gpuErrchk(cudaGetLastError());
            gpuErrchk(cudaDeviceSynchronize());
            end_render = clock();
            timer_seconds = ((double)(end_render-start))/CLOCKS_PER_SEC;
            //std::cerr << "Rendering complete took " << timer_seconds<<"\n";
            
            //starting our reduction kernel to find our final pixel array
            // shared_reduction_interleaved_approach_complete_unrolling_2d<<<grid_reduction,blocks_reduction>>>(device_anti_alias_pixel_arr,
            // device_final_pixel_arr,num_pixels*ns);

            depth_visualization_kernel<<<(nx*ny)/64,64>>>(device_final_pixel_arr_vec3,device_final_pixel_arr,*max_distance);
            gpuErrchk(cudaGetLastError());
            gpuErrchk(cudaDeviceSynchronize());
            
            gpuErrchk(cudaMemcpy(host_final_pixel_arr,device_final_pixel_arr,final_pixel_arr_size,
            cudaMemcpyDeviceToHost));
            end = clock();

            timer_seconds = ((double)(end-start))/CLOCKS_PER_SEC;
            //std::cerr<<"Reduction complete took "<<timer_seconds<<" seconds in the whole process\n";
            //std::cerr<<"Depth visualization complete took "<<timer_seconds<<" seconds in the whole process\n";
            //std::cerr<<"Rendering complete took "<<timer_seconds<<" seconds in the whole process\n";
            show_windows_animated(host_final_pixel_arr,nx,ny,first_time,window);
            //end the loop here
        }
        
    }

    // device_memory_check<<<1,1>>>(d_colliders,d_materials,d_textures,
    // d_world,d_camera,no_of_objects_in_the_scene);
    // gpuErrchk(cudaGetLastError());
    // gpuErrchk(cudaDeviceSynchronize());

    bool greyscale_visualization = false;

    // for(int i=0;i<num_pixels*3;i++)
    // {
    //     std::cerr<<(int)host_final_pixel_arr[i]<<std::endl;
    // }
    //show_windows(host_final_pixel_arr,nx,ny);
    // std::cout<<"P3\n"<<nx<<" "<<ny<<"\n255\n";
    // for(int j= ny-1;j>=0;j--){
    //     for(int i = 0;i<nx;i++){
    //         size_t pixel_index = (j*nx + i)*3;
    //         // float r = host_final_pixel_arr[pixel_index].r();
    //         // float g = host_final_pixel_arr[pixel_index].g();
    //         // float b = host_final_pixel_arr[pixel_index].b();
    //         // if(greyscale_visualization)
    //         // {
    //         //     if(r>0.0f)
    //         //         r/=*max_distance;
    //         //     if(g>0.0f)
    //         //         g/=*max_distance;
    //         //     if(b>0.0f)
    //         //         b/=*max_distance;
    //         // }
    //         // int ir = int(255.99*r);
    //         // int ig = int(255.99*g);
    //         // int ib = int(255.99*b);

    //         int ir = host_final_pixel_arr[pixel_index];
    //         int ig = host_final_pixel_arr[pixel_index+1];
    //         int ib = host_final_pixel_arr[pixel_index+2];

            
    //         std::cout<<ir<<" "<<ig<<" "<<ib<<"\n";
    //     }
    // }

    // std::cout<<"P3\n"<<nx<<" "<<ny<<"\n255\n";
    // for(int j= ny-1;j>=0;j--){
    //     for(int i = 0;i<nx;i++){
    //         size_t pixel_index = j*nx + i;
    //         float r = host_final_pixel_arr[pixel_index].r();
    //         float g = host_final_pixel_arr[pixel_index].g();
    //         float b = host_final_pixel_arr[pixel_index].b();
    //         if(greyscale_visualization)
    //         {
    //             if(r>0.0f)
    //                 r/=*max_distance;
    //             if(g>0.0f)
    //                 g/=*max_distance;
    //             if(b>0.0f)
    //                 b/=*max_distance;
    //         }
    //         int ir = int(255.99*r);
    //         int ig = int(255.99*g);
    //         int ib = int(255.99*b);

            
    //         std::cout<<ir<<" "<<ig<<" "<<ib<<"\n";
    //     }
    // }

    // free_world<<<1,1>>>(d_list,d_world,d_camera,no_of_objects_in_the_scene);
    // gpuErrchk(cudaDeviceSynchronize());
    // gpuErrchk(cudaGetLastError());
    // gpuErrchk(cudaFree(d_camera));
    // gpuErrchk(cudaFree(d_world));
    // gpuErrchk(cudaFree(d_list));
    // gpuErrchk(cudaFree(device_final_pixel_arr));
    // free(host_final_pixel_arr);
    // cudaDeviceReset();
    

}*/


// __global__ void create_world(hittable **d_list,hittable **d_world,
// camera **d_camera,float nx,float ny,int size,curandState *local_rand_state
// ,unsigned char *data,int width , int height){
//     if(threadIdx.x==0&&blockIdx.x==0)
//     {
//         //curandState local_rand_state = *rand_state;
//         checker_texture *checker = new checker_texture(vec3(0.9,0.2,0.2),vec3(0.2,0.9,0.2));

//         d_list[0] = new sphere(vec3(0,-1000.0,-1), 1000.0f,
//                                 new lambertian(checker));
//         int i = 1;
//         for(int a = -11; a < 11; a++) {
//             for(int b = -11; b < 11; b++) {
//                 float choose_mat = RND;
//                 vec3 center(a+RND,0.2,b+RND);
//                 if(choose_mat < 0.8f) {
//                     vec3 center2 = center + vec3(0,random_double(local_rand_state,0.0,0.05),0);
                    
//                     d_list[i++] = new moving_sphere(center,center2,0.0f,0.1f,0.2f,
//                                                 new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
//                 }
//                  else if(choose_mat < 0.95f) {
//                     d_list[i++] = new sphere(center, 0.2f,
//                                                  new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
//                 }
//                 else {
//                     d_list[i++] = new sphere(center, 0.2f, new dielectric(1.5));
                                               
//                 }
//             }
//         }
//         d_list[i++] = new sphere(vec3(0, 1,0),  1.0f, new dielectric(1.5));
//         image_texture *earth_texture = new image_texture(&data,width,height);
//         lambertian *earth_surface = new lambertian(earth_texture);
//         d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0f, earth_surface);
//         //d_list[i++] = new sphere(vec3(4, 1, 0),  1.0f, new metal(vec3(0.7, 0.6, 0.5), 0.0));
//         d_list[i++] = new sphere(vec3(4, 1, 0), 1.0f, earth_surface);
//         //*rand_state = local_rand_state;
//         *d_world  = new hittable_list(d_list, size);

//         vec3 lookfrom(13,2,3);
//         vec3 lookat(0,0,0);
//         float dist_to_focus = 10.0; (lookfrom-lookat).length();
//         float aperture = 0.1;
//         *d_camera   = new camera(lookfrom,lookat,vec3(0,1,0),30.0,float(nx)/float(ny),aperture,dist_to_focus
//         ,0.0,1.0);
//     }
    
// }
// __global__ void create_world_bvh(hittable **d_list,hittable **d_world,
// camera **d_camera,float nx,float ny,int size,curandState *local_rand_state
// ,unsigned char *data,int width , int height){

//     if(threadIdx.x==0&&blockIdx.x==0)
//     {
//         lambertian *white_surface = new lambertian(vec3(1,0.73,1));
        
//         int total_no_spheres = 500;
//         image_texture *earth_texture = new image_texture(&data,width,height);
//         lambertian *earth_surface = new lambertian(earth_texture);
//         hittable **temp_obj = new hittable*[total_no_spheres];
//         for(int i=0;i<total_no_spheres;i++)
//         {
//             //white_surface->albedo->color_value=vec3(RND*1.0f,RND*1.0f,RND*1.0f);
//             temp_obj[i] = new sphere(random_vector_in_range(local_rand_state,-2,2),0.2f,
//             new lambertian(vec3(RND*1.0f,RND*1.0f,RND*1.0f)));
//             //d_list[i] = new sphere(random_vector_in_range(rand_state,-2,2),0.2f,white_surface);
//         }
//         hittable_list box_of_sphere(temp_obj,total_no_spheres);
//         d_list[0] = new bvh_node(box_of_sphere,0,1,local_rand_state);
//         //*d_world = new hittable_list(d_list,size);
//         //diffuse_light *light = new diffuse_light(vec3(7,7,7));
//         // d_list[1] = new xz_rect(-1,1,-1,1,3,light);
//         // d_list[2] = new yz_rect(-1,1,-1,1,3,light);
//         // d_list[3] = new yz_rect(-1,1,-1,1,-3,light);
//         // d_list[4] = new xz_rect(-1,1,-1,1,-3,light);
//         *d_world = new hittable_list(d_list,size);

//         vec3 lookfrom(-5,0,12);
//         vec3 lookat(0,0,0);
//         float dist_to_focus = (lookfrom-lookat).length();
//         float aperture = 0.0;
//         *d_camera   = new camera(lookfrom,
//                                  lookat,
//                                  vec3(0,1,0),
//                                  40.0,
//                                  float(nx)/float(ny),
//                                  aperture,
//                                  dist_to_focus,0.0,1.0);
//     }

// }
// __global__ void create_world_defocus_blur(hittable **d_list, hittable **d_world, 
// camera **d_camera,float nx,float ny,int size,curandState *local_state) {
//     if (threadIdx.x == 0 && blockIdx.x == 0) {
//         d_list[0] = new sphere(vec3(0,0,-1), 0.5,
//                                 new lambertian(vec3(0.1, 0.2, 0.5)));
//         d_list[1] = new sphere(vec3(0,-100.5,-1), 100,
//                                 new lambertian(vec3(0.8, 0.8, 0.0)));
//         d_list[2] = new sphere(vec3(1,0,-1), 0.5,
//                                 new metal(vec3(0.8, 0.6, 0.2), 0.0));
//         d_list[3] = new sphere(vec3(-1,0,-1), 0.5,
//                                  new dielectric(1.5));
//         d_list[4] = new sphere(vec3(-1,0,-1), -0.45,
//                                  new dielectric(1.5));
//         *d_world = new hittable_list(d_list,size);
//         vec3 lookfrom(3,3,2);
//         vec3 lookat(0,0,-1);
//         float dist_to_focus = (lookfrom-lookat).length();
//         float aperture = 2.0;
//         *d_camera   = new camera(lookfrom,
//                                  lookat,
//                                  vec3(0,1,0),
//                                  20.0,
//                                  float(nx)/float(ny),
//                                  aperture,
//                                  dist_to_focus,0.0,1.0);
//     }
// }

// __global__ void create_world_artificial_light(hittable **d_list, hittable **d_world, 
// camera **d_camera,float nx,float ny,int size,curandState *local_state,
// unsigned char *data,int width , int height) {
//     if (threadIdx.x == 0 && blockIdx.x == 0) {
//         noise_texture *n = new noise_texture(*local_state,4);
//         diffuse_light *diffLight = new diffuse_light(vec3(4,4,4));


//         d_list[0] = new sphere(vec3(0,2,0), 2,
//                                 new lambertian(n));
//         d_list[1] = new sphere(vec3(0,-1000,0), 1000,
//                                 new lambertian(n));
//         d_list[2] = new xy_rect(3,5,1,3,-2,diffLight);
        

//         *d_world = new hittable_list(d_list,size);
//         vec3 lookfrom(26,3,6);
//         vec3 lookat(0,2,0);
//         float dist_to_focus = (lookfrom-lookat).length();
//         float aperture = 2.0;
//         *d_camera   = new camera(lookfrom,
//                                  lookat,
//                                  vec3(0,1,0),
//                                  20.0,
//                                  float(nx)/float(ny),
//                                  aperture,
//                                  dist_to_focus,0.0,1.0);
//     }
// }

// __global__ void create_world_cornell_box(hittable **d_list, hittable **d_world, 
// camera **d_camera,float nx,float ny,int size,curandState *local_state,
// unsigned char *data,int width , int height) {
//     if (threadIdx.x == 0 && blockIdx.x == 0) {
        
//         diffuse_light *diffLight = new diffuse_light(vec3(14,14,14));
//         lambertian *red = new lambertian(vec3(0.65,0.05,0.05));
//         lambertian *green = new lambertian(vec3(0.12,0.45,0.15));
//         lambertian *white = new lambertian(vec3(0.73,0.73,0.73));

//         d_list[0] = new yz_rect(0,555,0,555,555,green);
//         d_list[1] = new yz_rect(0,555,0,555,0,red);
//         d_list[2] = new xz_rect(213,343,227,332,554,diffLight);
//         d_list[3] = new xz_rect(0,555,0,555,0,white);
//         d_list[4] = new xz_rect(0,555,0,555,555,white);
//         d_list[5] = new xy_rect(0,555,0,555,555,white);
        
//         d_list[6] = new box(vec3(0,0,0),vec3(165,165,165),white);
//         d_list[6] = new rotate_y(d_list[6],-18,6);
//         d_list[6] = new translate(d_list[6],vec3(130,0,65));
        
        
//         d_list[7] = new box(vec3(0,0,0),vec3(165,330,165),white);
//         d_list[7] = new rotate_y(d_list[7],15,7);
//         d_list[7] = new translate(d_list[7],vec3(265,0,295));

//         *d_world = new hittable_list(d_list,size);
//         vec3 lookfrom(278,278,-800);
//         vec3 lookat(278,278,0);
//         float dist_to_focus = (lookfrom-lookat).length();
//         float aperture = 2.0;
//         *d_camera   = new camera(lookfrom,
//                                  lookat,
//                                  vec3(0,1,0),
//                                  40.0,
//                                  float(nx)/float(ny),
//                                  aperture,
//                                  dist_to_focus,0.0,1.0);
//     }
// }

// __global__ void create_world_smoke_cornell_box(hittable **d_list, hittable **d_world, 
// camera **d_camera,float nx,float ny,int size,curandState *local_state,
// unsigned char *data,int width , int height) {
//     if (threadIdx.x == 0 && blockIdx.x == 0) {
        
//         diffuse_light *diffLight = new diffuse_light(vec3(14,14,14));
//         lambertian *red = new lambertian(vec3(0.65,0.05,0.05));
//         lambertian *green = new lambertian(vec3(0.12,0.45,0.15));
//         lambertian *white = new lambertian(vec3(0.73,0.73,0.73));

//         d_list[0] = new yz_rect(0,555,0,555,555,green);
//         d_list[1] = new yz_rect(0,555,0,555,0,red);
//         d_list[2] = new xz_rect(113,443,127,432,554,diffLight);
//         d_list[3] = new xz_rect(0,555,0,555,0,white);
//         d_list[4] = new xz_rect(0,555,0,555,555,white);
//         d_list[5] = new xy_rect(0,555,0,555,555,white);
        
//         d_list[6] = new box(vec3(0,0,0),vec3(165,165,165),white);
//         d_list[6] = new rotate_y(d_list[6],-18,6);
//         d_list[6] = new translate(d_list[6],vec3(130,0,65));
//         d_list[6] = new constant_medium(d_list[6],0.01,vec3(0,0,0),local_state);
        
//         d_list[7] = new box(vec3(0,0,0),vec3(165,330,165),white);
//         d_list[7] = new rotate_y(d_list[7],15,7);
//         d_list[7] = new translate(d_list[7],vec3(265,0,295));
//         d_list[7] = new constant_medium(d_list[7],0.01,vec3(1,1,1),local_state);
        
//         *d_world = new hittable_list(d_list,size);
//         vec3 lookfrom(278,278,-800);
//         vec3 lookat(278,278,0);
//         float dist_to_focus = (lookfrom-lookat).length();
//         float aperture = 2.0;
//         *d_camera   = new camera(lookfrom,
//                                  lookat,
//                                  vec3(0,1,0),
//                                  40.0,
//                                  float(nx)/float(ny),
//                                  aperture,
//                                  dist_to_focus,0.0,1.0);
//     }
// }

// __global__ void create_world_perlin_noise(hittable **d_list, hittable **d_world, 
// camera **d_camera,float nx,float ny,int size,curandState *local_state,
// unsigned char *data,int width , int height) {
//     if (threadIdx.x == 0 && blockIdx.x == 0) {
        
//         noise_texture *n= new noise_texture(*local_state,4);

//         d_list[0] = new sphere(vec3(0,-1000,0), 1000,
//                                 new lambertian(n));
//         d_list[1] = new sphere(vec3(0,2,0), 2,
//                                 new lambertian(n));
        
//         *d_world = new hittable_list(d_list,size);
//         vec3 lookfrom(13,2,3);
//         vec3 lookat(0,0,0);
//         float dist_to_focus = 10.0;
//         float aperture = 0.0;
//         *d_camera   = new camera(lookfrom,
//                                  lookat,
//                                  vec3(0,1,0),
//                                  20.0,
//                                  float(nx)/float(ny),
//                                  aperture,
//                                  dist_to_focus,0.0,1.0);
//     }
// }

// __global__ void create_world_earth_texture(hittable **d_list, hittable **d_world, 
// camera **d_camera,float nx,float ny,int size,curandState *local_state,
// unsigned char *data,int width , int height)
// {
//     if(threadIdx.x==0 && blockIdx.x==0){
//         image_texture *earth_texture = new image_texture(&data,width,height);
//         lambertian *earth_surface = new lambertian(earth_texture);
//         d_list[0] = new sphere(vec3(0,0,0),2,earth_surface);

//         *d_world = new hittable_list(d_list,size);
//         vec3 lookfrom(13,2,3);
//         vec3 lookat(0,0,0);
//         float dist_to_focus = 10.0;
//         float aperture = 0.0;
//         *d_camera   = new camera(lookfrom,
//                                  lookat,
//                                  vec3(0,1,0),
//                                  20.0,
//                                  float(nx)/float(ny),
//                                  aperture,
//                                  dist_to_focus,0.0,1.0);
//     }
// }



// __global__ void triangle_renderer(hittable **d_list, hittable **d_world, 
// camera **d_camera,float nx,float ny,int size,curandState *local_rand_state,
// unsigned char *data,int width , int height) {

//     //lambertian *normal_material = new lambertian(vec3(0.48,0.83,0.53));

//     int no_of_triangles = 500;
//     for(int i = 0; i<no_of_triangles;i++)
//     {
//         vec3 rand_point = random_vector_in_range(local_rand_state,-4,4);

//         vec3 a = rand_point - random_vector_in_range(local_rand_state,-0.5,0.5);
//         vec3 b = rand_point - random_vector_in_range(local_rand_state,-0.5,0.5);
//         vec3 c = rand_point - random_vector_in_range(local_rand_state,-0.5,0.5);

//         d_list[i] = new triangle(a,b,c,new lambertian(vec3(RND*1.0f,RND*1.0f,RND*1.0f)));
//     }

//     diffuse_light *diff_light = new diffuse_light(vec3(11,11,11));
//     d_list[500] = new xz_rect(-5,5,-5,5,4,diff_light);
    

//     *d_world = new hittable_list(d_list,size);

//     vec3 lookfrom(-5,0,14);
//         vec3 lookat(0,0,0);
//         float dist_to_focus = (lookfrom-lookat).length();
//         float aperture = 0.0;
//         *d_camera   = new camera(lookfrom,
//                                  lookat,
//                                  vec3(0,1,0),
//                                  70.0,
//                                  float(nx)/float(ny),
//                                  aperture,
//                                  dist_to_focus,0.0,1.0);

// }

// __global__ void final_image_renderer_everything(hittable **d_list, hittable **d_world, 
// camera **d_camera,float nx,float ny,int size,curandState *local_state,
// unsigned char *data,int width , int height) {

    
//     lambertian *ground = new lambertian(vec3(0.48,0.83,0.53));

//     int boxes_per_side = 20;

//     for (int i = 0; i < boxes_per_side; i++) {
//         for (int j = 0; j < boxes_per_side; j++) {
//             auto w = 100.0;
//             auto x0 = -1000.0 + i*w;
//             auto z0 = -1000.0 + j*w;
//             auto y0 = 0.0;
//             auto x1 = x0 + w;
//             auto y1 = random_double(local_state,1,101);
//             auto z1 = z0 + w;

//             d_list[i*boxes_per_side+j] = new 
//             box(vec3(x0,y0,z0), vec3(x1,y1,z1), ground);
//         }
//     }

//     diffuse_light *light = new diffuse_light(vec3(7,7,7));
//     d_list[400] = new xz_rect(123,423,147,412,554,light);

//     *d_world = new hittable_list(d_list,size);

//     vec3 lookfrom(478,278,-600);
//     vec3 lookat(278,278,0);
//     float dist_to_focus = (lookfrom-lookat).length();
//     float aperture = 0.0;
//     *d_camera   = new camera(lookfrom,
//                                 lookat,
//                                 vec3(0,1,0),
//                                 40.0,
//                                 float(nx)/float(ny),
//                                 aperture,
//                                 dist_to_focus,0.0,1.0);


// }

// __global__ void review_three_renderer(hittable **d_list, hittable **d_world, 
// camera **d_camera,float nx,float ny,int size,curandState *local_rand_state,
// unsigned char *data,int width , int height) {

//     lambertian *ground = new lambertian(vec3(0.47,0.47,0.47));


//     // d_list[0] = new sphere(vec3(-5,3,0),2,new lambertian(vec3(RND*1.0f,RND*1.0f,RND*1.0f)));
//     // d_list[1] = new moving_sphere(vec3(-5,8,0),vec3(-4.8,8,0),0.0f,0.1f,2.0f,
//     //                     new lambertian(vec3(RND*1.0f,RND*1.0f,RND*1.0f)));
//     // d_list[2] = new sphere(vec3(0,-1000,0),1000,ground);
//     // d_list[3] = new triangle(vec3(3,0,0),vec3(3,0,-6),vec3(6,8,-3),new lambertian(vec3(RND*1.0f,RND*1.0f,RND*1.0f)));
//     // d_list[4] = new triangle(vec3(3,0,-6),vec3(9,0,-6),vec3(6,8,-3),new lambertian(vec3(RND*1.0f,RND*1.0f,RND*1.0f)));
//     // d_list[5] = new triangle(vec3(9,0,-6),vec3(9,0,0),vec3(6,8,-3),new lambertian(vec3(RND*1.0f,RND*1.0f,RND*1.0f)));
//     // d_list[6] = new triangle(vec3(9,0,0),vec3(3,0,0),vec3(6,8,-3),new lambertian(vec3(RND*1.0f,RND*1.0f,RND*1.0f)));
//     // d_list[7] = new sphere(vec3(-12,3,0),2,new lambertian(vec3(RND*1.0f,RND*1.0f,RND*1.0f)));
//     // d_list[8] = new moving_sphere(vec3(-12,8,0),vec3(-12,8.2,0),0.0f,0.1f,2.0f,
//     //                     new lambertian(vec3(RND*1.0f,RND*1.0f,RND*1.0f)));
//     // d_list[9] = new triangle(vec3(12,3,0),vec3(12,3,-6),vec3(15,9,-3),new lambertian(vec3(RND*1.0f,RND*1.0f,RND*1.0f)));
//     // d_list[10] = new triangle(vec3(12,3,-6),vec3(18,0,-6),vec3(15,9,-3),new lambertian(vec3(RND*1.0f,RND*1.0f,RND*1.0f)));
//     // d_list[11] = new triangle(vec3(18,0,-6),vec3(18,0,0),vec3(15,9,-3),new lambertian(vec3(RND*1.0f,RND*1.0f,RND*1.0f)));
//     // d_list[12] = new triangle(vec3(18,0,0),vec3(12,3,0),vec3(15,9,-3),new lambertian(vec3(RND*1.0f,RND*1.0f,RND*1.0f)));

    
//     // d_list[13] = new xy_rect(-40,0,0,20,-16,new lambertian(vec3(RND*0.8f,RND*0.8f,RND*0.8f)));
//     // d_list[13] = new rotate_y(d_list[13],18);

//     // d_list[14] = new xy_rect(0,40,0,20,-16,new lambertian(vec3(RND*1.0f,RND*1.0f,RND*1.0f)));
//     // d_list[14] = new rotate_y(d_list[14],-18);

//     // d_list[0] = new sphere(vec3(-5,3,0),2,new lambertian(vec3(1.0f,1.0f,1.0f)));
//     // d_list[0] = new constant_medium(d_list[0],0.5,vec3(1,1,1),local_rand_state);
//     // d_list[1] = new moving_sphere(vec3(-5,8,0),vec3(-4.8,8,0),0.0f,0.1f,2.0f,
//     //                     new lambertian(vec3(RND*1.0f,RND*1.0f,RND*1.0f)));
//     // d_list[2] = new sphere(vec3(0,-1000,0),1000,ground);
//     // d_list[3] = new triangle(vec3(3,0,0),vec3(3,0,-6),vec3(6,8,-3),
//     // new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.0f*RND));
//     // d_list[4] = new triangle(vec3(3,0,-6),vec3(9,0,-6),vec3(6,8,-3),new lambertian(vec3(RND*1.0f,RND*1.0f,RND*1.0f)));
//     // d_list[5] = new triangle(vec3(9,0,-6),vec3(9,0,0),vec3(6,8,-3),new lambertian(vec3(RND*1.0f,RND*1.0f,RND*1.0f)));
//     // d_list[6] = new triangle(vec3(9,0,0),vec3(3,0,0),vec3(6,8,-3),new lambertian(vec3(RND*1.0f,RND*1.0f,RND*1.0f)));
//     // d_list[7] = new sphere(vec3(-12,3,0),2,new dielectric(1.5f));
//     // d_list[8] = new moving_sphere(vec3(-12,8,0),vec3(-12,8.2,0),0.0f,0.1f,2.0f,
//     //                     new lambertian(vec3(RND*1.0f,RND*1.0f,RND*1.0f)));
//     // d_list[9] = new triangle(vec3(12,3,0),vec3(12,3,-6),vec3(15,9,-3),
//     // new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.0f*RND));
//     // d_list[10] = new triangle(vec3(12,3,-6),vec3(18,0,-6),vec3(15,9,-3),new lambertian(vec3(RND*1.0f,RND*1.0f,RND*1.0f)));
//     // d_list[11] = new triangle(vec3(18,0,-6),vec3(18,0,0),vec3(15,9,-3),new lambertian(vec3(RND*1.0f,RND*1.0f,RND*1.0f)));
//     // d_list[12] = new triangle(vec3(18,0,0),vec3(12,3,0),vec3(15,9,-3),new lambertian(vec3(RND*1.0f,RND*1.0f,RND*1.0f)));
//     noise_texture *n= new noise_texture(*local_rand_state,4);
//     image_texture *earth_texture = new image_texture(&data,width,height);
//     checker_texture *checker = new checker_texture(vec3(0.9,0.2,0.2),vec3(0.2,0.9,0.2));
//     d_list[0] = new sphere(vec3(-5,3,0),2,new lambertian(vec3(1.0f,1.0f,1.0f)));
//     d_list[0] = new constant_medium(d_list[0],0.5,vec3(1,1,1),local_rand_state);
//     d_list[1] = new sphere(vec3(-5,8,0),2.0f,
//                         new lambertian(n));
//     d_list[2] = new sphere(vec3(0,-1000,0),1000,ground);
//     d_list[3] = new triangle(vec3(3,0,0),vec3(3,0,-6),vec3(6,8,-3),
//     new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.0f*RND));
//     d_list[4] = new triangle(vec3(3,0,-6),vec3(9,0,-6),vec3(6,8,-3),new lambertian(vec3(RND*1.0f,RND*1.0f,RND*1.0f)));
//     d_list[5] = new triangle(vec3(9,0,-6),vec3(9,0,0),vec3(6,8,-3),new lambertian(vec3(RND*1.0f,RND*1.0f,RND*1.0f)));
//     d_list[6] = new triangle(vec3(9,0,0),vec3(3,0,0),vec3(6,8,-3),new lambertian(n));
//     d_list[7] = new sphere(vec3(-12,3,0),2,new lambertian(earth_texture));
//     d_list[8] = new moving_sphere(vec3(-12,8,0),vec3(-12,8.2,0),0.0f,0.1f,2.0f,
//                         new lambertian(vec3(RND*1.0f,RND*1.0f,RND*1.0f)));
//     d_list[9] = new triangle(vec3(12,3,0),vec3(12,3,-6),vec3(15,9,-3),
//     new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.0f*RND));
//     d_list[10] = new triangle(vec3(12,3,-6),vec3(18,0,-6),vec3(15,9,-3),new lambertian(vec3(RND*1.0f,RND*1.0f,RND*1.0f)));
//     d_list[11] = new triangle(vec3(18,0,-6),vec3(18,0,0),vec3(15,9,-3),new lambertian(vec3(RND*1.0f,RND*1.0f,RND*1.0f)));
//     d_list[12] = new triangle(vec3(18,0,0),vec3(12,3,0),vec3(15,9,-3),new lambertian(checker));
    
//     d_list[13] = new xy_rect(-40,0,0,20,-16,
//     new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.0f*RND));
//     d_list[13] = new rotate_y(d_list[13],18,13);

//     d_list[14] = new xy_rect(0,40,0,20,-16,
//     new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.0f*RND));
//     d_list[14] = new rotate_y(d_list[14],-18,14);


//     *d_world = new hittable_list(d_list,size);

//     vec3 lookfrom(0,5,20);
//         vec3 lookat(0,5,0);
//         float dist_to_focus = (lookfrom-lookat).length();
//         float aperture = 0.0;
//         *d_camera   = new camera(lookfrom,
//                                  lookat,
//                                  vec3(0,1,0),
//                                  70.0,
//                                  float(nx)/float(ny),
//                                  aperture,
//                                  dist_to_focus,0.0,1.0);
    

// }



            // triangle *t = new triangle(a,b,c,i);
            // d_colliders[i] = colliders(t,triangle_type_index);

            // sphere *s = new sphere(a,rand_radius,i);
            // d_colliders[i] = colliders(s,sphere_type_index);

            // lambertian *l =new lambertian();
            // d_materials[i] = collider_material(l,lambertian_type_index);

            // solid_color *s_c = new solid_color(vec3(RND*1.0f,RND*1.0f,RND*1.0f));
            // d_textures[i] = collider_texture(s_c,solid_color_type_index);

            // noise_texture *n = new noise_texture(localState,4);
            // d_textures[i] = collider_texture(n,noise_texture_type_index);

            // checker_texture *c_t = new checker_texture(vec3(0.8f,1.0f,0.3f),vec3(0.5f,0.2f,1.0f));
            // d_textures[i] = collider_texture(c_t,checker_texture_type_index);

            // image_texture *i_t = new image_texture(&data,width,height);
            // d_textures[i] = collider_texture(i_t,image_texture_type_index);

            // yz_rect *yzr = new yz_rect(b.x(),b.x()+2.0f,b.z(),
            // b.z()+0.5f,b.y(),i);

            //box *b = new box(a,a+vec3(0.5f,0.6f,0.7f),i);

            //sphere *s = new sphere(a,rand_radius,i);
            
            // constant_medium *c_m = new constant_medium(b,
            // 0.5f,i,&localState);
            //d_colliders[i] = colliders(b,xz_rect_type_index);
            
            // isotropic *iso = new isotropic();
            // d_materials[i] = collider_material(iso,isotropic_type_index);
            // rotate_y *rotate_y_xy = new rotate_y(yzr,15,i);

            // d_colliders[i] = colliders(rotate_y_xy,rotate_y_index);

            

            // lambertian *l = new lambertian();
            // d_materials[i] = collider_material(l,lambertian_type_index);

            // solid_color *s_c = new solid_color(vec3(RND*1.0f,RND*1.0f,RND*1.0f));
            // d_textures[i] = collider_texture(s_c,solid_color_type_index);