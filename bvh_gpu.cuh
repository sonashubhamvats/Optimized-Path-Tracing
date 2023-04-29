#ifndef BVH_H
#define BVH_H

#include "constants.cuh"
#include "collider.cuh"
#include "collider_list.cuh"
#include "aabb.cuh"

#include <curand_kernel.h>

struct bvh_gpu_node{
    int index_collider_list;
    int index_on_bvh_array;
    aabb box;

    __device__ __host__ bvh_gpu_node(){}

    __device__ __host__ bvh_gpu_node(int i_c_l,int i_b_a,aabb &b):
    index_collider_list(i_c_l),index_on_bvh_array(i_b_a),
    box(b){}

};

struct bvh_gpu_node_sah{
    int index_collider_list;
    int index_on_bvh_array;
    int left_child_index_on_bvh_array;
    int right_child_index_on_bvh_array;
    aabb box;

    __device__ __host__ bvh_gpu_node_sah(){
    }

    __device__ __host__ bvh_gpu_node_sah(int i_c_l,int i_b_a,int n_l,int n_r,aabb &b):
    index_collider_list(i_c_l),index_on_bvh_array(i_b_a),
    left_child_index_on_bvh_array(n_l),right_child_index_on_bvh_array(n_r),
    box(b){}

};



class bvh_gpu{
    
    bvh_gpu_node *bvh_node_arr;
    colliders *list_of_colliders;
    int bvh_node_arr_size;

    public:

        __device__ bvh_gpu(bvh_gpu_node *b_n_a,colliders *l_o_c,
        int size):
        bvh_node_arr(b_n_a),list_of_colliders(l_o_c),
        bvh_node_arr_size(size){}

        __device__ bool hit(const ray &r,float t_min,
        float t_max,collider_record &rec,int index)
        {
            
            int s_here[25];
            int index_on_bvh_arr=0;
            int t=-1;
            s_here[++t]=index_on_bvh_arr;
            bool has_hit = false;
            bool here = false;
            float closest_so_far = t_max;
            collider_record temp_record;
            while(t!=-1)
            {
                bvh_gpu_node top = bvh_node_arr[s_here[t]];
                --t;
                int dis=0;
                if(top.box.hit(r,t_min,t_max,dis))
                {

                    if(top.index_collider_list==-1)
                    {
                        index_on_bvh_arr = top.index_on_bvh_array;
                        s_here[++t]=(2*index_on_bvh_arr+1);
                        s_here[++t]=(2*index_on_bvh_arr+2);
                    }
                    else
                    {

                        if(list_of_colliders[top.index_collider_list]
                        .hit(r,t_min,closest_so_far,temp_record)){
                            has_hit = true;
                            closest_so_far = temp_record.t;
                            rec = temp_record;
                        }
                        
                    }
                }
            }

            return has_hit;
        }
        
};

//same as before with the exception being that I am using just triangles to render my scene now
class bvh_gpu_streamlined_tree{
    
    public:
        bvh_gpu_node *bvh_node_arr;
        triangle *list_of_colliders;
        int bvh_node_arr_size;

    public:

        __device__ bvh_gpu_streamlined_tree(bvh_gpu_node *b_n_a,triangle *l_o_c,
        int size):
        bvh_node_arr(b_n_a),list_of_colliders(l_o_c),
        bvh_node_arr_size(size){}

        __device__ bool hit(const ray &r,float t_min,
        float t_max,collider_record &rec,int index,int s_here[])
        {
            
            
            int index_on_bvh_arr=0;
            int t=-1;
            s_here[++t]=index_on_bvh_arr;
            bool has_hit = false;
            bool here = false;
            float closest_so_far = t_max;
            collider_record temp_record;
            while(t!=-1)
            {
                bvh_gpu_node top = bvh_node_arr[s_here[t]];
                --t;
                int dis = 0;
                if(top.box.hit(r,t_min,t_max,dis))
                {

                    if(top.index_collider_list==-1)
                    {
                        index_on_bvh_arr = top.index_on_bvh_array;
                        s_here[++t]=(2*index_on_bvh_arr+1);
                        s_here[++t]=(2*index_on_bvh_arr+2);
                    }
                    else
                    {

                        if(list_of_colliders[top.index_collider_list]
                        .hit(r,t_min,closest_so_far,temp_record)){
                            has_hit = true;
                            closest_so_far = temp_record.t;
                            rec = temp_record;
                        }
                        
                    }
                }
            }

            return has_hit;
        }
        
};


class bvh_gpu_streamlined_tree_sah{
    
    bvh_gpu_node_sah *bvh_node_arr;
    triangle *list_of_colliders;
    int bvh_node_arr_size;

    public:

        __device__ bvh_gpu_streamlined_tree_sah(bvh_gpu_node_sah *b_n_a,triangle *l_o_c
        ,int size):
        bvh_node_arr(b_n_a),list_of_colliders(l_o_c),
        bvh_node_arr_size(size){}

        __device__ bool hit(const ray &r,float t_min,
        float t_max,collider_record &rec,int index,int s_here[])
        {
            
            int index_on_bvh_arr=0;
            int t=-1;
            s_here[++t]=index_on_bvh_arr;
            bool has_hit = false;
            bool here = false;
            
            float closest_so_far = t_max;
            collider_record temp_record;
            
            while(t!=-1)
            {
                bvh_gpu_node_sah top = bvh_node_arr[s_here[t]];
                --t;
                int dis=0;
                if(top.box.hit(r,t_min,t_max,dis))
                {
                    if(top.index_collider_list==-1)
                    {
                        index_on_bvh_arr = top.index_on_bvh_array;
                        s_here[++t]=(top.left_child_index_on_bvh_array);
                        s_here[++t]=(top.right_child_index_on_bvh_array);
                    }
                    else
                    {
                        if(list_of_colliders[top.index_collider_list]
                        .hit(r,t_min,closest_so_far,temp_record)){
                            has_hit = true;
                            closest_so_far = temp_record.t;
                            rec = temp_record;
                        }
                    }
                }
            }

            return has_hit;
        }
        
};

#endif

// if(bvh_node_arr[index].index_collider_list==-1)
// {
//     if(!bvh_node_arr[index].box.hit(r,t_min,t_max))
//     {
//         return false;
//     }

//     bool hit_left = hit(r,t_min,t_max,rec,2*index+1);
//     bool hit_right = hit(r,t_min,hit_left?rec.t:t_max,rec,2*index+2);

//     return hit_left||hit_right;
// }
// else
// {
//     if(!bvh_node_arr[index].box.hit(r,t_min,t_max))
//     {
//         return false;
//     }
//     return list_of_colliders[bvh_node_arr[index].index_collider_list].hit(r,t_min,
//     t_max,rec);
// }

// class stack_device{
//     public:
//         int *arr;
//         int size;
//         int top;
//         __device__ stack_device(int s)
//         {
//             size = s;
//             top = -1;
//             arr = new int[s];
//         }

//         __device__ bool is_empty(){
//             return (top==-1);
//         }

//         __device__ bool pop(){
//             if(top==-1)
//                 return false;
//             top-=1;
//             return true;
//         }

//         __device__ bool push(int t){
//             if(top==size-1)
//                 return false;
//             arr[++top] = t;
//             return true;
//         }

//         __device__ int r_top(){
//             return top;
//         }

// };
