#ifndef CONSTANT_MEDIUM_H
#define CONSTANT_MEDIUM_H

#include "constants.cuh"
#include "sphere.cuh"
#include "box.cuh"


class constant_medium{

    public:
        
        __device__ constant_medium(sphere *b,
        float d,int i_o_c_l,curandState *rand_state)
        : boundary_s(b),type_of_collder_here(sphere_type_index),
        negative_inv_density(-1/d),
        index_on_collider_list(i_o_c_l),
        local_rand_state(rand_state)
        {}

        __device__ constant_medium(box *b,
        float d,int i_o_c_l,curandState *rand_state)
        : boundary_b(b),type_of_collder_here(box_index),
        negative_inv_density(-1/d),
        index_on_collider_list(i_o_c_l),
        local_rand_state(rand_state)
        {}


        
        __device__ bool hit(
            const ray &r,float t_min,float t_max,collider_record &rec
        )const;

        __device__ __host__ bool bounding_box(float time0,
        float time1,aabb &output_box,int index)const{
            if(type_of_collder_here==sphere_type_index)
                return boundary_s->bounding_box(time0,time1,output_box,index);
            if(type_of_collder_here==box_index)
                return boundary_b->bounding_box(time0,time1,output_box,index);
            
        }

    public:
        sphere *boundary_s;
        box *boundary_b;
        int index_on_collider_list;
        int type_of_collder_here;
        float negative_inv_density;
        curandState *local_rand_state;
};

__device__ bool constant_medium::hit(const ray &r,float t_min,float t_max,
collider_record &rec)const{

    collider_record rec1,rec2;

    if(type_of_collder_here==sphere_type_index)
    {
        if(!boundary_s->hit(r,-infinity,infinity,rec1))
            return false;

        if(!boundary_s->hit(r,rec1.t+0.0001,infinity,rec2))
            return false;
    }

    if(type_of_collder_here==box_index)
    {
        if(!boundary_b->hit(r,-infinity,infinity,rec1))
            return false;

        if(!boundary_b->hit(r,rec1.t+0.0001,infinity,rec2))
            return false;
    }


    if(rec1.t<t_min)
        rec1.t = t_min;
    
    if(rec2.t>t_max)
        rec2.t = t_max;

    if(rec1.t>=rec2.t)
        return false;

    if(rec1.t<0)
        rec1.t = 0;
    
    const float hit_distance = 
    negative_inv_density *log(curand_uniform(local_rand_state));
    const float ray_length = r.direction().length();
    const float distance_inside_boundary = (rec2.t-rec1.t)*ray_length;

    if(hit_distance>distance_inside_boundary)
        return false;

    rec.t = rec1.t + hit_distance/ray_length;
    rec.p = r.point_at_parameter(rec.t);
    rec.normal = vec3(1,0,0);
    rec.front_face = true;
    rec.index_on_the_collider_list = index_on_collider_list;

    return true;


}


#endif