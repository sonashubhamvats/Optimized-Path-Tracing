#ifndef COLLIDER_LIST_H
#define COLLIDER_LIST_H

#include "collider.cuh"
#include "aabb.cuh"

class collider_list{
    public:
        __device__ collider_list(){}
        __device__ collider_list(colliders *l,int n) :list(l),
        list_size(n){};
        __device__ virtual bool hit(const ray &r,float tmin,
        float tmax, collider_record& rec) const;
        __device__ __host__ bool bounding_box(
            float time0,float time1, aabb &output_box,int index
        )const;

        
        
        colliders *list;
        int list_size;
};

__device__ bool collider_list::hit(const ray &r,float t_min,
float t_max, collider_record &rec) const{
    collider_record temp_record;

    bool hit_anything = false;
    float closest_so_far = t_max;

    for(int i=0;i<list_size;i++)
    {
        if(list[i].hit(r,t_min,closest_so_far,temp_record)){
            hit_anything = true;
            closest_so_far = temp_record.t;
            rec = temp_record;
        }
    }

    return hit_anything;
}

__device__ __host__ bool collider_list::bounding_box(float time0,float time1,
aabb &output_box,int index)const {

    if(list_size == 0)
        return false;

    aabb temp_box;
    bool first_box = true;

    for(int i=0;i<list_size;i++)
    {
        colliders object = list[i];
        if(!object.bounding_box(time0,time1,temp_box,i))
            return false;
        output_box = first_box ? temp_box : surrounding_box(output_box,
        temp_box);
        first_box = false;
    }

    return true;
}

#endif