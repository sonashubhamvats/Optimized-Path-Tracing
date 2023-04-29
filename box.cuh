#ifndef BOX_H
#define BOX_H

#include "constants.cuh"
#include "aarect.cuh"


class box{
    public:
        vec3 box_min;
        vec3 box_max;
        xy_rect *xy_side_1,*xy_side_2;
        xz_rect *xz_side_1,*xz_side_2;
        yz_rect *yz_side_1,*yz_side_2;
        int index_on_collider_list;

        __device__ box(){}
        __device__ box(const vec3 &p0,const vec3 &p1,
        int i_o_c_l);

        __device__ bool hit(const ray &r,float t_min,
        float t_max,collider_record &rec)const;

        __device__ __host__ bool bounding_box(float time0,
        float time1,aabb &output_box,int index)const{
            output_box = aabb(box_min,box_max,index);
            return true;
        }
};

__device__ box::box(const vec3 &p0,const vec3 &p1,int i_o_c_l){
    box_min = p0;
    box_max = p1;

    xy_side_1 = new xy_rect(p0.x(),p1.x(),p0.y(),p1.y(),p1.z(),i_o_c_l);
    xy_side_2 = new xy_rect(p0.x(),p1.x(),p0.y(),p1.y(),p0.z(),i_o_c_l);

    xz_side_1 = new xz_rect(p0.x(),p1.x(),p0.z(),p1.z(),p1.y(),i_o_c_l);
    xz_side_2 = new xz_rect(p0.x(),p1.x(),p0.z(),p1.z(),p0.y(),i_o_c_l);

    yz_side_1 = new yz_rect(p0.y(),p1.y(),p0.z(),p1.z(),p1.x(),i_o_c_l);
    yz_side_2 = new yz_rect(p0.y(),p1.y(),p0.z(),p1.z(),p0.x(),i_o_c_l);

    index_on_collider_list = i_o_c_l;

    //sides = hittable_list(list_of_sides,6);
}


__device__ bool box::hit(const ray &r,float t_min,float t_max,
collider_record &rec)const{
    float closest_so_far= t_max;
    bool hit_anything = false;

    if(xy_side_1->hit(r,t_min,closest_so_far,
    rec)){
        hit_anything = true;
        closest_so_far = rec.t;
    }
    if(xy_side_2->hit(r,t_min,closest_so_far,
    rec)){
        hit_anything = true;
        closest_so_far = rec.t;
    }
    if(yz_side_1->hit(r,t_min,closest_so_far,
    rec)){
        hit_anything = true;
        closest_so_far = rec.t;
    }
    if(yz_side_2->hit(r,t_min,closest_so_far,
    rec)){
        hit_anything = true;
        closest_so_far = rec.t;
    }
    if(xz_side_1->hit(r,t_min,closest_so_far,
    rec)){
        hit_anything = true;
        closest_so_far = rec.t;
    }
    if(xz_side_2->hit(r,t_min,closest_so_far,
    rec)){
        hit_anything = true;
        closest_so_far = rec.t;
    }
    return hit_anything;
}


#endif