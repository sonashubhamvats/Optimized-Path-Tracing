#ifndef COLLIDER_H
#define COLLIDER_H

#include "constants.cuh"
#include "aabb.cuh"
#include "triangle.cuh"
#include "sphere.cuh"
#include "aarect.cuh"
#include "constant_medium.cuh"
#include "moving_spheres.cuh"
#include "box.cuh"
#include "rotate_collider_y.cuh"
#include "collider_record.cuh"

struct colliders{

    triangle *triangle_collider;
    sphere *sphere_collider;
    xy_rect *xy_rect_collider;
    xz_rect *xz_rect_collider;
    yz_rect *yz_rect_collider;
    constant_medium *constant_medium_collider;
    moving_sphere *moving_sphere_collider;
    box *box_collider;
    rotate_y *rotate_y_collder;
    int type_of_collider;

    __device__ colliders(triangle *t,int t_o_c){
            
        triangle_collider=t;
        type_of_collider = t_o_c;
        
    }

    __device__ colliders(sphere *s,int t_o_c){
        sphere_collider = s;
        type_of_collider = t_o_c;
    }

    __device__ colliders(xy_rect *xyr,int t_o_c){
        xy_rect_collider = xyr;
        type_of_collider = t_o_c;
    }

    __device__ colliders(xz_rect *xzr,int t_o_c){
        xz_rect_collider = xzr;
        type_of_collider = t_o_c;
    }

    __device__ colliders(yz_rect *yzr,int t_o_c){
        yz_rect_collider = yzr;
        type_of_collider = t_o_c;
    }

    __device__ colliders(constant_medium *c_m,int t_o_c){
        constant_medium_collider = c_m;
        type_of_collider = t_o_c;
    }

    __device__ colliders(moving_sphere *m_s,int t_o_c){
        moving_sphere_collider = m_s;
        type_of_collider = t_o_c;
    }

    __device__ colliders(box *b_c,int t_o_c){
        box_collider = b_c;
        type_of_collider = t_o_c;
    }

    __device__ colliders(rotate_y *r_y,int t_o_c){
        rotate_y_collder = r_y;
        type_of_collider = t_o_c;
    }
    
    __device__ bool hit(const ray &r,
    float t_min,float t_max,collider_record &rec){
        if(type_of_collider==triangle_type_index)
        {
            return triangle_collider->hit(r,t_min,t_max,rec);
        }
        else if(type_of_collider==sphere_type_index)
        {
            return sphere_collider->hit(r,t_min,t_max,rec);
        }
        else if(type_of_collider==xy_rect_type_index)
        {
            return xy_rect_collider->hit(r,t_min,t_max,rec);
        }
        else if(type_of_collider==xz_rect_type_index)
        {
            return xz_rect_collider->hit(r,t_min,t_max,rec);
        }
        else if(type_of_collider==yz_rect_type_index)
        {
            return yz_rect_collider->hit(r,t_min,t_max,rec);
        }
        else if(type_of_collider==constant_medium_index)
        {
            return constant_medium_collider->hit(r,t_min,t_max,rec);
        }
        else if(type_of_collider==moving_sphere_index)
        {
            return moving_sphere_collider->hit(r,t_min,t_max,rec);
        }
        else if(type_of_collider==box_index)
        {
            return box_collider->hit(r,t_min,t_max,rec);
        }
        else if(type_of_collider==rotate_y_index)
        {
            return rotate_y_collder->hit(r,t_min,t_max,rec);
        }
        
    }

    __device__ __host__ bool bounding_box(float time0,
    float time1,aabb &output_box,int index){
        if(type_of_collider==triangle_type_index)
        {
            return triangle_collider->bounding_box(time0,
            time1,output_box,index);
        }
        else if(type_of_collider==sphere_type_index)
        {
            return sphere_collider->bounding_box(time0,
            time1,output_box,index);
        }
        else if(type_of_collider==xy_rect_type_index)
        {
            return xy_rect_collider->bounding_box(time0,
            time1,output_box,index);
        }
        else if(type_of_collider==xz_rect_type_index)
        {
            return xz_rect_collider->bounding_box(time0,
            time1,output_box,index);
        }
        else if(type_of_collider==yz_rect_type_index)
        {
            return yz_rect_collider->bounding_box(time0,
            time1,output_box,index);
        }
        else if(type_of_collider==constant_medium_index)
        {
            return constant_medium_collider->bounding_box(time0,
            time1,output_box,index);
        }
        else if(type_of_collider==moving_sphere_index)
        {
            return moving_sphere_collider->bounding_box(time0,
            time1,output_box,index);
        }
        else if(type_of_collider==box_index)
        {
            return box_collider->bounding_box(time0,
            time1,output_box,index);
        }
        else if(type_of_collider==rotate_y_index)
        {
            return rotate_y_collder->bounding_box(time0,
            time1,output_box,index);
        }
    }


};




#endif