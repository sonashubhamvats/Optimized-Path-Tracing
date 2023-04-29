#ifndef AARECT_H
#define AARECT_H

#include "constants.cuh"
#include "collider_record.cuh"


class xy_rect{
    public:
        __device__ xy_rect(){}

        __device__ xy_rect(float _x0,float _x1,
        float _y0,float _y1,float _k,int i_o_c_l)
        :x0(_x0),x1(_x1),y0(_y0),y1(_y1),k(_k),index_on_collider_list(i_o_c_l){};

        __device__ bool hit(const ray &r,float t_min,
        float t_max,collider_record &rec) const;

        __device__ __host__ vec3 get_centroid()
        {
            return vec3((x0+x1)/2.0f,(y0+y1)/2.0f,k);
        }

        __device__ __host__  bool bounding_box(float time0,float time1, 
        aabb &output_box,int index){
            output_box = aabb(vec3(x0,y0,k-0.0001),
            vec3(x1,y1,k+0.0001),index);
            output_box.set_centroid(this->get_centroid());
            return true;
        }


        int index_on_collider_list;
        float x0,x1,y0,y1,k;
};

__device__ bool xy_rect::hit(const ray &r,float t_min,
float t_max,collider_record &rec) const {
    float t = (k-r.origin().z())/r.direction().z();

    if(t<t_min||t>t_max)
        return false;

    float x = r.origin().x() + t*r.direction().x();
    float y = r.origin().y() + t*r.direction().y();

    if(x<x0||x>x1||y<y0||y>y1)
        return false;
    
    rec.u = (x-x0)/(x1-x0);
    rec.v = (y-y0)/(y1-y0);

    rec.t = t;

    vec3 outward_normal= vec3(0,0,1);
    rec.set_face_normal(r,outward_normal);
    rec.index_on_the_collider_list = index_on_collider_list;
    rec.p = r.point_at_parameter(t);
    return true;
} 

class xz_rect{
    public:
        __device__ xz_rect(){}

        __device__ xz_rect(float _x0,float _x1,
        float _z0,float _z1,float _k,int i_o_c_l)
        :x0(_x0),x1(_x1),z0(_z0),z1(_z1),k(_k),index_on_collider_list(i_o_c_l){};

        __device__ virtual bool hit(const ray &r,float t_min,
        float t_max,collider_record &rec) const;

        __device__ __host__ bool bounding_box(float time0,float time1, 
        aabb &output_box,int index)const{
            output_box = aabb(vec3(x0,k-0.0001,z0),
            vec3(x1,k+0.0001,z1),index);
            return true;
        }


        int index_on_collider_list;
        float x0,x1,z0,z1,k;
};

class yz_rect{
    public:
        __device__ yz_rect(){}

        __device__ yz_rect(float _y0,float _y1,
        float _z0,float _z1,float _k,int i_o_c_l)
        :y0(_y0),y1(_y1),z0(_z0),z1(_z1),k(_k),index_on_collider_list(i_o_c_l){};

        __device__ virtual bool hit(const ray &r,float t_min,
        float t_max,collider_record &rec) const;

        __device__ __host__ bool bounding_box(float time0,float time1, 
        aabb &output_box,int index)const{
            output_box = aabb(vec3(k-0.0001,y0,z0),
            vec3(k+0.0001,y1,z1),index);
            return true;
        }


        int index_on_collider_list;
        float y0,y1,z0,z1,k;
};


__device__ bool xz_rect::hit(const ray &r,float t_min,
float t_max,collider_record &rec) const {
    float t = (k-r.origin().y())/r.direction().y();

    if(t<t_min||t>t_max)
        return false;

    float x = r.origin().x() + t*r.direction().x();
    float z = r.origin().z() + t*r.direction().z();

    if(x<x0||x>x1||z<z0||z>z1)
        return false;
    
    rec.u = (x-x0)/(x1-x0);
    rec.v = (z-z0)/(z1-z0);

    rec.t = t;

    vec3 outward_normal= vec3(0,1,0);
    rec.set_face_normal(r,outward_normal);
    rec.index_on_the_collider_list = index_on_collider_list;
    rec.p = r.point_at_parameter(t);
    return true;
} 

__device__ bool yz_rect::hit(const ray &r,float t_min,
float t_max,collider_record &rec) const {
    float t = (k-r.origin().x())/r.direction().x();

    if(t<t_min||t>t_max)
        return false;

    float z = r.origin().z() + t*r.direction().z();
    float y = r.origin().y() + t*r.direction().y();

    if(z<z0||z>z1||y<y0||y>y1)
        return false;
    
    rec.v = (z-z0)/(z1-z0);
    rec.u = (y-y0)/(y1-y0);

    rec.t = t;

    vec3 outward_normal= vec3(1,0,0);
    rec.set_face_normal(r,outward_normal);
    rec.index_on_the_collider_list = index_on_collider_list;
    rec.p = r.point_at_parameter(t);
    return true;
} 


#endif