#ifndef COLLIDER_RECORD_H
#define COLLIDER_RECORD_H

#include "constants.cuh"

struct collider_record{
    float t,u,v;
    vec3 p;
    vec3 normal;
    bool front_face;
    int index_on_the_collider_list;
    __device__ inline void set_face_normal(const ray &r,
    const vec3& outward_normal){
        front_face = dot(r.direction(),outward_normal)<0;
        normal = front_face?outward_normal:-outward_normal;
    }
};

#endif