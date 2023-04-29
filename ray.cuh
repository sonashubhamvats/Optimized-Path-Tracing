
#ifndef RAYH
#define RAYH
#include "vec3.cuh"

class ray
{
    public:
        vec3 orig;
        vec3 dir;
        float tme;
        __device__ ray() {}
        __device__ ray(const vec3& a, const vec3& b,
        float tm = 0.0f):
        orig(a), dir(b), tme(tm){} 
        __device__ vec3 origin() const       { return orig; }
        __device__ vec3 direction() const    { return dir; }
        __device__ vec3 point_at_parameter(float t) const { return orig + t*dir; }
        __device__ float time() const{return tme;}
        
};

#endif