#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <limits>
#include <memory>
// Common Headers

#include "ray.cuh"
#include "vec3.cuh"

#include <curand_kernel.h>

#define RANDVEC3 vec3(curand_uniform(temp_state),curand_uniform(temp_state),curand_uniform(temp_state))

__device__ vec3 random_in_unit_sphere(curandState *temp_state)
{
    vec3 p;
    do {
        p = 2.0f* RANDVEC3 - vec3(1,1,1);
    } while (p.squared_length() >= 1.0f);
    return p;
}

__device__ vec3 random_vector_in_range(curandState *temp_state,
float t1,float t2){
    return vec3(t1 + curand_uniform(temp_state)*(t2-t1),
    t1 + curand_uniform(temp_state)*(t2-t1),t1 + curand_uniform(temp_state)*(t2-t1));
}

__device__ float random_double(curandState *temp_state,
float time0,float time1){
    return time0 + curand_uniform(temp_state)*(time1-time0);
}

__device__ int random_int(curandState *temp_state,float mn,
float mx)
{
    return (int)(mn + curand_uniform(temp_state)*(mx-mn));
}

#define RND curand_uniform(&localState)
// Usings

using std::shared_ptr;
using std::make_shared;
using std::sqrt;

// Constants

__device__ const float infinity = INFINITY;
__device__ const float pi = 3.1415926535897932385;

// Utility Functions

__device__ __host__ inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0;
}

#define triangle_type_index 0
#define sphere_type_index 1
#define xy_rect_type_index 2
#define xz_rect_type_index 3
#define yz_rect_type_index 4
#define constant_medium_index 5
#define rotate_y_index 6
#define moving_sphere_index 7
#define box_index 8

#define lambertian_type_index 0
#define metal_type_index 1
#define diffuse_type_index 2
#define dielectric_type_index 3
#define isotropic_type_index 4

#define solid_color_type_index 0
#define noise_texture_type_index 1
#define checker_texture_type_index 2
#define image_texture_type_index 3


//camera config for the plane
// vec3 lookfrom(8.0f, 3.0f, -20.0f);
//     vec3 lookat(0,0,0);
//     float dist_to_focus = (lookfrom-lookat).length();
//     float aperture = 0.0;
//     *d_camera   = new camera(lookfrom,
//                     lookat,
//                     vec3(0,1,0),
//                     40.0,
//                     float(nx)/float(ny),
//                     aperture,
//                     dist_to_focus,0.0,1.0);


//camera config for budhha
// vec3 lookfrom(0.0f, 0.0f, -1.0f);
//     vec3 lookat(0,0,0);
//     float dist_to_focus = (lookfrom-lookat).length();
//     float aperture = 0.0;
//     *d_camera   = new camera(lookfrom,
//                     lookat,
//                     vec3(0,1,0),
//                     35.0,
//                     float(nx)/float(ny),
//                     aperture,
//                     dist_to_focus,0.0,1.0);
//budhha color vec3(0.721f,0.525f,0.043f)

//camera config for machine
// vec3 lookfrom(-5.0f, 0.0f, 2.0f);
//     vec3 lookat(0,0,0);
//     float dist_to_focus = (lookfrom-lookat).length();
//     float aperture = 0.0;
//     *d_camera   = new camera(lookfrom,
//                     lookat,
//                     vec3(0,1,0),
//                     35.0,
//                     float(nx)/float(ny),
//                     aperture,
//                     dist_to_focus,0.0,1.0);

#endif