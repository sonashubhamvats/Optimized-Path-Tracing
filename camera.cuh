#ifndef CAMERA_H
#define CAMERA_H


#include "constants.cuh"



class camera {
    public:
        vec3 origin;
        vec3 lower_left_corner;
        vec3 horizontal;
        vec3 vertical;
        vec3 u, v, w;
        float lens_radius;
        float time0,time1;
    public:
        __device__ camera(
            vec3 lookfrom,
            vec3 lookat,
            vec3   vup,
            float vfov,
            float aspect_ratio,
            float aperture,
            float focus_dist,
            float _time0 = 0.0,
            float _time1 = 0.0
        ) {
            float theta = degrees_to_radians(vfov);
            float h = tan(theta/2.0f);
            float viewport_height = 2.0f * h;
            float viewport_width = aspect_ratio * viewport_height;

            w = unit_vector(lookfrom - lookat);
            u = unit_vector(cross(vup, w));
            v = cross(w, u);

            origin = lookfrom;
            horizontal = focus_dist*viewport_width * u;
            vertical = focus_dist*viewport_height * v;
            lower_left_corner = origin - horizontal/2 - vertical/2 - focus_dist*w;

            lens_radius = aperture / 2;
            time0 = _time0;
            time1 = _time1;
        }

        __host__ void recalculate_look_from(vec3 lookfrom,
            vec3 lookat,
            vec3   vup,
            float vfov,
            float aspect_ratio,
            float focus_dist)
        {
            float theta = degrees_to_radians(vfov);
            float h = tan(theta/2.0f);
            float viewport_height = 2.0f * h;
            float viewport_width = aspect_ratio * viewport_height;

            w = unit_vector(lookfrom - lookat);
            u = unit_vector(cross(vup, w));
            v = cross(w, u);

            origin = lookfrom;
            horizontal = focus_dist * viewport_width * u;
            vertical = focus_dist * viewport_height * v;
            lower_left_corner = origin - horizontal/2 - vertical/2 - focus_dist*w;
        }

        __device__ void get_ray(float s, float t,curandState *rand_state,ray &r) const {
            vec3 rd = lens_radius * random_in_unit_sphere(rand_state);
            vec3 offset = u * rd.x() + v * rd.y();
            r.orig = origin + offset;
            r.dir = lower_left_corner + s*horizontal + t*vertical - origin - offset;
            r.tme = random_double(rand_state,time0,time1);
        }

    
};
#endif