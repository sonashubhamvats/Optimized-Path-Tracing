#ifndef MOVING_SPHERE_H
#define MOVING_SPHERE_H

#include "constants.cuh"
#include "collider_record.cuh"

class moving_sphere{
    public:
        __device__ moving_sphere() {}
        __device__ moving_sphere(
            vec3 cen0, vec3 cen1, float _time0, float _time1, float r, int i_o_c_l)
            : center0(cen0), center1(cen1), time0(_time0), time1(_time1), radius(r), index_on_collider_list(i_o_c_l)
        {};

        __device__ bool hit(
            const ray& r, float t_min, float t_max, collider_record& rec) const;

        __device__ __host__ bool bounding_box(
            float _time0,float _time1, aabb &output_box,int index
        )const;

        __device__ vec3 center(float time) const;

    public:
        vec3 center0, center1;
        float time0, time1;
        float radius;
        int index_on_collider_list;
};

__device__ __host__ vec3 moving_sphere::center(float time) const {
    return center0 + ((time - time0) / (time1 - time0))*(center1 - center0);
}

__device__ bool moving_sphere::hit(const ray& r, float t_min, float t_max, collider_record& rec) const {
    vec3 oc = r.origin() - center(r.time());
    float a = r.direction().squared_length();
    float half_b = dot(oc, r.direction());
    float c = oc.squared_length() - radius*radius;

    float discriminant = half_b*half_b - a*c;
    if (discriminant < 0) return false;
    float sqrtd = sqrt(discriminant);

    auto root = (-half_b - sqrtd) / a;
    if (root < t_min || t_max < root) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root)
            return false;
    }

    rec.t = root;
    rec.p = r.point_at_parameter(rec.t);
    vec3 outward_normal = (rec.p - center(r.time())) / radius;
    rec.set_face_normal(r, outward_normal);
    rec.index_on_the_collider_list = index_on_collider_list;

    return true;
}

__device__ __host__ bool moving_sphere::bounding_box(float _time0,float _time1,
aabb &output_box,int index)const {
    aabb box0(
        center(_time0) - vec3(radius, radius, radius),
        center(_time0) + vec3(radius, radius, radius),index);
    aabb box1(
        center(_time1) - vec3(radius, radius, radius),
        center(_time1) + vec3(radius, radius, radius),index);
    output_box = surrounding_box(box0, box1,index);
    return true;
}

#endif