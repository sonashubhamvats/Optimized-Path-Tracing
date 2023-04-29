#ifndef SPHEREH
#define SPHEREH

#include "collider_record.cuh"
#include "aabb.cuh"

class sphere{
    public:
        __device__ sphere(){}
        __device__ sphere(vec3 cen, float r,int index): center(cen),
        radius(r), index_on_collider_list(index){};
        __device__ bool hit(const ray &r,float tmin,
        float tmax,collider_record &rec)const;
        __device__ __host__ bool bounding_box(float time0,float time1,aabb &output_box,int index);

        vec3 center;
        float radius;
        int index_on_collider_list;
    private:
        __device__ static void get_sphere_uv(const vec3 &p,float &u,float &v){
            float theta = acosf(-p.y());
            float phi = atan2f(-p.z(),p.x())+ pi;

            u = phi / (2*pi);
            v = theta / pi;
        }
};

__device__ bool sphere::hit(const ray &r,float tmin, float tmax,
collider_record &rec)const{
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(),r.direction());
    float b = dot(oc,r.direction());
    float c = dot(oc,oc) - radius*radius;
    float discriminant = b*b - a*c;
    float temp = (-b-sqrt(discriminant))/a;
    if(discriminant<0)
        return false;
    if(temp>tmax||temp<tmin){
        temp = (-b+sqrt(discriminant))/a;
        if(temp>tmax||temp<tmin){
            return false;
        }
    }
    
    rec.t = temp;
    rec.p = r.point_at_parameter(rec.t);
    vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r,outward_normal);
    get_sphere_uv(outward_normal,rec.u,rec.v);
    rec.index_on_the_collider_list = index_on_collider_list;
    return true;

}

__device__ __host__ bool sphere::bounding_box(float time0,float time1,aabb &output_box,int index){
    output_box = aabb(
        center - vec3(radius,radius,radius),
        center + vec3(radius,radius,radius),index
    );
    output_box.set_centroid(this->center);
    return true;
}


#endif