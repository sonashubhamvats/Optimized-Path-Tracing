#ifndef MATERIAL_H
#define MATERIAL_H

#include "constants.cuh"
#include "texture.cuh"


__device__ vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2*dot(v,n)*n;
}

__device__ vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat) {
    auto cos_theta = fminf(dot(-uv, n), 1.0);
    vec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
    vec3 r_out_parallel = -sqrt(fabsf(1.0 - r_out_perp.squared_length())) * n;
    return r_out_perp + r_out_parallel;
}

__device__ float reflectance(float cosine, float ref_idx){
    // Use Schlick's approximation for reflectance.
    float r0 = (1.0f-ref_idx) / (1.0f+ref_idx);
    r0 = r0*r0;
    return r0 + (1.0f-r0)*powf((1.0f - cosine),5.0f);
}

class diffuse_light{
    public:
        __device__ diffuse_light(vec3 a): light_value(a){}

        __device__ bool scatter(
            const ray &r_int,const collider_record &rec,
            ray &scattered, curandState *local_rand_state
        )const
        {
            return false;
        }

        __device__ vec3 emitted() const{
            return light_value;
        }
        vec3 light_value;
};

class metal{
    public:
        __device__ metal(float f) :fuzz(f<1.0f?f:1.0f){};
        __device__ bool scatter(
            const ray &r_in,const collider_record &rec,
            ray &scattered, curandState *local_rand_state
        )const{
            vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
            scattered = ray(rec.p, reflected+fuzz*random_in_unit_sphere(local_rand_state)
            ,r_in.time());

            return (dot(scattered.direction(), rec.normal) > 0);
        }
        __device__ vec3 emitted() const{
            return vec3(0,0,0);
        }
        float fuzz;
};

class lambertian{
    public:
        __device__ lambertian(){}

        __device__ bool scatter(
            const ray &r_in,const collider_record &rec,
            ray &scattered, curandState *local_rand_state
        )const{
            vec3 scatter_direction = rec.normal 
            + random_in_unit_sphere(local_rand_state);
            if (scatter_direction.near_zero())
                scatter_direction = rec.normal;
            scattered = ray(rec.p, scatter_direction,r_in.time());
            
            return true;
        }
        __device__ vec3 emitted() const{
            return vec3(0,0,0);
        }
};

class dielectric{
    public:
        __device__ dielectric(){}
        __device__ dielectric(float index_of_refraction) : ir(index_of_refraction) {}

        __device__ bool scatter(
            const ray& r_in, const collider_record& rec,ray& scattered
            , curandState *local_rand_state
        ) const{
            float refraction_ratio = rec.front_face ? (1.0f/ir) : ir;

            vec3 unit_direction = unit_vector(r_in.direction());
            float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0f);
            float sin_theta = sqrt(1.0f - cos_theta*cos_theta);

            bool cannot_refract = refraction_ratio * sin_theta > 1.0;
            vec3 direction;
            if (cannot_refract || reflectance(cos_theta, refraction_ratio) > curand_uniform(local_rand_state))
                direction = reflect(unit_direction, rec.normal);
            else
                direction = refract(unit_direction, rec.normal, refraction_ratio);

            scattered = ray(rec.p, direction,r_in.time());
            return true;
        }

        __device__ vec3 emitted() const{
            return vec3(0,0,0);
        }

    public:
        float ir; // Index of Refraction
};

class isotropic{
    public:
        __device__ isotropic(){}

        __device__ virtual bool scatter(const ray &r_in,const collider_record &rec,
        ray &scattered, curandState *local_rand_state
        )const{
            scattered = ray(rec.p , random_in_unit_sphere(local_rand_state) , r_in.time());
            return true;
        }

        __device__ vec3 emitted() const{
            return vec3(0,0,0);
        }

};
struct collider_material{
    lambertian *lambertian_material;
    metal *metal_material;
    diffuse_light *diffuse_light_material;
    dielectric *dielectric_material;
    isotropic *isotropic_material;
    int type_of_material;

    __device__ collider_material(lambertian *l,int t_o_m){
        type_of_material=t_o_m;
        lambertian_material = l;
        
    }

    __device__ collider_material(metal *m,int t_o_m){
        metal_material = m;
        type_of_material = t_o_m;
    }

    __device__ collider_material(diffuse_light *d_l,int t_o_m){
        diffuse_light_material = d_l;
        type_of_material = t_o_m;
    }

    __device__ collider_material(dielectric *d,int t_o_m){
        dielectric_material = d;
        type_of_material = t_o_m;
    }

    __device__ collider_material(isotropic *i_m,int t_o_m){
        isotropic_material = i_m;
        type_of_material = t_o_m;
    }

    __device__ bool scatter(const ray &r_in,const collider_record &rec,
    ray &scattered, curandState *local_rand_state)
    {
        if(type_of_material==lambertian_type_index)
        {
            return lambertian_material->scatter(r_in,rec,scattered,
            local_rand_state);
        }
        else if(type_of_material==metal_type_index)
        {
            return metal_material->scatter(r_in,rec,scattered,
            local_rand_state);
        }
        else if(type_of_material==diffuse_type_index)
        {
            return diffuse_light_material->scatter(r_in,rec,scattered,
            local_rand_state);
        }
        else if(type_of_material==dielectric_type_index)
        {
            return dielectric_material->scatter(r_in,rec,scattered,local_rand_state);
        }
        else if(type_of_material==isotropic_type_index)
        {
            return isotropic_material->scatter(r_in,rec,scattered,local_rand_state);
        }

    }

    __device__ vec3 emitted(){
        if(type_of_material==lambertian_type_index)
        {
            return lambertian_material->emitted();
        }
        else if(type_of_material==metal_type_index)
        {
            return metal_material->emitted();
        }
        else if(type_of_material==diffuse_type_index)
        {
            return diffuse_light_material->emitted();
        }
        else if(type_of_material==dielectric_type_index)
        {
            return dielectric_material->emitted();
        }
        else if(type_of_material==isotropic_type_index)
        {
            return isotropic_material->emitted();
        }
    }
};











#endif