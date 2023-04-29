#ifndef AABB_H
#define AABB_H

#include "constants.cuh"
class aabb{
    public:

        __device__ __host__ aabb(){}
        __device__ __host__ aabb(const vec3 &a,const vec3 &b,int ind): 
        minimum(a), maximum(b),index(ind)
        {}

        __device__ __host__  vec3 min() const {return minimum;}
        __device__ __host__  vec3 max() const {return maximum;}

        //for now centroid set for only triangles
        __device__ __host__  void set_centroid(vec3 c) {centroid = c;}
        __device__ __host__ vec3 get_centroid(){return centroid;}


        __device__ __host__ float get_area(){
            vec3 box_extent = maximum - minimum;
            return box_extent.x()*box_extent.y() + box_extent.y()*box_extent.z()
            +box_extent.z()*box_extent.x();
        };

        __device__ bool hit(const ray &r,
        float t_min,float t_max,int &distance) const{
            for(int a = 0;a<3;a++)
            {
                float t0 = fminf((minimum[a]-r.origin()[a])/r.direction()[a]
                                ,(maximum[a]-r.origin()[a])/r.direction()[a]);
                float t1 = fmaxf((minimum[a]-r.origin()[a])/r.direction()[a]
                                ,(maximum[a]-r.origin()[a])/r.direction()[a]);
                t_min = fmaxf(t0,t_min);
                t_max = fminf(t1,t_max);

                if(t_max<=t_min)
                    return false;
            }
            distance = t_min;
            return true;
        }

        vec3 minimum,maximum;
        int index;
        vec3 centroid;

};

__device__ __host__  aabb surrounding_box(aabb box0,aabb box1,int index=-1){
    vec3 small(fminf(box0.min().x(),box1.min().x()),
    fminf(box0.min().y(),box1.min().y()),fminf(box0.min().z(),box1.min().z()));
    vec3 big(fmaxf(box0.max().x(),box1.max().x()),
    fmaxf(box0.max().y(),box1.max().y()),fmaxf(box0.max().z(),box1.max().z()));
    return aabb(small,big,index);
}

__device__ __host__ int return_longest_axis(aabb box)
{
    int longest_axis = 0;

    if((box.max()[1]-box.min()[1])>
    (box.max()[longest_axis]-box.min()[longest_axis]))
        longest_axis = 1;
    
    if((box.max()[2]-box.min()[2])>
    (box.max()[longest_axis]-box.min()[longest_axis]))
        longest_axis = 2;

    return longest_axis;
}
#endif