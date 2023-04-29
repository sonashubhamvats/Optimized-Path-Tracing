#ifndef RCY_H
#define RCY_H

#include "constants.cuh"
#include "collider_record.cuh"
#include "aarect.cuh"
#include "aabb.cuh"
#include "box.cuh"

class rotate_y{
    public:
        xy_rect *xy_rect_collider;
        yz_rect *yz_rect_collider;
        xz_rect *xz_rect_collider;
        box *b_collider;
        int type_collider_here;
        int index_on_the_collider_list;
        float sin_theta;
        float cos_theta;
        bool hasbox;
        aabb bbox;
        
        __device__ rotate_y(xy_rect *p, float angle,int index):
        xy_rect_collider(p),type_collider_here(xy_rect_type_index),index_on_the_collider_list(index){
            float radians = degrees_to_radians(angle);
            sin_theta = sinf(radians);
            cos_theta = cosf(radians);
            hasbox = xy_rect_collider->bounding_box(0, 1, bbox,index);

            vec3 min( infinity,  infinity,  infinity);
            vec3 max(-infinity, -infinity, -infinity);

            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    for (int k = 0; k < 2; k++) {
                        float x = i*bbox.max().x() + (1-i)*bbox.min().x();
                        float y = j*bbox.max().y() + (1-j)*bbox.min().y();
                        float z = k*bbox.max().z() + (1-k)*bbox.min().z();

                        float newx =  cos_theta*x + sin_theta*z;
                        float newz = -sin_theta*x + cos_theta*z;

                        vec3 tester(newx, y, newz);

                        for (int c = 0; c < 3; c++) {
                            min[c] = fmin(min[c], tester[c]);
                            max[c] = fmax(max[c], tester[c]);
                        }
                    }
                }
            }

            bbox = aabb(min, max,index);
        }

        __device__ rotate_y(xz_rect *p, float angle,int index):
        xz_rect_collider(p),type_collider_here(xz_rect_type_index),index_on_the_collider_list(index){
            float radians = degrees_to_radians(angle);
            sin_theta = sinf(radians);
            cos_theta = cosf(radians);
            hasbox = xz_rect_collider->bounding_box(0, 1, bbox,index);

            vec3 min( infinity,  infinity,  infinity);
            vec3 max(-infinity, -infinity, -infinity);

            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    for (int k = 0; k < 2; k++) {
                        float x = i*bbox.max().x() + (1-i)*bbox.min().x();
                        float y = j*bbox.max().y() + (1-j)*bbox.min().y();
                        float z = k*bbox.max().z() + (1-k)*bbox.min().z();

                        float newx =  cos_theta*x + sin_theta*z;
                        float newz = -sin_theta*x + cos_theta*z;

                        vec3 tester(newx, y, newz);

                        for (int c = 0; c < 3; c++) {
                            min[c] = fmin(min[c], tester[c]);
                            max[c] = fmax(max[c], tester[c]);
                        }
                    }
                }
            }

            bbox = aabb(min, max,index);
        }

        __device__ rotate_y(yz_rect *p, float angle,int index):
        yz_rect_collider(p),type_collider_here(yz_rect_type_index),index_on_the_collider_list(index){
            float radians = degrees_to_radians(angle);
            sin_theta = sinf(radians);
            cos_theta = cosf(radians);
            hasbox = yz_rect_collider->bounding_box(0, 1, bbox,index);

            vec3 min( infinity,  infinity,  infinity);
            vec3 max(-infinity, -infinity, -infinity);

            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    for (int k = 0; k < 2; k++) {
                        float x = i*bbox.max().x() + (1-i)*bbox.min().x();
                        float y = j*bbox.max().y() + (1-j)*bbox.min().y();
                        float z = k*bbox.max().z() + (1-k)*bbox.min().z();

                        float newx =  cos_theta*x + sin_theta*z;
                        float newz = -sin_theta*x + cos_theta*z;

                        vec3 tester(newx, y, newz);

                        for (int c = 0; c < 3; c++) {
                            min[c] = fmin(min[c], tester[c]);
                            max[c] = fmax(max[c], tester[c]);
                        }
                    }
                }
            }

            bbox = aabb(min, max,index);
        }
        
         __device__ rotate_y(box *p, float angle,int index):
        b_collider(p),type_collider_here(box_index),index_on_the_collider_list(index){
            float radians = degrees_to_radians(angle);
            sin_theta = sinf(radians);
            cos_theta = cosf(radians);
            hasbox = b_collider->bounding_box(0, 1, bbox,index);

            vec3 min( infinity,  infinity,  infinity);
            vec3 max(-infinity, -infinity, -infinity);

            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    for (int k = 0; k < 2; k++) {
                        float x = i*bbox.max().x() + (1-i)*bbox.min().x();
                        float y = j*bbox.max().y() + (1-j)*bbox.min().y();
                        float z = k*bbox.max().z() + (1-k)*bbox.min().z();

                        float newx =  cos_theta*x + sin_theta*z;
                        float newz = -sin_theta*x + cos_theta*z;

                        vec3 tester(newx, y, newz);

                        for (int c = 0; c < 3; c++) {
                            min[c] = fmin(min[c], tester[c]);
                            max[c] = fmax(max[c], tester[c]);
                        }
                    }
                }
            }

            bbox = aabb(min, max,index);
        }
        
        __device__ bool hit(
            const ray& r, float t_min, float t_max, collider_record& rec) const
        {
            vec3 origin = r.origin();
            vec3 direction = r.direction();

            origin[0] = cos_theta*r.origin()[0] - sin_theta*r.origin()[2];
            origin[2] = sin_theta*r.origin()[0] + cos_theta*r.origin()[2];

            direction[0] = cos_theta*r.direction()[0] - sin_theta*r.direction()[2];
            direction[2] = sin_theta*r.direction()[0] + cos_theta*r.direction()[2];

            ray rotated_r(origin, direction, r.time());

            if(type_collider_here==xy_rect_type_index)
            {
                if (!xy_rect_collider->hit(rotated_r, t_min, t_max, rec))
                    return false;
            }
            else if(type_collider_here==xz_rect_type_index)
            {
                if (!xz_rect_collider->hit(rotated_r, t_min, t_max, rec))
                    return false;
            }
            else if(type_collider_here==yz_rect_type_index)
            {
                if (!yz_rect_collider->hit(rotated_r, t_min, t_max, rec))
                    return false;
            }
            else if(type_collider_here==box_index)
            {
                if (!b_collider->hit(rotated_r, t_min, t_max, rec))
                    return false;
            }
            

            auto p = rec.p;
            auto normal = rec.normal;

            p[0] =  cos_theta*rec.p[0] + sin_theta*rec.p[2];
            p[2] = -sin_theta*rec.p[0] + cos_theta*rec.p[2];

            normal[0] =  cos_theta*rec.normal[0] + sin_theta*rec.normal[2];
            normal[2] = -sin_theta*rec.normal[0] + cos_theta*rec.normal[2];

            rec.p = p;
            rec.set_face_normal(rotated_r, normal);

            return true;
        }

        __device__ __host__ bool bounding_box(float time0, float time1, aabb& output_box,int index) const{
            output_box = bbox;
            return hasbox;
        }

};
#endif