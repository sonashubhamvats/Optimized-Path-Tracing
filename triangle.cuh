#ifndef TRAINGLE_H
#define TRIANGLE_H


#include "matrix3x3.cuh"
#include "collider_record.cuh"



class triangle{

    public:
        __device__ triangle(){}
        __device__ triangle(vec3 _a,vec3 _b,vec3 _c,int i_o_c_l)
        :a(_a),b(_b),c(_c),index_on_the_collider_list(i_o_c_l){}

        __device__ bool hit(const ray &r,float tmin,
        float tmax, collider_record &rec)const;

        __device__ __host__ vec3 get_centroid(){
            return vec3(((a.x()+b.x()+c.x())/3.0f),((a.y()+b.y()+c.y())/3.0f),
            ((a.z()+b.z()+c.z())/3.0f));
        }

        __device__ __host__ bool bounding_box(float time0,
        float time1,aabb &output_box,int index);

    public:
        vec3 a,b,c;
        int index_on_the_collider_list;
};

__device__ __host__  bool triangle::bounding_box(float time0,float time1,
aabb &output_box,int index)
{
    vec3 small(fminf(a.x(),fminf(b.x(),c.x()))
    ,fminf(a.y(),fminf(b.y(),c.y())),
    fminf(a.z(),fminf(b.z(),c.z())));

    vec3 big(fmaxf(a.x(),fmaxf(b.x(),c.x()))
    ,fmaxf(a.y(),fmaxf(b.y(),c.y())),
    fmaxf(a.z(),fmaxf(b.z(),c.z())));

    output_box=aabb(small,big,index);

    output_box.set_centroid(this->get_centroid());

    return true;
}


__device__ bool triangle::hit(const ray &r,float tmin,
float tmax,collider_record &rec)const{

    vec3 r_d = r.direction();
    vec3 r_o = r.origin();

    float a1 = a.x() - b.x();float b1 = a.x() - c.x();float c1 = r_d.x();
    float a2 = a.y() - b.y();float b2 = a.y() - c.y();float c2 = r_d.y();
    float a3 = a.z() - b.z();float b3 = a.z() - c.z();float c3 = r_d.z();
    float d1 = a.x() - r_o.x();float d2 = a.y() - r_o.y();float d3 = a.z() - r_o.z();

    

    matrix_3x3 A(
        a1,b1,c1,
        a2,b2,c2,
        a3,b3,c3
    );



    matrix_3x3 Dx(
        d1,b1,c1,
        d2,b2,c2,
        d3,b3,c3
    );

    

    matrix_3x3 Dy(
        a1,d1,c1,
        a2,d2,c2,
        a3,d3,c3
    );

    matrix_3x3 Dz(
        a1,b1,d1,
        a2,b2,d2,
        a3,b3,d3
    );

    

    float D = A.find_determinant();

    if(abs(D)<0.000001f)
        return false;
    
    float beta = Dx.find_determinant()/D;
    float gamma = Dy.find_determinant()/D;
    float temp_t = Dz.find_determinant()/D;

    if(beta<0||gamma<0||beta+gamma>1)
        return false;

    if(temp_t>tmax||temp_t<tmin)
    {
        return false;
    }

    
    rec.u = beta;
    rec.v = gamma;

    rec.t = temp_t;
    rec.p = r.point_at_parameter(rec.t);

    vec3 D_side = b-a;
    vec3 E_side = c-b;

    vec3 surface_normal = cross(D_side,E_side);
    surface_normal.make_unit_vector();
    rec.set_face_normal(r,surface_normal);
    rec.index_on_the_collider_list = index_on_the_collider_list;
    

    return true;
}

#endif