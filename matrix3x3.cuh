#ifndef MATRIX_3X3
#define MATRIX_3X3

class matrix_3x3{

    public:
        __device__ matrix_3x3(){}
        __device__ matrix_3x3(float _a1,float _a2,float _a3,
        float _b1,float _b2,float _b3,float _c1,float _c2,float _c3):
        a1(_a1),a2(_a2),a3(_a3),
        b1(_b1),b2(_b2),b3(_b3),
        c1(_c1),c2(_c2),c3(_c3){}

        float a1,a2,a3,b1,b2,b3,c1,c2,c3;

    public:
        __device__ float find_determinant(){
            return 
            a1 * (b2*c3 - c2*b3) 
            -a2 * (b1* c3 - c1 * b3) 
            + a3 * (b1 * c2 - c1 * b2);
        }
};



#endif