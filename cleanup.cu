
// class __align__(16) vec3{
//     int x,y,z;

//     public:
//         __device__ vec3(int a,int b,int c):
//         x(a),y(b),z(c){};
// };



// class __align__(16) material{
//     public:
//         __device__ virtual bool scatter(vec3 &attentuation)const = 0;

// };

// struct __align__(16) record{
//     material *ma;
// };

// class __align__(16) hittable{
//     public:
//         __device__ virtual bool hit(curandState *local_state,record &rec)const=0;
// };

// class __align__(16) textures{
//     public:
//     __device__ virtual vec3 value() const = 0;
// };

// class __align__(16) solid_color:public textures{
//     vec3 color_value;

//     public:

//         __device__ solid_color(vec3 c):
//         color_value(c){}
        

//         __device__ virtual vec3 value()const override{
//             return color_value;
//         }


// };

// class __align__(16) lambertian : public material{
//     textures *a;

//     public:
//         __device__ lambertian(vec3 tex):a(new solid_color(tex)){}

//         __device__ virtual bool scatter(vec3 &attentuation)const override{
//             attentuation = a->value();
//         }

    
// };


// class __align__(16) triangle:public hittable{
//     public:
//     vec3 point_x,point_y,point_z;
//     material *ma;
//         __device__ triangle(vec3 p_x,vec3 p_y,vec3 p_z,material *m):
//         point_x(p_x),point_y(p_y),point_z(p_z),ma(m){}
//         __device__ virtual bool hit(curandState *local_state,record &rec)const override{
//             rec.ma = ma;
//             return true;
//         }
// };

// class __align__(16) collection_of_triangles:public hittable{
//     hittable **list;
//     int size;

//     public:
//         __device__ collection_of_triangles(hittable **l,int s):
//         list(l),size(s){}

//         __device__ virtual bool hit(curandState *local_state,record &r)const override{
//             int index = curand_uniform(local_state)*size;
//             int ans=1;
//             list[index]->hit(local_state,r);
//             return true;
//         }
// };


// __global__ void init_render(hittable **list,
// hittable **c_o_t,int size)
// {
//     for(int i=0;i<size;i++)
//     {
//         list[i] = new triangle(vec3(1,1,1),vec3(1,1,1),vec3(1,1,1),
//         new lambertian(vec3(1,1,1)));
//     }

//     *c_o_t = new collection_of_triangles(list,size);
// }

// __global__ void render_hit(hittable **c_o_t)
// {
//     int total_no_threads_in_a_block = blockDim.x*blockDim.y;
//     int total_no_threads_in_a_row = total_no_threads_in_a_block*gridDim.x;
//     int pixel_index = threadIdx.x + threadIdx.y*blockDim.x + total_no_threads_in_a_block*blockIdx.x+
//     total_no_threads_in_a_row*blockIdx.y;
//     curandState local_state;
//     curand_init(1984+pixel_index, 0, 0, &local_state);
//     vec3 ans(0,0,0);
//     for(int i=0;i<51;i++)
//     {
//         record rec;
//         if((*c_o_t)->hit(&local_state,rec))
//         {
//             rec.ma->scatter(ans);
//         }
//     }

// }

// int main(){
//     hittable **triangle_list;
//     hittable **collection;
//     int size = 500;
//     cudaMalloc(&triangle_list,sizeof(hittable)*size);
//     cudaMalloc(&collection,sizeof(hittable));
//     init_render<<<1,1>>>(triangle_list,collection,size);
//     cudaDeviceSynchronize();
//     render_hit<<<dim3(512,512),dim3(12,12)>>>(collection);
// }

// class parent_int{
//     public:
//         __device__ __host__ virtual parent_int* Clone()=0;
//         __device__ __host__ virtual int get_int() const=0;
//         __device__ __host__ virtual void increment_int()=0;
// };
// class a_int: public parent_int{
//     public:
//         int a;
//         __device__ __host__ a_int(int _a=0): a(_a){}
//         __device__ __host__ virtual parent_int* Clone() override{
//             return new a_int(*this);
//         }
//         __device__ __host__ virtual int get_int()const override{return a;}
//         __device__ __host__ virtual void increment_int()override{a+=1;}
// };
// class b_int:public parent_int{
//     public:
        
//         int b;
//         __device__ __host__ b_int(int _b=0): b(_b){}
//         __device__ __host__ virtual parent_int* Clone() override{
//             return new b_int(*this);
//         }
//         __device__ __host__ virtual int get_int() const override{return b;}
//         __device__ __host__ virtual void increment_int() override{b+=2;}
// };

// class c_int : public parent_int{
//     public:
//         parent_int *left,*right;
//         __device__ c_int(parent_int **list ,int height)
//         {
//             printf("\nHere for height %d",height);
//             if(height==0)
//             {
//                 left = new a_int(10);
//                 right = new b_int(10);
//             }
//             else
//             {
//                 left = new c_int(list,height-1);
//                 right = new c_int(list,height-1);
//             }
//         }
//         __device__ __host__ virtual parent_int* Clone() override{
//             return new c_int(*this);
//         }
//         __device__ __host__ virtual int get_int() const override{return 0;}
//         __device__ __host__ virtual void increment_int() override{return;}
// };

// __global__ void stack_limit(curandState *rand_state){
//     if(threadIdx.x==0&&blockIdx.x==0)
//     {
//         curand_init(1984,0,0,rand_state);
//         parent_int **list = new parent_int*[500];
//         for(int i=0;i<500;i++)
//         {
//             if(curand_uniform(rand_state)<=0.5f)
//                 list[i] = new a_int(10);
//             else
//                 list[i] = new b_int(10);
//         }
//         c_int *c = new c_int(list,100);
//     }
// }

// __global__ void test(){
//     matrix_3x3 a(
//         1,2,3,
//         4,5,6,
//         6,7,9
//     );

//     printf("\n%d",a.find_determinant());
// }   

// int main(){
//     size_t limit = 0;

//     cudaDeviceGetLimit(&limit, cudaLimitStackSize);
//     printf("cudaLimitStackSize: %u\n", (unsigned)limit);
//     cudaDeviceGetLimit(&limit, cudaLimitPrintfFifoSize);
//     printf("cudaLimitPrintfFifoSize: %u\n", (unsigned)limit);
//     cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
//     printf("cudaLimitMallocHeapSize: %u\n", (unsigned)limit);
//     //stack_limit<<<1,1>>>(rand_state);

//     test<<<1,1>>>();
// }

// __device__ bool comparator(parent_int *a,parent_int *b)
// {
//     return a->get_int()<b->get_int();
// }
// __global__ void print_values(parent_int **d_a,int size,curandState *rand_state)
// {
//     if(threadIdx.x==0&&blockIdx.x==0)
//     {
//         curand_init(1984,0,0,rand_state);
//         for(int i=0;i<size;i++)
//         {
//             if(curand_uniform(rand_state)<=0.5f)
//             {
//                 d_a[i] = new a_int(curand_uniform(rand_state)*10);
//             }
//             else
//             {
//                 d_a[i] = new b_int(curand_uniform(rand_state)*10);
//             }
//         }
//         printf("\nOriginal array - \n");
//         for(int i=0;i<size;i++)
//         {
//             printf("%d ", d_a[i]->get_int());
//         }

        
//         thrust::sort(d_a,d_a+size,comparator);
//         printf("\n");
//         printf("\nSorted array - \n");
//         for(int i=0;i<size;i++)
//         {
//             printf("%d ", d_a[i]->get_int());
//         }
//     }
// }

// int main(){
    
//     parent_int **d_a;
//     int size = 10;
//     curandState *rand_state;
//     gpuErrchk(cudaMalloc(&rand_state,sizeof(rand_state)));
//     gpuErrchk(cudaMalloc(&d_a,sizeof(parent_int*)*size));
    
//     print_values<<<1,1>>>(d_a,size,rand_state);
//     gpuErrchk(cudaDeviceSynchronize());

// }


// class textures{
//     public:
//         __device__ virtual int value(int random_info)const = 0;
//         __device__ virtual textures* Clone()=0;
//         __device__ __host__ virtual void print_info()const = 0;
// };

// class texture1:public textures{
// public:
//     __device__ texture1(int inf):info(inf){}
//     __device__ virtual textures* Clone()override{
//         return new texture1(*this);
//     }
//     __device__ virtual int value(int random_info)const{
//         int _info_=random_info;
//         return _info_;
//     } 

//     __device__ __host__ virtual void print_info()const override{
//         printf("\nI am a texture1");
//     }

//     int info;
// };

// class texture2:public textures{
// public:
//     __device__ texture2(int inf):info(inf){}
//     __device__ virtual textures* Clone()override{
//         return new texture2(*this);
//     }
//     __device__ virtual int value(int random_info)const{
//         int _info_=random_info;
//         return _info_+1;
//     } 

//     __device__ __host__ virtual void print_info()const override{
//         printf("\nI am a texture2");
//     }

//     int info;
// };

// class material{
// public:
//     __device__ virtual bool scatter(int random_info)const = 0;
//     __device__ virtual material* Clone()=0;
//     __device__ __host__ virtual void print_info()const = 0;
// };


// class glass: public material{
// public:
//     __device__ virtual material* Clone()override{
//         return new glass(*this);
//     }
//     __device__ glass(int i_r): ir(i_r){}
//     __device__ virtual bool scatter(int random_info)const override{
//         return true;
//     }
//     __device__ __host__ virtual void print_info()const override{
//         printf("\nI am a glass");
//     }

//     int ir;
// };

// class metal: public material{
// public:
//     __device__ virtual material* Clone()override{
//         return new metal(*this);
//     }
//     __device__ metal(){}
//     __device__ virtual bool scatter(int random_info)const override{
//         return true;
//     }

//     __device__ __host__ virtual void print_info()const override{
//         printf("\nI am a metal");

//     }

// };


// class shapes{
// public:
//     __device__ virtual bool hit(int random_info)const = 0;
//     __device__ virtual shapes* Clone()=0;

//     __device__ __host__ virtual void print_info()const = 0;
// };

// class triangle:public shapes{
// public:
//     int pointA,pointB,pointC;


//     __device__ triangle(int p_a,int p_b,int p_c):
//     pointA(p_a),pointB(p_b),pointC(p_c){}

//     __device__ virtual bool hit(int random_info)const override{
//         return true;
//     }

//     __device__ virtual shapes* Clone()override{
//         return new triangle(*this);
//     }

//     __device__ __host__ virtual void print_info()const override{
//         printf("\nI am a triangle");

//     }

// };

// class square:public shapes{
// public:
//     int pointA,pointB;


//     __device__ square(int p_a,int p_b):
//     pointA(p_a),pointB(p_b){}

//     __device__ virtual bool hit(int random_info)const override{
//         return true;
//     }

//     __device__ virtual shapes* Clone()override{
//         return new square(*this);
//     }

//     __device__ __host__ virtual void print_info()const override{
//         printf("\nI am a square");

//     }

// };

// __global__ void device_init(shapes **s_list,textures **t_list,
// material **m_list,int size){

//     curandState localState;

//     curand_init(1984,0,0,&localState);

//     for(int i=0;i<size;i++)
//     {
//         float rand_no = curand_uniform(&localState);

//         if(rand_no<=0.5f)
//         {
//             t_list[i] = new texture1(3);
//         }
//         else
//         {
//             t_list[i] = new texture2(4);
//         }

//         rand_no = curand_uniform(&localState);
//         if(rand_no<=0.5f)
//         {
//             m_list[i] = new glass(5);
//         }
//         else
//         {
//             m_list[i] = new metal();
//         }

//         rand_no = curand_uniform(&localState);
//         if(rand_no<=0.5f)
//         {
//             s_list[i] = new triangle(2,3,4);
//         }
//         else
//         {
//             s_list[i] = new square(2,3);
//         }
//     }
// }

// __global__ void device_show(shapes **s_list,material **m_list,
// textures **t_list,int size){
//     printf("\n\nFor the device");
//     for(int i=0;i<size;i++)
//     {
//         s_list[i]->print_info();
//         m_list[i]->print_info();
//         t_list[i]->print_info();
//         printf("\n");
//     }

// }

// void host_show(shapes **s_list,material **m_list,
// textures **t_list,int size)
// {
//     printf("\n\nFor the host");
//     for(int i=0;i<size;i++)
//     {
//         s_list[i]->print_info();
//         m_list[i]->print_info();
//         t_list[i]->print_info();
//         printf("\n");
//     }
// }

// #include "cuda_common.cuh"
// #include "constants.cuh"
// #include <float.h>
// #include <time.h>
// #include "camera.cuh"
// #include "texture.cuh"
// #include "sphere.cuh"
// #include "collider_list.cuh"
// #include "materials.cuh"
// #include "moving_spheres.cuh"
// #include "rt_stb_image.h"
// #include "bvh_cpu.cuh"
// #include "bvh_gpu.cuh"
// #include "aarect.cuh"
// #include "box.cuh"
// #include "constant_medium.cuh"

// __global__ void object_init(colliders *d_list,collider_list **d_world
// ,aabb *list_of_bounding_boxes,int size)
// {
//     curandState localState;
//     curand_init(1984,0,0,&localState);

//     xy_rect *xyr = new xy_rect(-10,-5,0,10,0,0);
//     d_list[0] = colliders(xyr,xy_rect_type_index);
//     d_list[0].bounding_box(0,0,(list_of_bounding_boxes[0]),0);

//     xy_rect *xyr_1 =new xy_rect(5,10,0,10,0,1);
//     d_list[1] = colliders(xyr_1,xy_rect_type_index);
//     d_list[1].bounding_box(0,0,(list_of_bounding_boxes[1]),1);

//     sphere *s = new sphere(vec3(-2.0f,0.0f,0.0f),2.0f,2);
//     d_list[2] = colliders(s,sphere_type_index);
//     d_list[2].bounding_box(0,0,(list_of_bounding_boxes[2]),2);

//     sphere *s_1 = new sphere(vec3(2.0f,0.0f,0.0f),2.0f,3);
//     d_list[3] = colliders(s_1,sphere_type_index);
//     d_list[3].bounding_box(0,0,(list_of_bounding_boxes[3]),3);

//     *d_world = new collider_list(d_list,size);
// }

// // void traverse_bvh_node(bvh_cpu *parent)
// // {
// //     if(parent==NULL)
// //         return;
// //     printf("\nFor index- %d",parent->index);
// //     printf("\nx_min- %f  y_min- %f  z_min- %f\n",
// //     parent->box.min().x(),
// //     parent->box.min().y(),
// //     parent->box.min().z());

// //     printf("x_max- %f  y_max- %f  z_max- %f\n",
// //     parent->box.max().x(),
// //     parent->box.max().y(),
// //     parent->box.max().x());

// //     traverse_bvh_node(parent->left);
// //     traverse_bvh_node(parent->right);
// // }

// __device__ void bvh_tree_traversal(bvh_gpu_node *b_arr,int index,int size){
//     if(index>=size)
//         return;
//     printf("\nIndex on the bvh_arr %d %d",b_arr[index].index_on_bvh_array,index);
//     printf("\nFor index- %d",b_arr[index].index_collider_list);
//     printf("\nx_min- %f  y_min- %f  z_min- %f\n",
//     b_arr[index].box.min().x(),
//     b_arr[index].box.min().y(),
//     b_arr[index].box.min().z());

//     printf("x_max- %f  y_max- %f  z_max- %f\n",
//     b_arr[index].box.max().x(),
//     b_arr[index].box.max().y(),
//     b_arr[index].box.max().z());
//     bvh_tree_traversal(b_arr,2*index+1,size);
//     bvh_tree_traversal(b_arr,2*index+2,size);
    
// }
// __global__ void initialize_bvh_tree(bvh_gpu_node *b_arr,bvh_gpu *bvh_tree,colliders *d_list,int size)
// {

//     bvh_tree[0] = bvh_gpu(b_arr,d_list,2*size-1);
//     bvh_tree_traversal(b_arr,0,2*size-1);

//     // for(int i=0;i<2*size-1;i++)
//     // {
//     //     printf("\nFor index_in_collider_list- %d For index_in_bvh- %d"
//     //     ,b_arr[i].index_collider_list,b_arr[i].index_on_bvh_array);
//     //     printf("\nx_min- %f  y_min- %f  z_min- %f\n",
//     //     b_arr[i].box.min().x(),
//     //     b_arr[i].box.min().y(),
//     //     b_arr[i].box.min().z());

//     //     printf("x_max- %f  y_max- %f  z_max- %f\n",
//     //     b_arr[i].box.max().x(),
//     //     b_arr[i].box.max().y(),
//     //     b_arr[i].box.max().z());
//     // }
// }

// void host_check(aabb *list_of_bounding_boxes,bvh_gpu_node *b_arr,int size)
// {
//     for(int i=0;i<size;i++)
//     {
//         printf("\nFor index- %d",list_of_bounding_boxes[i].index);
//         printf("\nx_min- %f  y_min- %f  z_min- %f\n",
//         list_of_bounding_boxes[i].min().x(),
//         list_of_bounding_boxes[i].min().y(),
//         list_of_bounding_boxes[i].min().z());

//         printf("x_max- %f  y_max- %f  z_max- %f\n",
//         list_of_bounding_boxes[i].max().x(),
//         list_of_bounding_boxes[i].max().y(),
//         list_of_bounding_boxes[i].max().z());

//         printf("centroid- (%f,%f,%f)\n",
//         list_of_bounding_boxes[i].get_centroid().x(),
//         list_of_bounding_boxes[i].get_centroid().y(),
//         list_of_bounding_boxes[i].get_centroid().z());
//     }
//     printf("\n\n");

//     bvh_cpu_sah parent = 
//     bvh_cpu_sah(0,size,0);

//     parent.form_gpu_bvh(list_of_bounding_boxes,b_arr);

//     //traverse_bvh_node(parent);


// }
// int main(){
//     int size = 4;

//     colliders *d_list;
//     collider_list **d_world;
//     bvh_gpu_node *bvh_arr;
//     bvh_gpu *bvh_tree;

//     int total_size_hittable = size*sizeof(colliders);
//     int total_size_bvh  = (2*size-1)*sizeof(bvh_gpu_node);
    
//     gpuErrchk(cudaMalloc(&bvh_tree,sizeof(bvh_gpu)));
//     gpuErrchk(cudaMalloc(&d_list,total_size_hittable));
//     gpuErrchk(cudaMalloc(&d_world,sizeof(collider_list*)));
//     gpuErrchk(cudaMallocManaged(&bvh_arr,total_size_bvh));

//     aabb *list_of_bounding_boxes;
//     int total_aabb_size = size*sizeof(aabb);
//     gpuErrchk(cudaMallocManaged(&list_of_bounding_boxes,total_aabb_size));

    
    


//     object_init<<<1,1>>>(d_list,d_world,list_of_bounding_boxes,size);
//     gpuErrchk(cudaGetLastError());
//     gpuErrchk(cudaDeviceSynchronize());

//     host_check(list_of_bounding_boxes,bvh_arr,size);

//     initialize_bvh_tree<<<1,1>>>(bvh_arr,bvh_tree,d_list,size);
    
//     // gpuErrchk(cudaGetLastError());
//     // gpuErrchk(cudaDeviceSynchronize());

// }

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

int main(int argc, char *argv[])
{
    for(int i=1;i<100;i++)
        printf("\n%d",i*32);
}

