#ifndef TEXTURE_H
#define TEXTURE_H

#include "constants.cuh"
#include "perlin.cuh"

__device__ float clamp(float var , float low , float high)
{
    if(var<low)
        return low;
    if(var>high)
        return high;
    return var;
}

class solid_color{
    public:
        __device__ solid_color(){}
        __device__ solid_color(vec3 c): color_value(c){}

        __device__ solid_color(float red,float green,float blue)
        :solid_color(vec3(red,green,blue)){}

        __device__ vec3 value(float u,float v,const vec3 &p)const{
            return color_value;
        }

        vec3 color_value;
};

class noise_texture{
    public:
        perlin noise;
        float scale;
        __device__ noise_texture(){};
        __device__ noise_texture(curandState &localState,
        float s):
        noise(perlin(localState)),scale(s){}
        __device__ vec3 value(float u,float v,const vec3 &p)
        const{
            return vec3(1,1,1) * 0.5 * (1 + sinf(scale*p.z() + 10*noise.turb(p)));
        }
        
};

class checker_texture{
    public:
        __device__ checker_texture(){}
        __device__ checker_texture(vec3 &c1,vec3 &c2)
        :even(c1),odd(c2){}

        __device__ vec3 value(float u,float v, const vec3 &p) const{
            float sines = sinf(10*p.x())*sinf(10*p.y())*sinf(10*p.z());
            if (sines < 0)
                return odd;
            else
                return even;
        }
    
    vec3 odd;
    vec3 even;

};

class image_texture{
    public:
        const static int byte_per_pixel = 3;
        int width,height;
        unsigned char *data;

        __device__ image_texture()
            :data(NULL), width(0) , height(0){}

        __device__ image_texture(unsigned char **d,int w,int h)
            :data(*d), width(w) , height(h) {}

        __device__ vec3 value(float u, float v, 
        const vec3 &p) const{
            if(data==NULL)
                return vec3(0,1,1);
            
            u = clamp(u,0.0,1.0);
            v = 1.0 - clamp(v,0.0,1.0);

            int i = (int)(u*width);
            int j = (int)(v*height);

            if(i>=width)
                i = width - 1;
            if(j>=height)
                j = height - 1;

            const float color_scale = 1.0 / 255.0;
            unsigned char *pixel = data + j*byte_per_pixel*width + i*byte_per_pixel;

            return vec3(color_scale*pixel[0],color_scale*pixel[1],
            color_scale*pixel[2]);
        }
        
};


struct collider_texture{
    solid_color *solid_color_texture;
    noise_texture *noise_texture_here;
    checker_texture *checker_texture_here;
    image_texture *image_texture_here;
    int type_of_texture;

    __device__ collider_texture(solid_color *s,int t_o_t){
        type_of_texture = t_o_t;
        solid_color_texture = s;
    }

    __device__ collider_texture(noise_texture *n_t,int t_o_t){
        type_of_texture = t_o_t;
        noise_texture_here = n_t;
    }

    __device__ collider_texture(checker_texture *c_t,int t_o_t){
        type_of_texture = t_o_t;
        checker_texture_here = c_t;
    }
    __device__ collider_texture(image_texture *i_t,int t_o_t){
        type_of_texture = t_o_t;
        image_texture_here = i_t;
    }

    __device__ vec3 value(float u,float v,const vec3 &p){
        if(type_of_texture==solid_color_type_index)
        {
            return solid_color_texture->value(u,v,p);
        }
        else if(type_of_texture==noise_texture_type_index)
        {
            return noise_texture_here->value(u,v,p);
        }
        else if(type_of_texture==checker_texture_type_index)
        {
            return checker_texture_here->value(u,v,p);
        }
        else if(type_of_texture==image_texture_type_index)
        {
            return image_texture_here->value(u,v,p);
        }
    }
};






#endif