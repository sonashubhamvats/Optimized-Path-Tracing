# Optimized-Path-Tracing
## About the project:
* Before anything I would like to extend my huge gratitude to the <a href='https://github.com/RayTracing/raytracing.github.io'>ray tracing series</a> by Peter Shirley, an absolute go to book for any beginners looking for resources on image rendering and path/ray tracing. 
* This project was my take on optimizing the traditional CPU based path tracing approach discussed in the above mentioned ray tracing series by Peter Shirley.
* I came at this project with no prior experience to graphics in C++ and on completion of this project I had explored almost all the facets of basic path tracing from simulating materials to creating acceleration structures for speeding my program, at last I conducted a lot of experiments by rendering a variety of scenes and recording concurrent speedups.
* Through this project I also explored the possibility of real time path tracing by rendering relatively less complex scenes with basic properties and materials.

## My Contribution
* As I discussed, the project derives its basic structure from the ray tracing series by Peter Shirley, and replicates the mathematical concepts discussed in the same.
* In my version I added quite a few no of functionalities and added changes to the overall architecture of the solutions - 
  * In my solution I am using a hybrid architecture wherein I am using the combined capabilities of the CPU and the GPU to render my scenes.
  * The rendering and sampling is handled by the GPU while the creation of the acceleration structures is done on the CPU.
  * I added an additional version of BVH which uses SAH(Surface Area Heuristics) costs to determine the split plane.
  * I added the functionality of rendering triangle primitives.
  * I added a custom 3d model(.obj) renderer.
  * I explored the possibility of real time path tracing using my highlighted solution.

## Requirements:
* As a part of an experiment I tried adding another layer of paralleliztion by using MPI, this part of project is not included in the main file, and its necessary packages can be downloaded from - <a href='https://www.microsoft.com/en-us/download/details.aspx?id=57467'>Download here MPI (Message Parsing Interface)</a>
* Also to show my output I am using the GLFW library, which needs OpenGL to run, other than that all the necessary libraries are included in the repository itself.

## What the app uses:
* The solution uses CUDA(Compute Unified Device Architecture) framework to enable parallelization and to run our native CPU code on the GPU. 
* The solution uses custom built accleration structures - BVH(Bounding Volume Hierarchy) trees, to speed up the search time to reduce final render times. I have tried to implement two relatively simple versions one which just uses std::nth_element to split up the plane and other which uses SAH(Surface Area Heuristics) costs.
* I have tried to implement a hybrid architecture, wherein the rendering and the sampling is handled by the GPU, while the creation of the acceleration structures happen on the CPU.
* Atlast I experimented on rendering relatively less complex scenes in real time, and also built a custom .obj 3d model renderer(with limitation that the renderer can only render models which are completely made up of triangles).

## How the app works:
* Taking into mind the complexity of the entire scale of the project, explaining the entire working would be out of the scope of this readme file.
* Although on a basic level the solution works like this - 
  * First we start by loading our scene into GPU, this could mean loading our 3d models, or loading in our randomly generated scenes.
  * Next using just the geometric orientation of the scene, we create our acceleration structures on the CPU and later transfer it on our GPU.
  * Upon completion of the pre processing of the scene, we move on with the rendering part, wherein we use two kernels, one which does the sampling and subsequent    path tracing for each ray.
  * And the other which is basically a reduction kernel which helps us in computing the final render of the scene.
  * This concludes our basic work flow, now for varied scenes and use cases there had been some changes in the overall structure of the solution, like for rendering scenes in real time we are using a limited version of our path tracer which can render only triangles and simple materials and textures.
  * Other than this the mathematical computations that work on the background to simulate different materials and textures has been discussed in detail in the ray tracing series by Peter Shirley.


## Running the application:
* Once the additional packages and libaries are in place, running this solution is pretty simple, I have divided the execution of the solution into three parts - 
### Compilation:
* The following commands could be used to compile our program - 
```
Navigate to mpi_code folder-
    nvcc mpi_main_call.cu -o ../exefiles/mpi_renderer_img "C:/Program Files/MPI/MS_MPI/SDK/Lib/x64/msmpi.lib"
    mpiexec -n 1 ../exefiles/mpi_renderer_img > ../images/mpi_random_triangles_10k.ppm

Navigate to main folder -
    nvcc -arch=sm_60 -lineinfo main.cu -o ./exefiles/master_renderer_gpu -L./src/lib-vc2019 -lglfw3_mt -lopengl32 -luser32 -lgdi32 
 ```
### Options 1-21
* The renders in this part basically explores the capabilities of our renderer in rendering primitives such as triangles, spheres, planes/boxes and simulating a variety of materials and textures such as metals, glass, smoke, noisy textures, checker textures etc.
  * To run we can execute using the following commands -
  ```
  For commands  1-21
      ./exefiles/master_renderer_gpu -10 -100 -736 -640 -bvh
      -name -option_no -no_of_objects -nx -ny -bvh_or_linear
  ```
 ### Options 22-26
* The renders in this part explores the capabilities of our renderer in rendering simple to complex 3d models at blazingly fast speeds.
  * To run we can use - 
  ```
  For commands 22-26
     ./exefiles/master_renderer_gpu -22 -10000 -736 -640 -bvh -random
     -name -option_no -no_of_objects -nx -ny -bvh_or_bvh_sah -random_or_model_name
  ```
### Option 27
* We try to render relatively simple scenes in real time for this part of our renderer.
  * We can use - 
  ```
  for commands 27
    ./exefiles/master_renderer_gpu -27 -100000 -736 -640 -./models/satellite.obj -depth -4
    -name -option_no -no_of_objects -nx -ny -random_model_name -depth_or_recursive_metal -samples
    
  ```

## Demonstration:
- Options 1-21
  * 10k random spheres with lambertian material and solid color texture(Option -1)
  ![Screenshot 2023-05-02 220024](https://user-images.githubusercontent.com/66525380/235728399-1197ef13-8819-48d4-8f27-a8a587577bed.jpg)
  
  * 10k random spheres with lambertian material and noise texture(Option -2)
  ![Screenshot 2023-05-02 220102](https://user-images.githubusercontent.com/66525380/235728478-3850dec4-bf41-4a94-82f2-34a4761f4ccb.jpg)
  
  * 10k random boxes with lambertian material and solid color texture(Option -9)
  ![Screenshot 2023-05-02 220115](https://user-images.githubusercontent.com/66525380/235728513-b25e535d-fb7f-469f-8826-8b2839e40501.jpg)

- Option 22-26
  * Sample Model - 130k triangles(Option -24) in < 0.5 secs
  ![image](https://user-images.githubusercontent.com/66525380/235728979-08e73a41-34a4-4c5d-afe3-75dd16f14b3e.png)
  
  * Sample Model - 4k triangles(Option - 26) in <0.5 secs
  ![image](https://user-images.githubusercontent.com/66525380/235729396-4d6f5f50-3fc9-487b-95c4-34017dcb931f.png)

- Option 27
 * 4K metal triangles rendered in real time > 40fps
 
 https://user-images.githubusercontent.com/66525380/235732157-52d07831-29e4-49f4-877b-1e56703bc2a9.mp4


