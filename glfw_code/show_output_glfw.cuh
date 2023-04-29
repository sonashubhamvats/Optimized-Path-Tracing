#include <stdio.h>
#include <cuda_runtime.h>
#include "../src/include/GLFW/glfw3.h"


void show_windows_animated(unsigned char* h_rgb_values,int window_width,int window_height,bool &first_time,
GLFWwindow*& window)
{
    if(first_time)
    {
        // Initialize GLFW
        if (!glfwInit()) {
            fprintf(stderr, "Failed to initialize GLFW\n");
            return;
        }

        // Create a windowed mode window and its OpenGL context
        window = glfwCreateWindow(window_width, window_height, "Ray tracing output", NULL, NULL);
        if (!window) {
            fprintf(stderr, "Failed to create GLFW window\n");
            glfwTerminate();
            return;
        }

        // Make the window's context current
        glfwMakeContextCurrent(window);
        first_time = false;
    }
    

    // Loop until the user closes the window
    //while (!glfwWindowShouldClose(window)) {
        // Render RGB buffer to window
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawPixels(window_width, window_height, GL_RGB, GL_UNSIGNED_BYTE, h_rgb_values);
        glfwSwapBuffers(window);
        glfwPollEvents();
    //}

    // Clean up GLFW
    //glfwTerminate();

    

}

void show_windows(unsigned char* h_rgb_values,int window_width,int window_height,
GLFWwindow *window)
{
    
    // Initialize GLFW
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return;
    }

    // Create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(window_width, window_height, "Ray tracing output", NULL, NULL);
    if (!window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    
    

    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window)) {
        // Render RGB buffer to window
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawPixels(window_width, window_height, GL_RGB, GL_UNSIGNED_BYTE, h_rgb_values);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Clean up GLFW
    glfwTerminate();

    

}
