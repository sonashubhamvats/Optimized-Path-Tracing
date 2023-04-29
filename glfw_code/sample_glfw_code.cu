#include <stdio.h>
#include <cuda_runtime.h>
#include "../src/include/GLFW/glfw3.h"

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600

// CUDA kernel to fill the RGB buffer with values
__global__ void fillRGB(unsigned char *rgb) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = idx * 3;
    // rgb[offset] = idx % 255;     // R value
    // rgb[offset + 1] = (idx * 3) % 255; // G value
    // rgb[offset + 2] = (idx * 7) % 255; // B value

    rgb[offset] = 255;     // R value
    rgb[offset + 1] = 232; // G value
    rgb[offset + 2] = 12; // B value
}

int main(int argc, char **argv) {
    GLFWwindow* window;

    // Initialize GLFW
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return -1;
    }

    // Create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Custom RGB Window", NULL, NULL);
    if (!window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    // Allocate memory for RGB buffer on device
    unsigned char *dev_rgb;
    cudaMalloc((void**)&dev_rgb, WINDOW_WIDTH * WINDOW_HEIGHT * 3 * sizeof(unsigned char));

    // Fill RGB buffer with values using kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (WINDOW_WIDTH * WINDOW_HEIGHT + threadsPerBlock - 1) / threadsPerBlock;
    fillRGB<<<blocksPerGrid, threadsPerBlock>>>(dev_rgb);

    // Allocate memory for RGB buffer on host
    unsigned char *host_rgb = new unsigned char[WINDOW_WIDTH * WINDOW_HEIGHT * 3];

    // Copy RGB buffer from device to host
    cudaMemcpy(host_rgb, dev_rgb, WINDOW_WIDTH * WINDOW_HEIGHT * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window)) {
        // Render RGB buffer to window
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawPixels(WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, host_rgb);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Clean up GLFW
    glfwTerminate();

    // Clean up memory
    cudaFree(dev_rgb);
    delete[] host_rgb;

    return 0;
}