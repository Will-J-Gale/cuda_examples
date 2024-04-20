#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vector_add(const float* a, const float* b, float* c, int num_elements)
{
    // printf("Print inside cuda");
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < num_elements)
    {
        c[i] = a[i] + b[i] + 0.0f;
    }
}

int vector_add_launch()
{
    cudaError_t status= cudaSuccess;

    // Print the vector length to be used, and compute its size
    int num_elements = 50000;
    size_t size = num_elements * sizeof(float);
    printf("[Vector addition of %d elements]\n", num_elements);

    float* host_a = (float*)malloc(size);
    float* host_b = (float*)malloc(size);
    float* host_c = (float*)malloc(size);

    // Verify that allocations succeeded
    if (host_a == NULL || host_b == NULL || host_c == NULL) 
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        return 1;
    }

    for (int i = 0; i < num_elements; ++i) 
    {
        host_a[i] = rand() / (float)RAND_MAX;
        host_b[i] = rand() / (float)RAND_MAX;
    }

    float* device_a = NULL;
    float* device_b = NULL;
    float* device_c = NULL;

    status = cudaMalloc((void**)&device_a, size);
    status = cudaMalloc((void**)&device_b, size);
    status = cudaMalloc((void**)&device_c, size);

    if(status != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory!\n");
        return 1;
    }

    status = cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);
    status = cudaMemcpy(device_b, host_b, size, cudaMemcpyHostToDevice);

    if(status != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy host to device!\n");
        return 1;
    }

    int threads_per_block = 256;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    vector_add<<<blocks_per_grid, threads_per_block>>>(device_a, device_b, device_c, num_elements);
    status = cudaGetLastError();

    if(status != cudaSuccess)
    {
        std::cout << "Failed to run kernal" << std::endl << cudaGetErrorString(status) << std::endl;
    }

    status = cudaMemcpy(host_c, device_c, size, cudaMemcpyDeviceToHost);

    if(status != cudaSuccess)
    {
        std::cout << "Failed to copy result from device" << std::endl;
    }

    //Destroy everything!
    free(host_a);
    free(host_b);
    free(host_c);
    cudaFree((void*)device_a);
    cudaFree((void*)device_b);
    cudaFree((void*)device_c);

    return 0;
}