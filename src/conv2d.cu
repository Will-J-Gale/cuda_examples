#include <iostream>
#include <stdio.h>
#include <cmath>

#include <cuda_runtime.h>
#include <opencv4/opencv2/opencv.hpp>

using byte = unsigned char;

__device__ static uint xy_to_image_index(uint x, uint y, uint width, uint channels)
{
    return (x + (width * y)) * channels;
}

__global__ void conv2d(
    const byte* image, 
    byte* dst, 
    float* kernel, 
    int image_size_in_bytes, 
    int kernel_size, 
    int image_width,
    int image_height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = xy_to_image_index(x, y, image_width, 3);
    int k_mid = (kernel_size - 1) / 2;

    if(index < image_size_in_bytes)
    {
        float avg_b = 0.0f;
        float avg_g = 0.0f;
        float avg_r = 0.0f;

        int filter_accum = 0;
        for(int filter_y = -k_mid; filter_y <= k_mid; filter_y++)
        {
            for(int filter_x = -k_mid; filter_x <= k_mid; filter_x++)
            {
                filter_accum += 1;
                int neighbor_x = x + filter_x;
                int neighbor_y = y + filter_y;

                if(neighbor_x < 0 or neighbor_x >= image_width or neighbor_y < 0 or neighbor_y >= image_height)
                {
                    continue;
                }

                uint filter_index = xy_to_image_index(filter_x + k_mid, filter_y + k_mid, kernel_size, 1);
                float filter_value = kernel[filter_index];

                int neighbor_index = xy_to_image_index(x + filter_x, y + filter_y, image_width, 3);
                float neightbor_value_b = (float)image[neighbor_index];
                float neightbor_value_g = (float)image[neighbor_index + 1];
                float neightbor_value_r = (float)image[neighbor_index + 2];

                avg_b += neightbor_value_b * filter_value;
                avg_g += neightbor_value_g * filter_value;
                avg_r += neightbor_value_r * filter_value;
            }
        }

        dst[index] = avg_b;
        dst[index+1] = avg_g;
        dst[index+2] = avg_r;
    }
}

void conv2d_launch(const cv::Mat& image, cv::Mat& dst, float* kernel, int kernel_size)
{
    cudaError_t status = cudaSuccess;
    // printf("%f %f %f %f %f %f %f %f %f\n", kernel[0], kernel[1], kernel[2], kernel[3], kernel[4], kernel[5], kernel[6], kernel[7], kernel[8]);
    // printf("%f, %f, %f\n", kernel[0], kernel[1], kernel[2]);
    // return;
    long image_size_in_bytes = image.rows * image.cols * image.elemSize();
    int kernel_size_in_bytes = (kernel_size * kernel_size) * sizeof(float);
    dst = cv::Mat(cv::Size(image.cols, image.rows), CV_8UC3, cv::Scalar(0, 0, 0));

    byte* device_image = NULL;
    byte* device_dst = NULL;
    float* device_kernel = NULL;

    status = cudaMalloc((void**)&device_image, image_size_in_bytes);
    status = cudaMalloc((void**)&device_dst, image_size_in_bytes);
    status = cudaMalloc((void**)&device_kernel, kernel_size_in_bytes);

    if(status != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory!\n");
        return;
    }

    status = cudaMemcpy(device_image, image.data, image_size_in_bytes, cudaMemcpyHostToDevice);
    status = cudaMemcpy(device_kernel, kernel, kernel_size_in_bytes, cudaMemcpyHostToDevice);

    if(status != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy host to device!\n");
        return;
    }

    int block_size = 32;
    dim3 threads_per_block(block_size, block_size);
    dim3 num_blocks(std::ceil((float)image.cols / (float)block_size), std::ceil((float)image.rows / (float)block_size));

    conv2d<<<num_blocks, threads_per_block>>>(
        device_image, 
        device_dst, 
        device_kernel, 
        image_size_in_bytes, 
        kernel_size,
        image.cols,
        image.rows
    );
    status = cudaGetLastError();

    if(status != cudaSuccess)
    {
        std::cout << "Failed to run kernal" << std::endl << cudaGetErrorString(status) << std::endl;
        return; 
    }

    status = cudaMemcpy(dst.data, device_dst, image_size_in_bytes, cudaMemcpyDeviceToHost);

    if(status != cudaSuccess)
    {
        std::cout << "Failed to copy result from device" << std::endl;
        return;
    }

    //Destroy everything!
    cudaFree((void*)device_image);
    cudaFree((void*)device_dst);
    cudaFree((void*)device_kernel);
}