#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <chrono>

#include <kernels.h>
#include <opencv4/opencv2/opencv.hpp>

#include <conv2d_cpu.h>

int main(int argc, char* argv[])
{
    if(argc != 2)
    {
        std::cout << "Image path required as first argument" << std::endl;
        return 1;
    }
    // int kernel_size = 3;
    // float gausian[kernel_size * kernel_size] = {
    //     0.0625f, 0.125f,  0.0625f, 
    //     0.125f,  0.25f,   0.125f, 
    //     0.0625f, 0.125f,  0.0625f
    // };

    // int kernel_size = 5;
    // float gausian[kernel_size * kernel_size] = {
    //     0.00390625, 0.015625  , 0.0234375 , 0.015625  , 0.00390625,
    //     0.015625  , 0.0625    , 0.09375   , 0.0625    , 0.015625  ,
    //     0.0234375 , 0.09375   , 0.140625  , 0.09375   , 0.0234375 ,
    //     0.015625  , 0.0625    , 0.09375   , 0.0625    , 0.015625  ,
    //     0.00390625, 0.015625  , 0.0234375 , 0.015625  , 0.00390625
    // };

    int kernel_size = 7;
    float gausian[kernel_size * kernel_size] = {
        0.00097656, 0.00341797, 0.00683594, 0.00878906, 0.00683594,0.00341797, 0.00097656,
        0.00341797, 0.01196289, 0.02392578, 0.03076172, 0.02392578, 0.01196289, 0.00341797,
        0.00683594, 0.02392578, 0.04785156, 0.06152344, 0.04785156, 0.02392578, 0.00683594,
        0.00878906, 0.03076172, 0.06152344, 0.07910156, 0.06152344, 0.03076172, 0.00878906,
        0.00683594, 0.02392578, 0.04785156, 0.06152344, 0.04785156, 0.02392578, 0.00683594,
        0.00341797, 0.01196289, 0.02392578, 0.03076172, 0.02392578, 0.01196289, 0.00341797,
        0.00097656, 0.00341797, 0.00683594, 0.00878906, 0.00683594, 0.00341797, 0.00097656
    };

    int num_runs = 10;
    cv::Mat image = cv::imread(argv[1]);
    cv::resize(image, image, cv::Size(0, 0), 0.25f, 0.25f);
    cv::Mat dst;

    //CPU
    float avg_cpu_time = 0.0f;
    for(int i = 0; i < num_runs; i++)
    {
        double cpu_start = time();
        convolve(image, dst, &gausian[0], kernel_size);
        double cpu_dt = time() - cpu_start;
        avg_cpu_time += cpu_dt;
    }

    avg_cpu_time /= (float)num_runs;
    std::cout << "CPU: " << avg_cpu_time << std::endl;
    cv::imshow("image", image);
    cv::imshow("filtered", dst);
    cv::waitKey(0);

    //CUDA
    //Warmup GPU
    conv2d_launch(image, dst, &gausian[0], kernel_size);

    float avg_cuda_time = 0.0f;
    for(int i = 0; i < 10; i++)
    {
        double cuda_start = time();
        conv2d_launch(image, dst, &gausian[0], kernel_size);
        double cuda_dt = time() - cuda_start;
        avg_cuda_time += cuda_dt;
    }

    avg_cuda_time /= (float)num_runs;
    std::cout << "CUDA: " << avg_cuda_time << std::endl;
    cv::imshow("image", image);
    cv::imshow("filtered", dst);
    cv::waitKey(0);

    float speed_increase = avg_cpu_time / avg_cuda_time;
    std::cout << "Speed increase: " << speed_increase << std::endl;

    return 0;
}