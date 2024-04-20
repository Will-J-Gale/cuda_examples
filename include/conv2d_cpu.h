/*
    Image byte order
    1 pixel = 3 bytes
    bgr

    0,  1,  2,  3
    4,  5,  6,  7,
    8,  9,  10, 11,
    12, 13, 14, 15

    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    For actual image data

    0bgr,  1bgr,  2bgr,  3bgr
    4bgr,  5bgr,  6bgr,  7bgr,
    8bgr,  9bgr,  10bgr, 11bgr,
    12bgr, 13bgr, 14bgr, 15bgr

    [0b, 0g, 0r, 1b, 1g, 1r, 2b, 2g, 2r, 3b, 3g, 3r, ... 15b, 15g, 15r
*/


#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <chrono>

#include <kernels.h>
#include <opencv4/opencv2/opencv.hpp>

using byte = unsigned char;

inline uint xy_to_image_index(uint x, uint y, uint width, uint channels = 3)
{
    return (x + (width * y)) * channels;
}

inline double time()
    {
        auto currentTime = std::chrono::system_clock::now();
        auto duration = std::chrono::duration<double>(currentTime.time_since_epoch());

        return duration.count();
    }

inline void convolve(const cv::Mat& image, cv::Mat& dst, float* kernel, int kernel_size)
{
    int width = image.cols;
    int height = image.rows;

    dst = cv::Mat(cv::Size(width, height), CV_8UC3, cv::Scalar(0, 0, 0));

    const byte* src_data = image.data;
    byte* dst_data = dst.data;

    int k_mid = (kernel_size - 1) / 2;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int index = xy_to_image_index(x, y, width);
            float avg_b = 0.0f;
            float avg_g = 0.0f;
            float avg_r = 0.0f;


            for(int filter_y = -k_mid; filter_y <= k_mid; filter_y++)
            {
                for(int filter_x = -k_mid; filter_x <= k_mid; filter_x++)
                {
                    int neighbor_x = x + filter_x;
                    int neighbor_y = y + filter_y;

                    if(neighbor_x < 0 or neighbor_x >= width or neighbor_y < 0 or neighbor_y >= height)
                    {
                        continue;
                    }

                    uint filter_index = xy_to_image_index(filter_x + k_mid, filter_y + k_mid, kernel_size, 1);
                    float filter_value = kernel[filter_index];

                    int neighbor_index = xy_to_image_index(x + filter_x, y + filter_y, width);
                    float neightbor_value_b = (float)src_data[neighbor_index];
                    float neightbor_value_g = (float)src_data[neighbor_index + 1];
                    float neightbor_value_r = (float)src_data[neighbor_index + 2];

                    avg_b += neightbor_value_b * filter_value;
                    avg_g += neightbor_value_g * filter_value;
                    avg_r += neightbor_value_r * filter_value;
                }
            }

            dst_data[index] = avg_b;
            dst_data[index+1] = avg_g;
            dst_data[index+2] = avg_r;
        }
    }
}