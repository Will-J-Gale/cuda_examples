#include <vector>
#include <cmath>

constexpr float TWO_PI = 6.283185307179586;

inline std::vector<float> generate_gausian(uint kernel_size, float sigma=1.0f, float muu=0.0f)
{
    std::vector<float> gausian = std::vector<float>();
    int k_mid = (kernel_size - 1) / 2;
    float sigma_sqr = sigma * sigma;
    float sum = 0.0f;

    for(int i = -k_mid; i <= k_mid; i++)
    {
        float value = (1 / (sigma * std::sqrt(TWO_PI))) * std::exp(-((i - muu) * (i - muu)) / (2 * sigma_sqr));
        gausian.push_back(value);
        sum += value;
    }

    for(int i = 0; i < gausian.size(); i++)
    {
        gausian[i] /= sum;
    }

    return gausian;
}