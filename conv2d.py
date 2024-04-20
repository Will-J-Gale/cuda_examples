import cv2
import numpy as np

def filter_image(image, kernel):
    height, width, channels = image.shape
    # filter_size = kernel.shape[1]
    dst = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            avg_b = 0.0
            avg_g = 0.0
            avg_r = 0.0

            for filter_y in range(-1, 2):
                for filter_x in range(-1, 2):
                    neighbor_x = x + filter_x
                    neighbor_y = y + filter_y

                    if(neighbor_x < 0 or neighbor_x >= width or neighbor_y < 0 or neighbor_y >= height):
                        continue
                
                    avg_b += kernel[filter_y, filter_x] * image[neighbor_y, neighbor_x, 0]
                    avg_g += kernel[filter_y, filter_x] * image[neighbor_y, neighbor_x, 1]
                    avg_r += kernel[filter_y, filter_x] * image[neighbor_y, neighbor_x, 2]

            dst[y, x, 0] = avg_b
            dst[y, x, 1] = avg_g
            dst[y, x, 2] = avg_r
    
    return dst

kernel1 = np.array([
    [0.0625, 0.125,  0.0625], 
    [0.125,  0.25,   0.125], 
    [0.0625, 0.125,  0.0625]
])
image = cv2.imread("/home/will/Pictures/Wallpapers/crystal-skies-4k-j6.jpg")
image = cv2.resize(image, (0,0), fx=0.25, fy=0.25)

filtered_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel1)
filtered_image_2 = filter_image(image, kernel1)

print(np.array_equal(filtered_image, filtered_image_2))

cv2.imshow("image", image)
cv2.imshow("filter", filtered_image)
cv2.imshow("filter2", filtered_image_2)
cv2.waitKey(0)
