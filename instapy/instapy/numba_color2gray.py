from numba import jit

import numpy as np 
import matplotlib.pyplot as plt 

import cv2

@jit(nopython=True)
def loop(gray_image, image, len_x, len_y):
    """Function that calculates the grayscale values in a jit-friendly loop.

    Args:
        gray_image (numpy.ndarray): The grayscale image we are calculating.
        image (numpy.ndarray): The original image. 
        len_x (int): The length of the numpy.ndarrays in the 1st dimension.
        len_y (int): The lenght of the numpy.ndarrays in the 2nd dimesion.
    """
    for i in range(len_x):
        for j in range(len_y):
            gray_image[i][j][0] = 0.21*image[i,j,0] + 0.72*image[i][j][1] + 0.07*image[i][j][2]
            gray_image[i][j][1] = 0.21*image[i,j,0] + 0.72*image[i][j][1] + 0.07*image[i][j][2]
            gray_image[i][j][2] = 0.21*image[i,j,0] + 0.72*image[i][j][1] + 0.07*image[i][j][2]

def numba_grayscale_filter(image):
    """Function that turns a image into greyscale. 

    The function takes in an image in the BGR-format that is read using the cv2 module. 
    The weights used to calculate the grayscale values are 0.21B + 0.72G + 0.07R.
    
    Args:
        image (numpy.ndarray): A colorful image.
    Returns:
        grey_image (numpy.ndarray): A greyscale version of the image. 
    """
    gray_image = np.empty(image.shape)
    len_x = len(image)
    len_y = len(image[0])
    loop(gray_image, image, len_x, len_y)
    gray_image = gray_image.astype("uint8")

    return gray_image

if __name__ == '__main__':
    # Load image from file
    image = cv2.imread("../../figures/rain.jpg")
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

    gray_image = numba_grayscale_filter(image)

    plt.imshow(cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB))
    plt.show()




