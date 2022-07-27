import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
import time 


def python_grayscale_filter(image):
    """Function that turns a image into greyscale. 

    The function takes in an image in the BGR-format that is read using the cv2 module. 
    The weights used to calculate the grayscale values are 0.21B + 0.72G + 0.07R.
    
    Args:
        image (numpy.ndarray): A colorful image.
    Returns:
        grey_image (numpy.ndarray): A greyscale version of the image. 
    """
    gray_image = np.empty(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            gray_image[i][j] = 0.21*image[i][j][0] + 0.72*image[i][j][1] + 0.07*image[i][j][2]
    gray_image = gray_image.astype("uint8")
    return gray_image

if __name__ == '__main__':
    # Load image from file
    image = cv2.imread("../../figures/rain.jpg")
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

    gray_image = python_grayscale_filter(image)

    plt.imshow(cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB))
    # plt.imshow(gray_image)
    plt.show()