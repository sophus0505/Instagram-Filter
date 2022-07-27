import numpy as np
import cv2
import matplotlib.pyplot as plt
from numba import jit


@jit(nopython=True)
def loop(sepia_image, image, sepia_matrix, len_x, len_y):
    """Function that calculates the sepia values in a jit-friendly loop.

    Args:
        sepia_image (numpy.ndarray): The sepia image we are calculating.
        image (numpy.ndarray): The original image. 
        len_x (int): The length of the numpy.ndarrays in the 1st dimension.
        len_y (int): The lenght of the numpy.ndarrays in the 2nd dimesion.
    """
    for i in range(len_x):
            for j in range(len_y):
                B, G, R = image[i, j]
                C1 = int(
                    B * sepia_matrix[0, 0] + G * sepia_matrix[0, 1] + R * sepia_matrix[0, 2]
                )
                C2 = int(
                    B * sepia_matrix[1, 0] + G * sepia_matrix[1, 1] + R * sepia_matrix[1, 2]
                )
                C3 = int(
                    B * sepia_matrix[2, 0] + G * sepia_matrix[2, 1] + R * sepia_matrix[2, 2]
                )
                sepia_image[i, j] = np.array([C1, C2, C3])



def numba_sepia_filter(image):
    """Function that puts a sepia filter on an image.

    The function takes in an image in the BGR-format that is read using the cv2 module.

    Args:
        image (numpy.ndarray): A colorful image.
    Returns:
        sepia_image (numpy.ndarray): A sepia version of the image.
    """
    sepia_matrix = np.array(
        [[0.131, 0.534, 0.272], [0.168, 0.686, 0.349], [0.189, 0.769, 0.393]]
    )

    sepia_image = np.empty(image.shape)

    loop(sepia_image, image, sepia_matrix, image.shape[0], image.shape[1])

    # Scale the values such that no value exceeds 255
    max_val = np.max(sepia_image)
    scale_fac = 255 / max_val
    sepia_image = sepia_image*scale_fac

    sepia_image = sepia_image.astype("uint8")

    return sepia_image


if __name__ == "__main__":
    # Load image from file
    image = cv2.imread("../../figures/rain.jpg")

    sepia_image = numba_sepia_filter(image)

    plt.imshow(cv2.cvtColor(sepia_image, cv2.COLOR_BGR2RGB))
    # plt.imshow(sepia_image)
    plt.show()

    
