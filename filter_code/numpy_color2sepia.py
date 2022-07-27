import numpy as np
import cv2
import time
import matplotlib.pyplot as plt



def numpy_sepia_filter(image):
    """Function that puts a sepia filter on an image.

    The function takes in an image that is read using the cv2 module.
    
    Args:
        image (numpy.ndarray): A colorful image.
    Returns:
        sepia_image (numpy.ndarray): A sepia version of the image.
    """
    sepia_matrix = np.array(
        [[0.131, 0.534, 0.272], [0.168, 0.686, 0.349], [0.189, 0.769, 0.393]]
    )

    sepia_image = np.empty(image.shape)

    C1 = (
        sepia_matrix[0, 0] * image[:, :, 0]
        + sepia_matrix[0, 1] * image[:, :, 1]
        + sepia_matrix[0, 2] * image[:, :, 2]
    )
    C2 = (
        sepia_matrix[1, 0] * image[:, :, 0]
        + sepia_matrix[1, 1] * image[:, :, 1]
        + sepia_matrix[1, 2] * image[:, :, 2]
    )
    C3 = (
        sepia_matrix[2, 0] * image[:, :, 0]
        + sepia_matrix[2, 1] * image[:, :, 1]
        + sepia_matrix[2, 2] * image[:, :, 2]
    )

    

    sepia_image[:, :, 0] = C1
    sepia_image[:, :, 1] = C2
    sepia_image[:, :, 2] = C3


    # Scale the values such that no value exceeds 255
    max_val = np.max(sepia_image)
    scale_fac = 255 / max_val
    sepia_image = sepia_image*scale_fac
    
    sepia_image = sepia_image.astype("uint8")

    return sepia_image


if __name__ == "__main__":
    # Load image from file
    image = cv2.imread("../../figures/rain.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sepia_image = numpy_sepia_filter(image)

    plt.imshow(cv2.cvtColor(sepia_image, cv2.COLOR_BGR2RGB))
    plt.show()