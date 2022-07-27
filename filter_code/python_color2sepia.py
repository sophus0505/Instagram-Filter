import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

def python_sepia_filter(image):
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

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # print(image[i, j])
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

    # Scale the values such that no value exceeds 255
    max_val = np.max(sepia_image)
    scale_fac = 255 / max_val
    sepia_image = sepia_image*scale_fac

    sepia_image = sepia_image.astype("uint8")

    return sepia_image


if __name__ == "__main__":
    # Load image from file
    image = cv2.imread("../../figures/rain.jpg")
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sepia_image = python_sepia_filter(image)

    plt.imshow(cv2.cvtColor(sepia_image, cv2.COLOR_BGR2RGB))
    plt.show()


    
