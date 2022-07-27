import numpy as np 
import matplotlib.pyplot as plt 

from random import randint 

import cv2 

from instapy import grayscale_filter
from instapy import sepia_filter

from instapy.python_color2gray import python_grayscale_filter
from instapy.numpy_color2gray import numpy_grayscale_filter
from instapy.numba_color2gray import numba_grayscale_filter

from instapy.python_color2sepia import python_sepia_filter
from instapy.numpy_color2sepia import numpy_sepia_filter
from instapy.numba_color2sepia import numba_sepia_filter


def test_grayscale_image():
    """This function tests the python-, numpy- and numba implementations of the grayscale filter
    against an exact solution to a randomly generated image."""
    shape = (300, 300, 3)
    image = np.random.randint(0, 455, size=shape)
    python_gray_image = python_grayscale_filter(image)
    numpy_gray_image = numpy_grayscale_filter(image)
    numba_gray_image = numba_grayscale_filter(image)


    # Test against random indexes. Note: we use the BGR-format
    for _ in range(100):
        x, y = randint(0, 299), randint(0, 299)
        correct_val = 0.21*image[x, y, 0] + 0.72*image[x, y, 1] + 0.07*image[x, y, 2]
        correct_val = correct_val.astype("uint8")
        assert correct_val == python_gray_image[x, y, 0]
        assert correct_val == python_gray_image[x, y, 1]
        assert correct_val == python_gray_image[x, y, 2]

        assert correct_val == numpy_gray_image[x, y, 0]
        assert correct_val == numpy_gray_image[x, y, 1]
        assert correct_val == numpy_gray_image[x, y, 2]

        assert correct_val == numba_gray_image[x, y, 0]
        assert correct_val == numba_gray_image[x, y, 1]
        assert correct_val == numba_gray_image[x, y, 2]

def test_sepia_image():
    """This function tests the python-, numpy- and numba implementations of the sepia filter
    against an exact solution to a randomly generated image."""
    shape = (300, 300, 3)
    image = np.random.randint(0, 255, size=shape)
    python_sepia_image = python_sepia_filter(image)
    numpy_sepia_image = numpy_sepia_filter(image)
    numba_sepia_image = numba_sepia_filter(image)

    sepia_matrix = np.array([[0.131, 0.534, 0.272], [0.168, 0.686, 0.349], [0.189, 0.769, 0.393]])

    # Find the scale factor used in the different implementations and remove it from the final answer
    first_exact_val = image[0,0,0] * sepia_matrix[0, 0] + image[0,0,1] * sepia_matrix[0, 1] + image[0,0,2] * sepia_matrix[0, 2]
    
    python_first_val = python_sepia_image[0,0,0]
    numpy_first_val = numpy_sepia_image[0,0,0]
    numba_first_val = numba_sepia_image[0,0,0]

    python_scale_fac = python_first_val / first_exact_val
    numpy_scale_fac = numpy_first_val / first_exact_val
    numba_scale_fac = numba_first_val / first_exact_val

    print(python_scale_fac, numpy_scale_fac, numba_scale_fac)

    # Test against random indexes. Note: we use the BGR-format
    for _ in range(100):
        x, y = randint(0, 299), randint(0, 299)

        B, G, R = image[x, y]
        
        C1 = int(B * sepia_matrix[0, 0] + G * sepia_matrix[0, 1] + R * sepia_matrix[0, 2])
        C2 = int(B * sepia_matrix[1, 0] + G * sepia_matrix[1, 1] + R * sepia_matrix[1, 2])
        C3 = int(B * sepia_matrix[2, 0] + G * sepia_matrix[2, 1] + R * sepia_matrix[2, 2])
        
        C1 = C1
        C2 = C2
        C3 = C3

        python_sepia_image = python_sepia_image.astype(int)
        numpy_gray_image = numpy_sepia_image.astype(int)
        numba_gray_image = numba_sepia_image.astype(int)

        
        # Note: we check only that the difference is less than four because of roundoff error when finding the scale factor
        assert abs(python_sepia_image[x, y, 0] - int(C1*python_scale_fac)) <= 4
        assert abs(python_sepia_image[x, y, 1] - int(C2*python_scale_fac)) <= 4
        assert abs(python_sepia_image[x, y, 2] - int(C3*python_scale_fac)) <= 4

        assert abs(numpy_sepia_image[x, y, 0] - int(C1*numpy_scale_fac)) <= 4
        assert abs(numpy_sepia_image[x, y, 1] - int(C2*numpy_scale_fac)) <= 4
        assert abs(numpy_sepia_image[x, y, 2] - int(C3*numpy_scale_fac)) <= 4

        assert abs(numba_sepia_image[x, y, 0] - int(C1*numba_scale_fac)) <= 4
        assert abs(numba_sepia_image[x, y, 1] - int(C2*numba_scale_fac)) <= 4
        assert abs(numba_sepia_image[x, y, 2] - int(C3*numba_scale_fac)) <= 4


