import numpy as np
import cv2

from instapy.numpy_color2gray import numpy_grayscale_filter
from instapy.numpy_color2sepia import numpy_sepia_filter
from instapy.numba_color2gray import numba_grayscale_filter
from instapy.numba_color2sepia import numba_sepia_filter
from instapy.python_color2gray import python_grayscale_filter
from instapy.python_color2sepia import python_sepia_filter

def grayscale_filter(image, output_filename=None, implementation='numpy'):
    """Function that turns a image into greyscale.

    The function takes in an image in the BGR-format that is read using the cv2 module.
    The weights used to calculate the grayscale values are 0.21B + 0.72G + 0.07R.
    If output filename is given as an argument it will save the image as the specified file.

    Args:
        image (numpy.ndarray): A colorful image.
        output_filename (string): (optional) If given, the function will save the image as the specified file.
        implementation (string): (optional, default: numpy) Implementation to be used
    Returns:
        grey_image (numpy.ndarray): A greyscale version of the image.
    """
    if implementation == 'numpy':
        gray_image = numpy_grayscale_filter(image)
    elif implementation == 'numba':
        gray_image = numba_grayscale_filter(image)
    else:
        gray_image = python_grayscale_filter(image)

    if output_filename != None:
        cv2.imwrite(output_filename, gray_image)

    return gray_image

def sepia_filter(image, output_filename=None, implementation='numpy', intensity=None):
    """Function that puts a sepia filter on an image.

    The function takes in an image that is read using the cv2 module.
    If output filename is given as an argument it will save the image as the specified file.

    Args:
        image (numpy.ndarray): A colorful image.
        output_filename (string): (optional) If given, the function will save the image as the specified file.
        implementation (string): (optional, default: numpy) Implementation to be used.
        intensity (float): (optional) Intensity of sepia filter from 0 to 1. Only applied if implementation is numpy.
    Returns:
        sepia_image (numpy.ndarray): A sepia version of the image.
    """
    if implementation == 'numpy':
        sepia_image = numpy_sepia_filter(image, intensity)
    elif implementation == 'numba':
        sepia_image = numba_sepia_filter(image)
    else:
        sepia_image = python_sepia_filter(image)

    if output_filename != None:
        cv2.imwrite(output_filename, sepia_image)

    return sepia_image
