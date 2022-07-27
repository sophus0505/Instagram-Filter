import cv2
import time
import sys

import numpy as np 

sys.path.insert(0, '../filter_code')
from python_color2gray import python_grayscale_filter
from numpy_color2gray import numpy_grayscale_filter
from numba_color2gray import numba_grayscale_filter

# Read image from file 
image = cv2.imread("../figures/rain.jpg")

# Calculate average runtime over three runs of python_grayscale_filter()
times = []
for _ in range(3):
    start = time.time()
    gray_image = python_grayscale_filter(image)
    end = time.time()
    times.append(end - start)
python_average_time = np.average(times)

# Write results to the report file
with open("python_report_color2gray.txt", 'w') as infile:
    infile.write(f"Timing: python_color2gray \n")
    infile.write(f"Average runtime running python_color2gray after 3 runs: {python_average_time:.7} s \n")
    infile.write(f"Timing performed using: {image.shape} \n")

# Calculate average runtime over three runs of numpy_grayscale_filter()
times = []
for _ in range(3):
    start = time.time()
    gray_image = numpy_grayscale_filter(image)
    end = time.time()
    times.append(end - start)
numpy_average_time = np.average(times)

# Write the results to the report file
with open("numpy_report_color2gray.txt", "w") as infile:
    infile.write(f"Timing: numpy_color2gray \n")
    infile.write(
        f"Average runtime running numpy_color2gray after 3 runs: {numpy_average_time:.7} s \n"
    )
    if numpy_average_time <= python_average_time:
        infile.write(
            f"Average runtime running of numpy_color2gray is {python_average_time/numpy_average_time:.7} faster than python_color2gray \n"
        )
    else:
        infile.write(
            f"Average runtime running of numpy_color2gray is {numpy_average_time/python_average_time:.7} slower than python_color2gray \n"
        )
    infile.write(f"Timing performed using: {image.shape} \n")

# Calculate average runtime over three runs of numba_grayscale_filter()
times = []
for _ in range(3):
    start = time.time()
    gray_image = numba_grayscale_filter(image)
    end = time.time()
    times.append(end - start)
numba_average_time = np.average(times)

# Write results to the report file
with open("numba_report_color2gray.txt", 'w') as infile:
    infile.write(f"Timing: numba_color2gray \n")
    infile.write(f"Average runtime running numba_color2gray after 3 runs: {numba_average_time:.7} s \n")
    if (numba_average_time <= python_average_time):
        infile.write(f"Average runtime running of numba_color2gray is {python_average_time/numba_average_time:.7} faster than python_color2gray \n")
    else:
        infile.write(f"Average runtime running of numba_color2gray is {numba_average_time/python_average_time:.7} slower than python_color2gray \n")
    if (numba_average_time <= numpy_average_time):
        infile.write(f"Average runtime running of numba_color2gray is {numpy_average_time/numba_average_time:.7} faster than numpy_color2gray \n")
    else:
        infile.write(f"Average runtime running of numba_color2gray is {numba_average_time/numpy_average_time:.7} slower than numpy_color2gray \n")
    infile.write(f"Timing performed using: {image.shape} \n")