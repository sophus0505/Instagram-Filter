import cv2 as cv2 
import time as time
import numpy as np
import sys 

sys.path.insert(0, '../filter_code')
from instapy.python_color2sepia import python_sepia_filter
from instapy.numpy_color2sepia import numpy_sepia_filter
from instapy.numba_color2sepia import numba_sepia_filter

# Read image from file
image = cv2.imread("../figures/rain.jpg")

# Calculate average runtime over three runs of python_sepia_filter()
times = []
for _ in range(3):
    start = time.time()
    sepia_image = python_sepia_filter(image)
    end = time.time()
    times.append(end - start)
python_average_time = np.average(times)

# Write results to the report file
with open("python_report_color2sepia.txt", "w") as infile:
    infile.write(f"Timing: python_color2sepia \n")
    infile.write(
        f"Average runtime running python_color2gray after 3 runs: {python_average_time:.7} s \n"
    )
    infile.write(f"Timing performed using: {image.shape} \n")

# Calculate average runtime over three runs of numpy_sepia_filter()
    times = []
    for _ in range(3):
        start = time.time()
        sepia_image = numpy_sepia_filter(image)
        end = time.time()
        times.append(end - start)
    numpy_average_time = np.average(times)

    # Write results to the report file
    with open("numpy_report_color2sepia.txt", "w") as infile:
        infile.write(f"Timing: numpy_color2sepia \n")
        infile.write(
            f"Average runtime running numpy_color2sepia after 3 runs: {numpy_average_time:.7} s \n"
        )
        if numpy_average_time <= python_average_time:
            infile.write(
                f"Average runtime running of numpy_color2sepia is {python_average_time/numpy_average_time:.7} faster than python_color2sepia \n"
            )
        else:
            infile.write(
                f"Average runtime running of numpy_color2sepia is {numpy_average_time/python_average_time:.7} slower than python_color2sepia \n"
            )
        infile.write(f"Timing performed using: {image.shape} \n")


# Calculate average runtime over three runs of numba_sepia_filter()
times = []
for _ in range(3):
    start = time.time()
    sepia_image = numba_sepia_filter(image)
    end = time.time()
    times.append(end - start)
numba_average_time = np.average(times)

# Write results to the report file
with open("numba_report_color2sepia.txt", "w") as infile:
    infile.write(f"Timing: numba_color2sepia \n")
    infile.write(
        f"Average runtime running numba_color2sepia after 3 runs: {numba_average_time:.7} s \n"
    )

    if numba_average_time <= python_average_time:
        infile.write(
            f"Average runtime running of numba_color2sepia is {python_average_time/numba_average_time:.7} faster than python_color2sepia \n"
        )
    else:
        infile.write(
            f"Average runtime running of numba_color2sepia is {numba_average_time/python_average_time:.7} slower than python_color2sepia \n"
        )

    if numba_average_time <= numpy_average_time:
        infile.write(
            f"Average runtime running of numba_color2sepia is {numpy_average_time/numba_average_time:.7} faster than numpy_color2sepia \n"
        )
    else:
        infile.write(
            f"Average runtime running of numba_color2sepia is {numba_average_time/numpy_average_time:.7} slower than numpy_color2sepia \n"
        )
    infile.write(f"Timing performed using: {image.shape} \n")

