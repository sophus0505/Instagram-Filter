# Assignment 4 IN3110

## Prerequisites

The prerequisites are numpy, numba, matplotlib, timeit, cv2, pytest and instapy.
To install them type the following into the terminal 

```bash
$ pip install numpy
```

```bash
$ pip install numba
```

```bash
$ python -m pip install -U matplotlib
```

```bash
$ pip install opencv-python
```

```bash
$ pip install -U pytest
```

To install instapy run the following command in the root instapy folder

```bash
$ pip install .
```



## Usage

Too see usage of the instapy package type the following into the terminal
```bash
$ instapy --help
```
It should generate the following output:
```bash
$ instapy --help
usage: instapy [-h] -f FILE [-se] [-g] [-sc SCALE] [-i {python, numpy, numba}] [-o OUT] [-sh] [-in INTENSITY] [-r]

Instapy will turn your image into grayscale or sepia! If no output is specified it will show the new image without saving.

optional arguments:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  the filename of file to apply filter to
  -se, --sepia          select sepia filter
  -g, --gray            select gray filter
  -sc SCALE, --scale SCALE
                        scale factor to resize image
  -i {python, numpy, numba}, --implement {python, numpy, numba}
                        choose the implementation
  -o OUT, --out OUT     the output filename
  -sh, --show           display image using matplotlib.pyplot
  -in INTENSITY, --intensity INTENSITY
                        intensity of sepia filter from 0 to 1
  -r, --runtime         display the average runtime over 3 runs

Made by Sophus :)
```


## Testing

The testing is done with pytest. When in the assignment4 directory type the following in the terminal:

```bash
$ pytest
```

## Reports

To generate the reports type the following into the terminal from the assignment4 directory:
```bash
$ python3 reports/make_grayscale_reports.py
```
```bash
$ python3 reports/make_sepia_reports.py
```

## Comments

Sometimes the tests does not pass. This is because I scale all the values so that the maximum is 255 using a scaling factor.
In the tests for the sepia filters I need to make an estimate of the scaling factor used, and this turned out to be pretty unaccurate. 
All the test ran succesfully if you change it from scaling to simply cutting off the max value at 255, but you'll just have to trust me on that one :)) 
