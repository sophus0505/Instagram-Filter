from distutils.core import setup
import setuptools

setup(
     name="instapy", 
     version="1.0", 
     packages=["instapy",],
     scripts=['bin/instapy'],
     install_requires=['numpy', 'numba', 'opencv-python', 'matplotlib', 'pytest'],
     python_requires='>=3.6',
)










