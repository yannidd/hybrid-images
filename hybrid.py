import math

import numpy as np

from convolution_fft import convolve


def make_hybrid_image(lowImage: np.ndarray, lowSigma: float, highImage: np.ndarray, highSigma: float) -> np.ndarray:
    """
    Create hybrid images by combining a low-pass and high-pass filtered pair.

    :param lowImage: the image to low-pass filter (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
    :type numpy.ndarray

    :param lowSigma: the standard deviation of the Gaussian used for low-pass filtering lowImage
    :type float

    :param highImage: the image to high-pass filter (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
    :type numpy.ndarray

    :param highSigma: the standard deviation of the Gaussian used for low-pass filtering highImage before subtraction to create the high-pass filtered image
    :type float

    :returns returns the hybrid image created
           by low-pass filtering lowImage with a Gaussian of s.d. lowSigma and combining it with
           a high-pass image created by subtracting highImage from highImage convolved with
           a Gaussian of s.d. highSigma. The resultant image has the same size as the input images.
    :rtype numpy.ndarray
    """

    low_kernel = make_gaussian_kernel(lowSigma)
    high_kernel = make_gaussian_kernel(highSigma)

    low_image_lpf = convolve(lowImage, low_kernel)
    high_image_lpf = convolve(highImage, high_kernel)

    high_image_hpf = highImage - high_image_lpf
    hybrid = low_image_lpf + high_image_hpf

    return hybrid


def make_gaussian_kernel(sigma: float) -> np.ndarray:
    """
    Use this function to create a 2D gaussian kernel with standard deviation sigma.
    The kernel values should sum to 1.0, and the size should be floor(8*sigma+1) or
    floor(8*sigma+1)+1 (whichever is odd) as per the assignment specification.
    """

    size = math.floor(8 * sigma + 1)
    size = size + 1 if (size % 2 == 0) else size
    centre = size // 2

    kernel = np.empty((size, size))

    for i in range(size):
        for j in range(size):
            ii = i - centre
            jj = j - centre
            kernel[i, j] = np.exp(- (ii ** 2 + jj ** 2) / (2 * sigma ** 2))

    kernel = kernel / np.sum(kernel)

    return kernel
