import math
import unittest

import numpy as np
from scipy import ndimage

# from MyConvolutionFFT import convolve as convolve_fft
from MyConvolution import convolve as convolve_fft
from MyHybridImages import makeGaussianKernel

"""
Your implementation must support arbitrary shaped kernels, as long as both dimensions are odd (e.g. 7x9 kernels but not 
4x5 kernels). 

You should utilise (possibly implicit) zero-padding of the input image to ensure that the output image retains the same 
size as the input image and that the kernel can reach into the image edges and corners. 

The implementation must also support convolution of both grey-scale and colour images. 

Note that colour convolution is achieved by applying the convolution operator to each of the colour bands separately 
(i.e. treating each band as an independent grey-level image).
"""


class TestGaussianKernel(unittest.TestCase):
  def test_shape(self):
    sigmas = range(1, 10)
    for sigma in sigmas:
      gaussian_kernel = makeGaussianKernel(sigma)

      size = math.floor(8 * sigma + 1)
      size = size + 1 if (size % 2 == 0) else size

      self.assertEqual(gaussian_kernel.shape[0], size)
      self.assertEqual(gaussian_kernel.shape[1], size)

  def test_sum(self):
    self.assertTrue(np.allclose(1, np.sum(makeGaussianKernel(0.1))))
    self.assertTrue(np.allclose(1, np.sum(makeGaussianKernel(0.5))))
    self.assertTrue(np.allclose(1, np.sum(makeGaussianKernel(0.9))))
    self.assertTrue(np.allclose(1, np.sum(makeGaussianKernel(1.5))))

  def test_kernel(self):
    sigma = 0.84089642

    template = np.array(
      [[0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00019117, 0.00002292, 0.00000067],
       [0.00002292, 0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633, 0.00002292],
       [0.00019117, 0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965, 0.00019117],
       [0.00038771, 0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373, 0.00038771],
       [0.00019117, 0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965, 0.00019117],
       [0.00002292, 0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633, 0.00002292],
       [0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00019117, 0.00002292, 0.00000067]]
    )

    generated = makeGaussianKernel(sigma)

    self.assertTrue(np.allclose(template, generated))

  def test_max_location(self):
    for sigma in np.arange(0.1, 2, 0.25):
      kernel = makeGaussianKernel(sigma)
      centre = kernel.shape[0] // 2
      max_location = np.unravel_index(np.argmax(kernel, axis=None), kernel.shape)
      self.assertTrue(max_location[0] == centre and max_location[1] == centre)


class TestConvolution(unittest.TestCase):
  def test_scipy_similarity_bw(self):
    images = [
      np.random.rand(10, 10),
      np.random.rand(1, 1),
      np.random.rand(15, 100),
    ]
    kernels = [
      # np.random.rand(5, 7),
      # np.random.rand(3, 3),
      # np.random.rand(10, 10),
      # np.random.rand(1, 1),
      makeGaussianKernel(1)
    ]

    for image in images:
      for kernel in kernels:
        convolved_sp = ndimage.convolve(image, kernel, mode='constant', cval=0)
        convolved_my = convolve_fft(image, kernel)
        self.assertTrue(np.allclose(convolved_my, convolved_sp) and np.allclose(convolved_my, convolved_sp))

  def test_scipy_similarity_colour(self):
    images = [
      np.random.rand(10, 10, 3),
      np.random.rand(1, 1, 3),
      np.random.rand(15, 100, 3),
    ]
    kernels = [
      np.random.rand(5, 7),
      np.random.rand(3, 3),
      np.random.rand(10, 10),
      np.random.rand(1, 1),
    ]

    for image in images:
      for kernel in kernels:
        convolved_sp = np.stack((ndimage.convolve(image[:, :, 0], kernel, mode='constant', cval=0),
                                 ndimage.convolve(image[:, :, 1], kernel, mode='constant', cval=0),
                                 ndimage.convolve(image[:, :, 2], kernel, mode='constant', cval=0),),
                                axis=-1)
        convolved_my = convolve_fft(image, kernel)
        self.assertTrue(np.allclose(convolved_my, convolved_sp) and np.allclose(convolved_my, convolved_sp))


if __name__ == '__main__':
  unittest.main()
