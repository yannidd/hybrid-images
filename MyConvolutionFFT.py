import numpy as np


def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
  """
  Convolve an image with a kernel assuming zero-padding of the image to handle the borders

  :param image: the image (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
  :type numpy.ndarray

  :param kernel: the kernel (shape=(kheight,kwidth); both dimensions odd)
  :type numpy.ndarray

  :returns the convolved image (of the same shape as the input image)
  :rtype numpy.ndarray
  """

  is_colour = True if len(image.shape) == 3 else False

  if is_colour:
    padded_image = np.pad(image, ((0, 2 * kernel.shape[0]), (0, 2 * kernel.shape[1]), (0, 0)), mode='constant')
  else:
    padded_image = np.pad(image, ((0, 2 * kernel.shape[0]), (0, 2 * kernel.shape[1])), mode='constant')

  padded_kernel = np.zeros((padded_image.shape[0], padded_image.shape[1]))
  padded_kernel[:kernel.shape[0], :kernel.shape[1]] = kernel

  padded_kernel_spectrum = np.fft.fftn(padded_kernel, axes=(0, 1))
  padded_image_spectrum = np.fft.fftn(padded_image, axes=(0, 1))

  if is_colour:
    convolved_spectrum = padded_image_spectrum * padded_kernel_spectrum[:, :, None]
  else:
    convolved_spectrum = padded_image_spectrum * padded_kernel_spectrum

  shift_start = np.array(kernel.shape) // 2

  if is_colour:
    new_image = np.abs(np.fft.ifftn(convolved_spectrum, axes=(0, 1)))[shift_start[0]:shift_start[0] + image.shape[0],
                                                                      shift_start[1]:shift_start[1] + image.shape[1], :]
  else:
    new_image = np.abs(np.fft.ifftn(convolved_spectrum, axes=(0, 1)))[shift_start[0]:shift_start[0] + image.shape[0],
                                                                      shift_start[1]:shift_start[1] + image.shape[1]]

  return new_image
