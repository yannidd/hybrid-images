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

  kernel_size = kernel.shape[0]
  new_image = np.empty(image.shape, image.dtype)
  is_colour = True if len(image.shape) == 3 else False
  padding_size = np.array(kernel.shape) // 2

  kernel = np.flip(kernel)

  if is_colour:
    # Add padding
    kernel_colour = kernel[:, :, None]
    padded_image = np.pad(image,
                          ((kernel.shape[0] // 2, kernel.shape[0] // 2),
                           (kernel.shape[1] // 2, kernel.shape[1] // 2),
                           (0, 0)),
                          mode='constant')

    # Convolve
    for i in range(image.shape[0]):
      for j in range(image.shape[1]):
        product = padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1]] * kernel_colour
        summation = np.sum(product, axis=(0, 1))
        new_image[i, j] = summation
  else:
    # Add padding
    padded_image = np.pad(image,
                          ((kernel.shape[0] // 2, kernel.shape[0] // 2),
                           (kernel.shape[1] // 2, kernel.shape[1] // 2)),
                          mode='constant')

    # Convolve
    for i in range(image.shape[0]):
      for j in range(image.shape[1]):
        product = np.multiply(padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1]], kernel)
        summation = np.sum(product)
        new_image[i, j] = summation

  return new_image
