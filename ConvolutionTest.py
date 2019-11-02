import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from MyConvolutionFFT import convolve
from MyHybridImages import makeGaussianKernel

np.set_printoptions(precision=3, suppress=True)

image = mpimg.imread('./hybrid-images/bicycle.bmp', format='bmp') / 255.0
kernel = makeGaussianKernel(5)
result = convolve(image, kernel)

print(f'Min: {np.min(result)}\n'
      f'Max: {np.max(result)}')

plt.figure()
plt.imshow(result)

plt.show()
