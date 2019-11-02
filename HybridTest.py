import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from MyHybridImages import myHybridImages

# Load images
# ----------------------------------------------------------------------------------------------------------------------
low_image = mpimg.imread('./hybrid-images/peter_parker.bmp', format='bmp') / 255.0
high_image = mpimg.imread('./hybrid-images/spiderman.bmp', format='bmp') / 255.0

# Default Sigmas
# ----------------------------------------------------------------------------------------------------------------------
low_sigma = 0.01
high_sigma = 0.01

# Plot
# ----------------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
hybrid = myHybridImages(low_image, low_sigma, high_image, high_sigma)
l = plt.imshow(hybrid)
print(f'{np.min(hybrid), np.max(hybrid)}')
ax.margins(x=0)

ax_low_sigma = plt.axes([0.25, 0.1, 0.65, 0.03])
ax_high_sigma = plt.axes([0.25, 0.15, 0.65, 0.03])

slider_low_sigma = Slider(ax_low_sigma, 'Low Sigma', 0.01, 20.0, valinit=low_sigma)
slider_high_sigma = Slider(ax_high_sigma, 'High Sigma', 0.01, 20.0, valinit=high_sigma)


def update(val):
  low_sigma = slider_low_sigma.val
  high_sigma = slider_high_sigma.val
  hybrid = myHybridImages(low_image, low_sigma, high_image, high_sigma)
  print(f'{np.min(hybrid), np.max(hybrid)}')
  l.set_data(hybrid)
  fig.canvas.draw_idle()


slider_low_sigma.on_changed(update)
slider_high_sigma.on_changed(update)

plt.show()

# ----------------------------------------------------------------------------------------------------------------------
matplotlib.image.imsave('finished-hybrids/new.bmp', myHybridImages(low_image, 6.15, high_image, 1.36))
