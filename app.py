import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from hybrid import make_hybrid_image


def run_interactive_plot(low_image, high_image):
    # Default Sigmas
    low_sigma = 0.01
    high_sigma = 0.01

    # Plot
    fig, ax = plt.subplots()
    fig.canvas.set_window_title('hybrid images - interactive mode')
    plt.subplots_adjust(left=0.25, bottom=0.25)
    hybrid = make_hybrid_image(low_image, low_sigma, high_image, high_sigma)
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
        hybrid = make_hybrid_image(low_image, low_sigma, high_image, high_sigma)
        print(f'{np.min(hybrid), np.max(hybrid)}')
        l.set_data(hybrid)
        fig.canvas.draw_idle()

    slider_low_sigma.on_changed(update)
    slider_high_sigma.on_changed(update)

    plt.show()


def main():
    # Load images.
    low_image = mpimg.imread('./img/dog.bmp', format='bmp') / 255.0
    high_image = mpimg.imread('./img/cat.bmp', format='bmp') / 255.0

    run_interactive_plot(low_image, high_image)


if __name__ == '__main__':
    main()
