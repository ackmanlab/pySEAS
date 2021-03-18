import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import warnings


def get_mpl_colormap(cmap_name):
    assert float(
        cv2.__version__[:3]) > 3.3, 'OpenCV version {0} not compatible with \
vector colormaps.\n\t(Must be > 3.3)'.format(cv2.__version__)

    ## from: https://stackoverflow.com/questions/52498777/apply-matplotlib-or-custom-colormap-to-opencv-image
    cmap = plt.get_cmap(cmap_name)
    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)
    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
    return color_range.reshape(256, 1, 3)


# Get or set default parameters.
#------------------------------------

try:
    DEFAULT_COLORMAP = get_mpl_colormap('rainbow')
    COMPONENT_COLORMAP = get_mpl_colormap('coolwarm')

except Exception as e:

    message = '''Encountered error when converting mpl colormap.
    Error: {0}'''.format(e)

    warnings.warn(message)
    warnings.warn(
        'Reverting to similar cv2 colormaps, instead of mpl equivalents.')

    DEFAULT_COLORMAP = cv2.COLORMAP_JET
    COMPONENT_COLORMAP = cv2.COLORMAP_JET

# colormap for assigning different brain regions to custom pastel cmap
# motor
# somatosensory
# auditory
# visual
# retrosplenial
# olfactory bulb
REGION_CM_COLORS = np.array([[123, 219, 148, 255], [255, 178, 240, 255],
                             [153, 85, 255, 255], [102, 191, 213, 255],
                             [183, 200, 196, 255], [190, 237, 232, 255]]) / 255

REGION_CM = ListedColormap(REGION_CM_COLORS)

# Functions for colormap manipulation
#------------------------------------


def rescaletodfof(value, scale, amin):
    return scale * (value - amin)


def dfoftorescale(value, scale, amin):
    return value / scale + amin


def save_colorbar(scale, path, colormap=DEFAULT_COLORMAP):
    if np.allclose(colormap, DEFAULT_COLORMAP):
        colormap = 'rainbow'
    else:
        colormap = ListedColormap(DEFAULT_COLORMAP.squeeze() / 256)

    ticks = np.linspace(scale['min'], scale['max'], 5).round(4)
    # print(ticks)

    plt.figure(figsize=(1, 2))
    plt.imshow([[0, 0], [0, 0]], cmap=colormap)

    cb = plt.colorbar()
    cb.ax.set_yticklabels(ticks)

    plt.cla()
    plt.axis('off')

    plt.savefig(path, bbox_inches='tight')
    plt.close()


def apply_colormap(video, cmap=DEFAULT_COLORMAP):

    print('\nApplying Color Map to Movie\n-----------------------')

    sz = video.shape
    A2 = np.zeros((sz[0], sz[1], sz[2], 3),
                  dtype='uint8')  #create extra 4th dim for color
    for i in range(sz[0]):
        cv2.applyColorMap(video[i, :, :].astype('uint8'), cmap, A2[i, :, :, :])

    print('\n')
    return A2
