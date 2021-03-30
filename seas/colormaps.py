import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import warnings


def get_mpl_colormap(colormap_name):
    '''
    Convert a matplotlib colormap to a cv2-compatible colormap.

    Arguments:
        colormap_name: The name of the matplotlib colormap

    Returns:
        color_range: a 256x1x3 dimensional array holding the 8-bit color map representation compatible with opencv

    Raises:
        ValueError: if colormap_name was invalid
    '''
    # Initialize the matplotlib color map, convert to scalar mappable colormap
    matplotlib_colormap = plt.get_cmap(colormap_name)
    scalarmappable_colormap = plt.cm.ScalarMappable(cmap=matplotlib_colormap)

    # Obtain linear color range
    cv2_colormap = scalarmappable_colormap.to_rgba(np.linspace(0, 1, 256),
                                                   bytes=True)[:, 2::-1]

    return cv2_colormap.reshape(256, 1, 3)


# Functions for colormap manipulation
#------------------------------------


def rescaled_to_dfof(rescaled_value, slope, array_min):
    '''
    Convert a value in 8-bit rescaled units (0-255) to a dfof value.

    Arguments:
        rescaled_value: The rescaled_value in rescaled units to convert
        slope: The slope returned by video.rescale during the dfof to rescaled operation
        array_min: The new minimum value returned by video.rescale during the dfof to rescaled operation

    Returns:
        dfof_value: The rescaled_value converted back to dfof by the scale and array_min parameters.
    '''
    return slope * (rescaled_value - array_min)


def dfof_to_rescaled(dfof_value, slope, array_min):
    '''
    Convert a dfof value to 8-bit rescaled units (0-255).

    Arguments:
        dfof_value: The rescaled_value in dfof units to convert
        slope: The slope returned by video.rescale during the dfof to rescaled operation
        array_min: The new minimum value returned by video.rescale during the dfof to rescaled operation

    Returns:
        rescaled_value: The rescaled_value converted to rescaled units by the scale and array_min parameters.
    '''
    return dfof_value / slope + array_min


def save_colorbar(scale, path, colormap='default'):
    '''
    Save a plt colorbar with a given scale to a specified path.  Accepts plt or cv2 colormap objects.

    Arguments:
        scale: The rescaled_value in dfof units to convert

    Returns:
        rescaled_value: The rescaled_value converted to rescaled units by the scale and array_min parameters.
    '''


    if colormap == 'default':
        colormap = DEFAULT_COLORMAP

    if np.allclose(colormap, DEFAULT_COLORMAP):
        colormap = 'rainbow'
    else:
        colormap = ListedColormap(colormap.squeeze() / 256)

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


def apply_colormap(video, cmap='default'):

    print('\nApplying Color Map to Movie\n-----------------------')

    if colormap == 'default':
        colormap = DEFAULT_COLORMAP

    sz = video.shape
    A2 = np.zeros((sz[0], sz[1], sz[2], 3),
                  dtype='uint8')  #create extra 4th dim for color
    for i in range(sz[0]):
        cv2.applyColorMap(video[i, :, :].astype('uint8'), cmap, A2[i, :, :, :])

    print('\n')
    return A2


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
