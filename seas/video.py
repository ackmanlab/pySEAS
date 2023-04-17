from timeit import default_timer as timer
import tifffile
import warnings
import sys
import numpy as np
from math import ceil
import cv2
import os

from seas.rois import get_masked_region, insert_masked_region
from seas.colormaps import DEFAULT_COLORMAP, save_colorbar

from typing import Tuple, List


def load(pathlist: List[str],
         downsample: int = False,
         t_downsample: int = False,
         dtype: str = None,
         verbose: bool = True,
         tiffloading: bool = True,
         greyscale: bool = True) -> np.ndarray:
    '''
    Loads a list of tiff paths, returns concatenated arrays.
    Implemented size-aware loading for tiff arrays with pre-allocation
    Expects a list of pathnames: 

    Arguments:
        pathlist: List of paths to load.  Can be mixes formats, as long as dimensions match.
            ex: ['./testvideo1.mat', '/home/sydney/testfile2.tif']
        downsample: The spatial factor to downsample by.
        t_downsample: The temporal factor to downsample by.
        dtype: The data type to load into.  Must be a numpy data type string.
        verbose: Whether to print verbose output.
        tiffloading: Whether to preallocate array in memory based on file size expected from each tiff file.
        greyscale: Whether the files are in color or greyscale.

    Returns:

    Files in list must be the same xy dimensions.
    if downsample is an integer greater than one, movie will be downsampled 
    # by that factor.
    '''
    print('\nLoading Files\n-----------------------')

    if type(pathlist) is str:  # If just one path got in.
        pathlist = [pathlist]

    # Make sure pathlist is a list of strings.
    assert type(pathlist) is list

    # Use tiff loading if all paths are tiffs.
    for obj in pathlist:
        assert type(obj) is str
        if not (obj.endswith('.tif') | obj.endswith('.tiff')):
            print('Found non-tiff file to load:', obj)
            tiffloading = False

    # If downsampling, explicitly state spatial and temporal factors.
    if downsample or t_downsample:

        if not t_downsample:
            t_downsample = 1
        if not downsample:
            downsample = 1

        assert type(downsample) is int
        assert type(t_downsample) is int

    # If no datatype was given, assume it's uint16.
    if dtype == None:
        dtype = 'uint16'

    # Use size-aware tiff loading to preallocate matrix and load one at a time.
    if tiffloading:
        # Ignore tiff warning (lots of text, unnecessary info).
        warnings.simplefilter('ignore', UserWarning)
        try:

            print('Using size-aware tiff loading.')
            nframes = 0

            # Loop through tiff files to determine matrix size.
            for f, path in enumerate(pathlist):
                with tifffile.TiffFile(path) as tif:
                    if (len(tif.pages) == 1) and (len(tif.pages[0].shape) == 3):
                        # Sometimes movies save as a single page.
                        pageshape = tif.pages[0].shape[1:]
                        nframes += tif.pages[0].shape[0]

                    else:
                        nframes += len(tif.pages)
                        pageshape = tif.pages[0].shape

                    if f == 0:
                        shape = pageshape
                    else:
                        assert pageshape == shape, \
                            'shape was not consistent for all tiff files loaded'

            shape = (nframes, shape[0], shape[1])
            print('shape:', shape)

            # Resize preallocated matrix if downsampling.
            if downsample:
                shape = (shape[0] // t_downsample, shape[1] // downsample,
                         shape[2] // downsample)
                print('downsample size:', shape, '\n')

                # Initialize remainder for temporal downsampling.
                rmarray = np.zeros(shape=(0, shape[1], shape[2]), dtype='uint8')

            array = np.empty(shape, dtype=dtype)

            # Load video one at a time and assign to preallocated matrix.
            i = 0
            for f, path in enumerate(pathlist):
                t0 = timer()
                print('Loading file:', path)
                with tifffile.TiffFile(path) as tif:
                    if downsample:
                        if downsample > 1:
                            print('\t spatially downsampling by {0}..'.format(
                                downsample))
                        if t_downsample > 1:
                            print('\t temporally downsampling by {0}..'.format(
                                t_downsample))

                        tiff_array = tif.asarray()
                        if rmarray.shape[0] > 0:
                            if verbose:
                                print('concatenating remainder..')
                            tiff_array = np.concatenate((rmarray, tiff_array),
                                                        axis=0)

                        downsampled_array, rmarray = scale_video(tiff_array,
                                                                 downsample,
                                                                 t_downsample,
                                                                 verbose=False,
                                                                 remainder=True)
                        npages = downsampled_array.shape[0]
                        array[i:i + npages] = downsampled_array

                    else:
                        tiff_array = tif.asarray()
                        if tiff_array.ndim == 3:
                            npages = tiff_array.shape[0]
                        else:  # If a movie has only one frame, shape is only 2d.
                            npages = 1
                        array[i:i + npages] = tiff_array

                    i += npages

                print("\t Loading file took: {0} sec".format(timer() - t0))
        except Exception as e:
            # If something failed,  try again without size aware tiff loading.
            print('Size-aware tiff-loading failed!')
            print('\tException:', e)
            tiffloading = False

    # Don't use tiff size-aware loading.  load each file and append to growing matrix.
    if not tiffloading:
        print('Not using size-aware tiff loading.')
        if t_downsample:
            remainder = True
        hdf5_loadkey = None

        # General function for loading a path of any type.
        # Add if/elif statements for more file types
        def loadFile(path, downsample, t_downsample, remainder=None):
            t0 = timer()

            if path.endswith('.tif') | path.endswith('.tiff'):
                print("Loading tiff file at " + path)
                with tifffile.TiffFile(path) as tif:
                    array = tif.asarray()
                    if type(A) is np.memmap:
                        array = np.array(array, dtype=dtype)

            elif path.endswith('.hdf5') | path.endswith('.mat'):
                print("Loading hdf5 file at", path)
                f = h5(path)

                if 'movie' in f.keys():
                    array = f.load('movie')
                elif 'tr' in f.keys():
                    array = f.load('tr')
                else:
                    nonlocal hdf5_loadkey

                    if hdf5_loadkey is None:
                        print("'movie' not found in file keys.")
                        print('Keys in file:')
                        [print('\t', key) for key in f.keys()]

                        while hdf5_loadkey is None:
                            print('Which key do you want to load?')
                            loadinput = input()
                            if loadinput in f.keys():
                                hdf5_loadkey = loadinput
                            else:
                                print('key:', loadinput, 'was not valid.')

                    array = f.load(hdf5_loadkey)

            elif path.endswith('.avi') | path.endswith('.mp4'):
                cap = cv2.VideoCapture(path)

                frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                if greyscale:
                    array = np.empty((frameCount, frameHeight, frameWidth),
                                     np.dtype('uint8'))
                else:
                    array = np.empty((frameCount, frameHeight, frameWidth, 3),
                                     np.dtype('uint8'))

                fc = 0
                ret = True
                while (fc < frameCount and ret):
                    if greyscale:
                        ret, frame = cap.read()
                        array[fc] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    else:
                        ret, array[fc] = cap.read()
                    fc += 1
                cap.release()

            else:
                print('File path is of unknown file type!')
                raise Exception("'{0}' does not have a supported \
                    path extension".format(path))

            if downsample is not False:
                print('\t spatially downsampling by {0}..'.format(downsample))
                print(
                    '\t temporally downsampling by {0}..'.format(t_downsample))
                if remainder is not None:
                    if (type(remainder) == np.ndarray) and \
                            (remainder.shape[0] != 0):
                        array = np.concatenate((remainder, array), axis=0)
                    A, remainder = scale_video(A,
                                               downsample,
                                               t_downsample,
                                               verbose=True,
                                               remainder=True)
                else:
                    array = scale_video(A,
                                        downsample,
                                        t_downsample,
                                        verbose=True)

            print("Loading file took: {0} sec".format(timer() - t0))

            if remainder is not None:
                return remainder, array
            else:
                return array

        # Load either one file, or load and concatenate list of files.
        if len(pathlist) == 1:
            array = loadFile(pathlist[0], downsample, t_downsample)
        else:
            remainder = None
            for i, path in enumerate(pathlist):
                if t_downsample > 1:
                    remainder, temporary_array = loadFile(path,
                                                          downsample,
                                                          t_downsample,
                                                          remainder=remainder)
                else:
                    temporary_array = loadFile(path, downsample, t_downsample,
                                               remainder)

                t0 = timer()
                if i == 0:
                    array = temporary_array
                else:
                    try:
                        array = np.concatenate([A, temporary_array], axis=0)
                    except:
                        array = np.concatenate([A, temporary_array[None, :, :]],
                                               axis=0)

                print("Concatenating arrays took: {0} sec\n".format(timer() -
                                                                    t0))

    return array


def save(array: np.ndarray,
         path: str,
         resize_factor: int = 1,
         apply_cmap: bool = True,
         rescale_range: bool = False,
         colormap: np.ndarray = 'default',
         speed: int = 1,
         fps: int = 10,
         codec: str = None,
         mask: np.ndarray = None,
         overlay: np.ndarray = None,
         overlay_color: Tuple[int, int, int] = (0, 0, 0),
         save_cbar: bool = False) -> None:
    '''
    Check what the extension of path is, and use the appropriate function
    for saving the array.  Functionality can be added for more 
    file/data types.


    Arguments:
        array: The (x,y,t) or (x,y,c,t) array to save to a video.
        path: Where to save the file to.
        resize_factor: A spatial resize factor to apply when saving
        apply_cmap: Whether to apply a colormap.  Defaults to True.
        rescale_range: Whether to rescale the dyanmic range of the video when saving.  Defaults to True.
        colormap: The colormap to apply.  Defaults to the global default colormap.
        speed: The speed factor to save the video at.  Multiplies the fps of the video.
        fps: The video fps to save at.
        codec: The video codec to use.  See the note below for more information.
        mask: The mask to apply when saving.  If selected, the areas where mask == 0 will be black.
        overlay: An overlay to draw on top of the video when saving.
        overlay_color: The color to save the overlay as.
        save_cbar: Whether to save an additional figure showing the color bar of the colormap applied to the video.

    Returns:
        None

    
    A note on codecs:
    For AVIs:
    Parameters (args 3+) are used for creating an avi output.  
    MJPG and XVID codecs seem to work well for linux systems, 
    so they are set as the default.
    A full list of codecs could be found at:
    http://www.fourcc.org/codecs.php.
    '''
    print('\nSaving File\n-----------------------')
    assert (type(array) == np.ndarray), ('Movie to save was not a '
                                         'numpy array')

    if colormap == 'default':
        colormap = DEFAULT_COLORMAP

    if path.endswith('.tif') | path.endswith('.tiff'):
        print('Saving to: ' + path)
        t0 = timer()
        if array.shape[2] == 3:
            # Convert RGB to BGR.
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
            print('Converted RGB image to BGR.')
        with tifffile.TiffWriter(path) as tif:
            tif.save(array)
        print("Save file: {0} sec\n".format(timer() - t0))
        save_cbar = False

    elif path.endswith('.png'):
        assert array.ndim <= 3, 'File was not an image'

        if rescale_range:
            array, scale = rescale(array, return_scale=True)

        array = array.astype('uint8')

        if apply_cmap and (array.ndim == 2):
            array = cv2.applyColorMap(array.copy(), colormap)

        if mask is not None:
            ind = np.where(mask == 0)

            if ind[0].size > 0:
                alpha = np.ones(
                    (array.shape[0], array.shape[1]), dtype='uint8') * 255

                alpha[ind] = 0
                array = np.concatenate((array, alpha[:, :, None]), axis=2)

        cv2.imwrite(path, array)

    elif path.endswith('.avi') | path.endswith('.mp4'):
        sz = array.shape

        if rescale_range:
            array, scale = rescale(array, return_scale=True)

        array = array.astype('uint8')

        if codec == None:  # No codec specified.
            if path.endswith('.avi'):
                codec = 'MJPG'
            elif path.endswith('.mp4'):
                if os.name == 'posix':
                    codec = 'X264'
                elif os.name == 'nt':
                    codec = 'XVID'
                else:
                    raise TypeError('Unknown os type: {0}'.format(os.name))

        # Check codec and dimensions.
        if array.ndim == 3:
            if apply_cmap == False:
                sz = array.shape
                array = array.reshape(sz[0], sz[1] * sz[2])
                array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
                array = array.reshape((sz[0], sz[1], sz[2], 3))
                movietype = 'black and white'
            else:
                movietype = 'color'

        elif array.ndim == 4:
            apply_cmap = False
            movietype = 'color'
        else:
            raise Exception('Input matrix was {0} dimensions. videos '
                            'cannot be written in this format.'.format(
                                array.ndim))

        print('Movie will be written in {0} using the {1} codec'.format(
            movietype, codec))
        print('Saving to: ' + path)

        # Set up resize.
        w = int(ceil(sz[1] * resize_factor))
        h = int(ceil(sz[2] * resize_factor))

        # Initialize movie writer.
        display_speed = fps * speed
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(path, fourcc, display_speed, (h, w), True)

        if overlay is not None:
            if resize_factor != 1:
                overlay = cv2.resize(overlay, (h, w),
                                     interpolation=cv2.INTER_AREA)
            overlayindices = np.where(overlay > 0)

        for i in range(sz[0]):
            frame = array[i].copy().astype('uint8')

            if (resize_factor != 1):
                frame = cv2.resize(frame, (h, w), interpolation=cv2.INTER_AREA)

            if apply_cmap:
                frame = cv2.applyColorMap(frame, colormap)

            if overlay is not None:
                frame[overlayindices[0], overlayindices[1], :] = overlay_color

            out.write(frame)

        out.release()

    else:
        print('Save path is of unknown file type!')
        raise Exception("'{0}' does not have a supported \
            path extension".format(path))

    print('File saved to:' + path)

    if save_cbar and rescale_range:
        cbarpath = os.path.splitext(path)[0] + '_colorbar.pdf'
        print('Saving Colorbar to:' + cbarpath)
        save_colorbar(scale, cbarpath, colormap=colormap)


def dfof(array, win_size: int = None, win_type: str = 'box'):
    '''
    Calculates the change in fluorescence over mean 
    fluorescense for a video.  If no win_size is given, 
    it will calculate the dfof using an average projection of the whole video.
    Define window size and window type from (see function from 
    time course analysis) to calculate a rolling average dfof.

    Arguments:
        array: The input (x,y,t or xy,t) array to calculate dfof of.
        win_size: The temporal window size to calculate rolling mean effects, if applicable.
        win_type: The type of windowing effect applied to the movie. 

    Returns:
        array: The array after dfof is applied.
    '''

    assert (type(array) == np.ndarray), 'Input was not a numpy array'
    if array.ndim == 3:
        reshape = True
        sz = array.shape
        array = np.reshape(array, (sz[0], int(array.size / sz[0])))
    elif array.ndim == 2:
        reshape = False
    else:
        assert array.ndim == 1, (
            'Input was not 1-3 dimensional: {0} dim'.format(array.ndim))
        reshape = False
        array = array[:, None]
        print('Reshaped to two dimensional.')

    print("Array Shape (t,xy): {0}".format(array.shape))
    print('Array Type:', array.dtype)

    t0 = timer()
    if win_size == None:
        print('\nCalculating dF/F\n-----------------------')
        print('Calculating dFoF based on average projection')
        array_mean = np.nanmean(array, axis=0, dtype='float32')
        print("z mean: {0} sec".format(timer() - t0))
        print("Array mean shape (xy): {0}".format(array_mean.shape))
        print("Array mean type: {0}".format(array_mean.dtype))
    else:
        print('\nCalculating rolling dF/F\n-----------------------')
        lower_bound = int(win_size // 2)
        upper_bound = int(win_size - lower_bound)
        window = tca.windowFunc(width=win_size, win_type=win_type)[:, None]
        window = np.repeat(window, array.shape[1], axis=1)
        print("Upper bound frame: ", upper_bound, "Lower bound frame: ",
              lower_bound)
        print("Window shape (t,x*y): ", window.shape)

    t0 = timer()
    array = array.astype('float32', copy=False)
    print("float32: {0} sec".format(timer() - t0))

    t0 = timer()
    t1 = timer()
    if win_size != None:
        moving_array = array.copy()
        for i in np.arange(array.shape[0]):
            if i < lower_bound:
                new_window = True
                window = tca.windowFunc(width=upper_bound + i,
                                        win_type=win_type)[:, None]
                window = np.repeat(window, array.shape[1], axis=1)
                wind_product = window * array[:i + upper_bound, :]
            elif i > array.shape[0] - upper_bound:
                new_window = True
                window = tca.windowFunc(width=lower_bound + array.shape[0] - i,
                                        win_type=win_type)[:, None]
                window = np.repeat(window, array.shape[1], axis=1)
                wind_product = window * array[i - lower_bound:, :]
            else:
                if new_window:
                    window = tca.windowFunc(width=win_size,
                                            win_type=win_type)[:, None]
                    window = np.repeat(window, array.shape[1], axis=1)
                    new_window = False
                wind_product = np.multiply(
                    window, array[i - lower_bound:i + upper_bound, :])

            array_mean = np.nansum(wind_product, axis=0, dtype='float32')

            if timer() - t1 > 300:
                t1 = timer()
                print(
                    '\tWorking on {0} frame, time passed: {1:.1f} secs'.format(
                        i,
                        timer() - t0))
            moving_array[i, :] /= array_mean
            moving_array[i, :] -= 1.0
    else:
        for i in np.arange(array.shape[0]):
            array[i, :] /= array_mean
            array[i, :] -= 1.0

    print("dfof normalization: {0} sec".format(timer() - t0))
    if reshape:
        if win_size != None:
            array = np.reshape(moving_array, sz)
        else:
            array = np.reshape(array, sz)
    print("Array type: {0}".format(array.dtype))
    print("Array shape (t,x,y): {0}\n".format(array.shape))

    return array


def rescale(array: np.ndarray,
            low: float = 3,
            high: float = 7,
            cap: bool = True,
            mean_std: Tuple[float, float] = None,
            mask: np.ndarray = None,
            maskval: float = 1,
            verbose: bool = True,
            min_max: bool = None,
            return_scale: bool = False) -> Tuple[np.ndarray, dict]:
    '''
    determine upper and lower limits of colormap for playing movie files. 
    limits based on standard deviation from mean.  low, high are defined 
    in terms of standard deviation.  Image is updated in-place, 
    and doesn't have to be returned.

    Arguments:
        array: The array to rescale.
        low: The number of standard deviations below to scale the range to.
        high: The number of standard deviations above to scale the range to.
        cap: If the dynamic range would be reduced, 
            instead cap range to the minimum/maximum values present within the array.
        mean_std: Use a given mean and standard deviation to apply the transformation, 
            rather than gathering from the video.
        mask: A mask to apply to the video when calculating the min/max range to scale by.
        maskval: The value of the mask to include as signal.
        verbose: Whether to print a verbose output.
        min_max: Apply a specific min/max as the range, rather than calculating dyamically.
        return_scale: Whether to return the resize scale which was applied.

    Returns:
        array: The rescaled array.
            The original array should be updated in place as well.
        scale: A dictionary containing all the scaling parameters.
            Only returned if return_scale = True.
    '''

    if verbose:
        print('\nRescaling Movie\n-----------------------')

    # Mask the region if mask provided.
    if mask is not None:
        copy = array.copy()
        array = get_masked_region(array, mask, maskval)

    if np.isnan(array).any():
        array[np.where(np.isnan(array))] = 0

    # If unmasked color image, add extra temporary first dimension.
    if array.ndim == 3:
        if array.shape[2] == 3:
            array = array[None, :]

    if min_max is None:
        if mean_std is None:
            mean = np.nanmean(array, dtype=np.float64)
            std = np.nanstd(array, dtype=np.float64)
        else:
            mean = mean_std[0]
            std = mean_std[1]

        new_min = mean - low * std
        new_max = mean + high * std
        if verbose:
            print('mean:', mean, 'low:', low, 'high:', high, 'std:', std)
    else:
        assert len(min_max) == 2
        new_min = min_max[0]
        new_max = min_max[1]

    if verbose:
        print('new_min:', new_min)
        print('new_max', new_max)

    if cap:  # Don't reduce dynamic range.
        if verbose:
            print('array min', np.nanmin(array))
        if new_min < array.min():
            new_min = array.min()
            if verbose:
                print('array min scaled', new_min)

        if verbose:
            print('amax', np.nanmax(array))
        if new_max > array.max():
            new_max = array.max()
            if verbose:
                print('amax scaled', new_max)

    new_slope = 255.0 / (new_max - new_min)
    if verbose:
        print('new_slope:', new_slope)
    array = array - new_min
    array = array * new_slope

    if mask is not None:
        array = insert_masked_region(copy, A, mask, maskval)

    array[np.where(array > 255)] = 255
    array[np.where(array < 0)] = 0

    if array.shape[0] == 1:  # If was converted to one higher dimension.
        array = array[0, :]

    if verbose:
        print('\n')

    if not return_scale:
        return array
    else:
        return array, {'scale': new_slope, 'min': new_min, 'max': new_max}


def play(array: np.ndarray,
         textsavepath: str = None,
         min_max: Tuple[float, float] = None,
         preprocess: bool = True,
         overlay: np.ndarray = None,
         min_max_toolbars: bool = False,
         rescale_movie: bool = True,
         cmap: np.ndarray = 'default',
         loop: bool = True) -> None:
    '''
    Play movie in opencv after normalizing display range.

    Arguments:
        array: The (x,y,t or x,y,c,t) array to play as a movie.
        textsavepath: An output to save time points to.
        min_max: A specific min max value to scale the video to.
        preprocess: Whether to preprocess the video, or calculate ranges dynamically.
        overlay: An overlay frame to put on top of the video.
        min_max_toolbars: Whether to display min/max slider toolbars for adjusting color range.
        rescale_movie: Whether to rescale the movie.
        cmap: The color map to apply.
        loop: Whether to loop the video.

    Returns:
        None.

    Note: if preprocess is set to true, the array normalization is done 
    in place, thus the array will be rescaled outside scope of 
    this function.
    '''
    if colormap == 'default':
        colormap = DEFAULT_COLORMAP

    sz = array.shape

    def frameWrite(w, savepath=textsavepath):
        update = open(savepath, 'a', buffering=1)
        update.write(w)
        update.flush()
        update.close()

    if textsavepath is not None:
        frameWrite(w='Start index, End index', savepath=textsavepath)

    if overlay is not None:
        o_values = np.unique(overlay)
        for val in o_values:
            if val not in [0, 1, 255]:
                raise AssertionError('Overlay must be a 3d binary image.')
        # Create negative image.
        overlay = overlay == 0
        overlay = overlay.astype(np.uint8)

    print('\nPlaying Movie\n-----------------------')
    assert (type(array) == np.ndarray), 'array was not a numpy array'
    assert (array.ndim == 3) | (array.ndim == 4), ('array was not three or '
                                                   'four-dimensional array')

    windowname = "Press Esc to Close"
    cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
    print('preprocessing data...')
    # Create a resizable window.

    sz = array.shape

    if array.ndim == 3:
        if min_max == None:
            # if min/max aren't set, default to 3 and 7
            lowB = [3]
            highB = [7]
        else:
            # otherwise, use values given
            lowB = min_max[0]
            highB = min_max[1]

        if rescale_movie == False:
            # if the movie shouldn't be rescaled, don't display rescaling toolbars
            min_max_toolbars = False
        else:
            # mean, std not required if not rescaling
            mean = np.nanmean(A, dtype=np.float64)
            std = np.nanstd(A, dtype=np.float64)

        if preprocess:
            print('Pre-processing movie rescaling...')
            #Normalize movie range and change to uint8 before display
            if min_max_toolbars:
                imgclone = array.copy()
            t0 = timer()
            array = np.reshape(array, (sz[0], int(array.size / sz[0])))
            array = rescale(array,
                            low=lowB[0],
                            high=highB[0],
                            mean_std=(mean, std))

            array = np.reshape(array, sz)
            array = array.astype('uint8', copy=False)
            print("Movie range normalization: {0}".format(timer() - t0))

        if min_max_toolbars:

            def updateColormap(array):
                lowB[0] = 0.5 * (8 -
                                 cv2.getTrackbarPos("Low Limit", windowname))
                highB[0] = 0.5 * cv2.getTrackbarPos("High Limit", windowname)

                if preprocess:
                    array = imgclone.copy()
                    array = rescale(array,
                                    low=lowB[0],
                                    high=highB[0],
                                    mean_std=(mean, std))
                return

            cv2.createTrackbar("Low Limit", windowname, (-2 * lowB[0] + 8), 8,
                               lambda e: updateColormap(array))
            cv2.createTrackbar("High Limit", windowname, (2 * highB[0]), 16,
                               lambda e: updateColormap(array))

    i = 0
    toggleNext = True
    tf = True
    zoom = 1
    zfactor = 5 / 4
    firstwrite = True

    print('starting video playback..')

    while True:
        im = np.copy(array[i])
        if zoom != 1:
            im = cv2.resize(im,
                            None,
                            fx=1 / zoom,
                            fy=1 / zoom,
                            interpolation=cv2.INTER_CUBIC)

        if A.ndim == 3:
            if (preprocess != True) & (rescale_movie == True):
                im = rescale(im,
                             low=lowB[0],
                             high=highB[0],
                             mean_std=(mean, std),
                             verbose=False,
                             cap=False)
            color = np.zeros((im.shape[0], im.shape[1], 3))
            color = cv2.applyColorMap(im.astype('uint8'), cmap, color)
            if overlay is not None:
                im *= overlay  # Black:
            cv2.putText(color, str(i), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255))  # Draw frame text.
            cv2.imshow(windowname, color)

        elif A.ndim == 4:
            if overlay is not None:
                im *= overlay
            cv2.putText(im, str(i), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255))  #D raw frame text.
            cv2.imshow(windowname, im.astype('uint8'))

        k = cv2.waitKey(10)
        if k == 27:  # If esc is pressed.
            break
        elif (k == ord(' ')) and (toggleNext == True):
            tf = False
        elif (k == ord(' ')) and (toggleNext == False):
            tf = True
        toggleNext = tf  # Toggle the switch.

        if k == ord("="):
            zoom = zoom * 1 / zfactor
        if k == ord("-"):
            zoom = zoom * zfactor

        if k == ord('b') and toggleNext:
            i -= 100
        elif k == ord('f') and toggleNext:
            i += 100
        elif k == ord('m') and (toggleNext == False):
            i += 1
        elif k == ord('n') and (toggleNext == False):
            i -= 1
            if i == -1:
                i = sz[0] - 1

        elif k == ord('j'):
            if textsavepath is not None:
                print('Start index: ', i)
                frameWrite(w='\n{}'.format(i), savepath=textsavepath)
            else:
                print(
                    'Savefile flag was not given.\nRestart with the save flag if you would like to save the indices'
                )
        elif k == ord('k'):
            if textsavepath is not None:
                print('End index: ', i)
                frameWrite(w=',{}'.format(i), savepath=textsavepath)
            else:
                print(
                    'Savefile flag was not given.\nRestart with the save flag if you would like to save the indices'
                )
        elif k == ord('g'):
            ind = input('Which index would you like to jump to? ')
            ind = int(''.join(i for i in ind if i.isdigit()))
            if ind > -1 and ind < sz[0]:
                i = ind
            else:
                print('Not a valid index. Try again.')

        elif toggleNext:
            i += 1

        if (i > (A.shape[0] - 1)) or (i < -1):
            # Reset to 0 if looping, otherwise break the while loop.
            if loop:
                i = 0
            else:
                break

    cv2.destroyAllWindows()
    for i in range(5):
        cv2.waitKey(1)

    print('\n')

    if min_max_toolbars:
        return (lowB[0], highB[0])


def scale_video(array: np.ndarray,
                s_factor: int = 1,
                t_factor: int = 1,
                verbose: bool = True,
                remainder: np.ndarray = False) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Scale a video according to spatial and temporal resizing factors.

    Arguments:
        array: The array to rescale.
        s_factor: The spatial factor to downsize by.
        t_factor: The temporal factor to downsize by.
        verbose: Whether to produce verbose print statements.
        remainder: Whether to return a temporal remainder array.

    Returns:
        dsarray: The downsampled array.
        remainder: A tempora remainder, 
            in the dimensions and scale of the orignal video.
    '''

    if verbose:
        print('\nRescaling Video\n-----------------------')
    assert array.ndim == 3, 'Input was not a video'
    shape = array.shape

    if s_factor == None:
        s_factor = 1
    if t_factor == None:
        t_factor = 1

    if s_factor == 0:
        print('WARNING: Spatial scaling factor was 0.  Reverting to 1.')
        s_factor = 1
    if t_factor == 0:
        print('WARNING: Temporaldef  scaling factor was 0.  Reverting to 1.')
        t_factor = 1

    if (s_factor == 1) and (t_factor == 1):
        print('No downsample factor given.')
        print('Returning array.')

        if remainder:
            return array, np.zeros(shape=(0, shape[1], shape[2]),
                                   dtype=array.dtype)
        else:
            return array

    newshape = (shape[0] // t_factor, shape[1] // s_factor,
                shape[2] // s_factor)
    cropshape = (newshape[0] * t_factor, newshape[1] * s_factor,
                 newshape[2] * s_factor)

    if cropshape != array.shape:
        if remainder:
            rmarray = array[cropshape[0]:]

        if verbose:
            print('Cropping', array.shape, 'to', cropshape)
        if verbose and remainder:
            print('remainder array:', rmarray.shape)

        array = array[:cropshape[0], :cropshape[1], :cropshape[2]]

    elif remainder:
        rmarray = np.zeros(shape=(0, cropshape[1], cropshape[2]),
                           dtype=array.dtype)

    if verbose:
        print('Spatially downsampling by:', s_factor)
    if verbose:
        print('Temporally downsampling by:', t_factor)

    dsarray = downsample(array, newshape)
    if verbose:
        print('New shape:', dsarray.shape)

    if remainder:
        return dsarray, rmarray
    else:
        return dsarray


def downsample(array: np.ndarray,
               new_shape: Tuple[int, int, int],
               keepdims: bool = False) -> np.ndarray:
    '''
    Reshape m by n matrix by factor f by reshaping matrix into
    m f n f matricies, then applying sum across mxf, nxf matrices
    if keepdims, video is downsampled, but number of pixels remains the same.

    The scale_video function is an application of this function, 
    is more user friendly, and should be used in most cases.

    Arguments:
        array: The array to downsample.
        new_shape: The new array shape to convert array to.
        keepdims: Whether to keep the orignal dimensions.  
            Only useful to differentiating whether downsample 
            differences are just due to number of samples.

    Returns:
        array: The downsampled array.
    '''

    if array.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(
            array.shape, new_shape))

    array_shape = array.shape

    compression_pairs = [(d, c // d) for d, c in zip(new_shape, array_shape)]
    flattened_pairs = [l for p in compression_pairs for l in p]
    array = array.reshape(flattened_pairs)

    if not keepdims:
        for i in range(len(new_shape)):
            ax = -1 * (i + 1)
            array = array.mean(ax)

    else:
        for i in range(len(new_shape)):
            ax = -1 * (2 * i + 1)
            n = array.shape[ax]
            array = array.mean(ax, keepdims=True)
            array = np.repeat(array, n, axis=ax)

        array = array.reshape(array_shape)

    return array


def rotate(array: np.ndarray, n: int) -> np.ndarray:
    '''
    Rotate an image or video n times counterclockwise.

    Arguments:
        array: The array to rotate.
        n: The number of counterclockwise rotations.

    Returns:
        array: The array after rotation.
    '''

    assert type(array) == np.ndarray

    if n > 0:
        ndim = array.ndim

        if ndim == 3:
            array = np.rot90(array, n, axes=(1, 2))

        elif (ndim == 2) or (ndim == 3):
            array = np.rot90(array, n)

        else:
            raise TypeError('Input of dimension {0} was invalid'.format(n))
    return array
