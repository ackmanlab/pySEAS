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


def load(pathlist,
         downsample=False,
         t_downsample=False,
         dtype=None,
         verbose=True,
         tiffloading=True,
         greyscale=True):
    '''
    Loads a list of tiff paths, returns concatenated arrays.
    Implemented size-aware loading for tiff arrays with pre-allocation
    Expects a list of pathnames: 
    ex: ['./testvideo1.mat', '/home/sydney/testfile2.tif']
    Files in list must be the same xy dimensions.
    if downsample is an integer greater than one, movie will be downsampled 
    # by that factor.
    '''
    print('\nLoading Files\n-----------------------')

    if type(pathlist) is str:  # if just one path got in
        pathlist = [pathlist]

    # make sure pathlist is a list of strings
    assert type(pathlist) is list

    # use tiff loading if all paths are tiffs.
    for obj in pathlist:
        assert type(obj) is str
        if not (obj.endswith('.tif') | obj.endswith('.tiff')):
            print('Found non-tiff file to load:', obj)
            tiffloading = False

    # if downsampling, explicitly state spatial and temporal factors.
    if downsample or t_downsample:

        if not t_downsample:
            t_downsample = 1
        if not downsample:
            downsample = 1

        assert type(downsample) is int
        assert type(t_downsample) is int

    # if no datatype was given, assume it's uint16
    if dtype == None:
        dtype = 'uint16'

    # use size-aware tiff loading to preallocate matrix and load one at a time
    if tiffloading:
        # ignore tiff warning (lots of text, unnecessary info)
        warnings.simplefilter('ignore', UserWarning)
        try:

            print('Using size-aware tiff loading.')
            nframes = 0

            # loop through tiff files to determine matrix size
            for f, path in enumerate(pathlist):
                with tifffile.TiffFile(path) as tif:
                    if (len(tif.pages) == 1) and (len(tif.pages[0].shape) == 3):
                        # sometimes movies save as a single page
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

            # resize preallocated matrix if downsampling
            if downsample:
                shape = (shape[0] // t_downsample, shape[1] // downsample,
                         shape[2] // downsample)
                print('downsample size:', shape, '\n')

                # initialize remainder for temporal downsampling
                rmarray = np.zeros(shape=(0, shape[1], shape[2]), dtype='uint8')

            A = np.empty(shape, dtype=dtype)

            # load video one at a time and assign to preallocated matrix
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

                        array = tif.asarray()
                        if rmarray.shape[0] > 0:
                            if verbose:
                                print('concatenating remainder..')
                            array = np.concatenate((rmarray, array), axis=0)

                        dsarray, rmarray = scale_video(array,
                                                       downsample,
                                                       t_downsample,
                                                       verbose=False,
                                                       remainder=True)
                        npages = dsarray.shape[0]
                        A[i:i + npages] = dsarray

                    else:
                        array = tif.asarray()
                        if array.ndim == 3:
                            npages = array.shape[0]
                        else:  # if a movie has only one frame, shape is only 2d
                            npages = 1
                        A[i:i + npages] = array

                    i += npages

                print("\t Loading file took: {0} sec".format(timer() - t0))
        except Exception as e:
            # if something failed,  try again without size aware tiff loading
            print('Size-aware tiff-loading failed!')
            print('\tException:', e)
            tiffloading = False

    # don't use tiff size-aware loading.  load each file and append to growing matrix
    if not tiffloading:
        print('Not using size-aware tiff loading.')
        if t_downsample:
            remainder = True
        hdf5_loadkey = None

        # general function for loading a path of any type.
        # add if/elif statements for more file types
        def loadFile(path, downsample, t_downsample, remainder=None):
            t0 = timer()

            if path.endswith('.tif') | path.endswith('.tiff'):
                print("Loading tiff file at " + path)
                with tifffile.TiffFile(path) as tif:
                    A = tif.asarray()
                    if type(A) is np.memmap:
                        A = np.array(A, dtype=dtype)

            elif path.endswith('.hdf5') | path.endswith('.mat'):
                print("Loading hdf5 file at", path)
                f = h5(path)

                if 'movie' in f.keys():
                    A = f.load('movie')
                elif 'tr' in f.keys():
                    A = f.load('tr')
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

                    A = f.load(hdf5_loadkey)

            elif path.endswith('.avi') | path.endswith('.mp4'):
                cap = cv2.VideoCapture(path)

                frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                if greyscale:
                    A = np.empty((frameCount, frameHeight, frameWidth),
                                 np.dtype('uint8'))
                else:
                    A = np.empty((frameCount, frameHeight, frameWidth, 3),
                                 np.dtype('uint8'))

                fc = 0
                ret = True
                while (fc < frameCount and ret):
                    if greyscale:
                        ret, frame = cap.read()
                        A[fc] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    else:
                        ret, A[fc] = cap.read()
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
                        A = np.concatenate((remainder, A), axis=0)
                    A, remainder = scale_video(A,
                                               downsample,
                                               t_downsample,
                                               verbose=True,
                                               remainder=True)
                else:
                    A = scale_video(A, downsample, t_downsample, verbose=True)

            print("Loading file took: {0} sec".format(timer() - t0))

            if remainder is not None:
                return remainder, A
            else:
                return A

        # load either one file, or load and concatenate list of files
        if len(pathlist) == 1:
            A = loadFile(pathlist[0], downsample, t_downsample)
        else:
            remainder = None
            for i, path in enumerate(pathlist):
                if t_downsample > 1:
                    remainder, Atemp = loadFile(path,
                                                downsample,
                                                t_downsample,
                                                remainder=remainder)
                else:
                    Atemp = loadFile(path, downsample, t_downsample, remainder)

                t0 = timer()
                if i == 0:
                    A = Atemp
                else:
                    try:
                        A = np.concatenate([A, Atemp], axis=0)
                    except:
                        A = np.concatenate([A, Atemp[None, :, :]], axis=0)

                print("Concatenating arrays took: {0} sec\n".format(timer() -
                                                                    t0))

    return A


def save(array,
         path,
         resize_factor=1,
         apply_cmap=True,
         rescale_range=False,
         colormap='default',
         speed=1,
         fps=10,
         codec=None,
         mask=None,
         overlay=None,
         overlay_color=(0, 0, 0),
         save_cbar=False):
    '''
    Check what the extension of path is, and use the appropriate function
    for saving the array.  Functionality can be added for more 
    file/data types.

    For AVIs:
    Parameters (args 3+) are used for creating an avi output.  
    MJPG and XVID codecs seem to work well for linux systems, 
    so they are set as the default.
    A full list of codecs could be found at:
    http://www.fourcc.org/codecs.php.

    We may need to investigate which codec gives us the best output. 
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
            # convert RGB to BGR
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

        if codec == None:  # no codec specified
            if path.endswith('.avi'):
                codec = 'MJPG'
            elif path.endswith('.mp4'):
                if os.name == 'posix':
                    codec = 'X264'
                elif os.name == 'nt':
                    codec = 'XVID'
                else:
                    raise TypeError('Unknown os type: {0}'.format(os.name))

        # check codec and dimensions
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

        # Set up resize
        w = int(ceil(sz[1] * resize_factor))
        h = int(ceil(sz[2] * resize_factor))

        # initialize movie writer
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

    if save_cbar:
        cbarpath = os.path.splitext(path)[0] + '_colorbar.pdf'
        print('Saving Colorbar to:' + cbarpath)
        save_colorbar(scale, cbarpath, colormap=colormap)


def dfof(A, win_size=None, win_type='box'):
    '''
    Calculates the change in fluorescence over mean 
    fluorescense for a video.
    If no win_size is given, it will calculate the dfof using
    an average projection of the whole video.
    Define window size and window type from (see function from 
    time course analysis) to calculate a rolling average dfof.
    '''

    assert (type(A) == np.ndarray), 'Input was not a numpy array'
    print(A.shape)
    if A.ndim == 3:
        reshape = True
        sz = A.shape
        A = np.reshape(A, (sz[0], int(A.size / sz[0])))
    elif A.ndim == 2:
        reshape = False
    else:
        assert A.ndim == 1, ('Input was not 1-3 dimensional: {0} dim'.format(
            A.ndim))
        reshape = False
        A = A[:, None]
        print('Reshaped to two dimensional.')

    print("Array Shape (t,xy): {0}".format(A.shape))
    print('Array Type:', A.dtype)

    t0 = timer()
    if win_size == None:
        print('\nCalculating dF/F\n-----------------------')
        print('Calculating dFoF based on average projection')
        Amean = np.nanmean(A, axis=0, dtype='float32')
        print("z mean: {0} sec".format(timer() - t0))
        print("Amean shape (xy): {0}".format(Amean.shape))
        print("Amean type: {0}".format(Amean.dtype))
    else:
        print('\nCalculating rolling dF/F\n-----------------------')
        lbound = int(win_size // 2)
        ubound = int(win_size - lbound)
        window = tca.windowFunc(width=win_size, win_type=win_type)[:, None]
        window = np.repeat(window, A.shape[1], axis=1)
        print("Upper bound frame: ", ubound, "Lower bound frame: ", lbound)
        print("Window shape (t,x*y): ", window.shape)

    t0 = timer()
    A = A.astype('float32', copy=False)
    print("float32: {0} sec".format(timer() - t0))

    t0 = timer()
    t1 = timer()
    if win_size != None:
        Amoving = A.copy()
        for i in np.arange(A.shape[0]):
            if i < lbound:
                new_window = True
                window = tca.windowFunc(width=ubound + i,
                                        win_type=win_type)[:, None]
                window = np.repeat(window, A.shape[1], axis=1)
                windprod = window * A[:i + ubound, :]
            elif i > A.shape[0] - ubound:
                new_window = True
                window = tca.windowFunc(width=lbound + A.shape[0] - i,
                                        win_type=win_type)[:, None]
                window = np.repeat(window, A.shape[1], axis=1)
                windprod = window * A[i - lbound:, :]
            else:
                if new_window:
                    window = tca.windowFunc(width=win_size,
                                            win_type=win_type)[:, None]
                    window = np.repeat(window, A.shape[1], axis=1)
                    new_window = False
                windprod = np.multiply(window, A[i - lbound:i + ubound, :])

            Amean = np.nansum(windprod, axis=0, dtype='float32')

            if timer() - t1 > 300:
                t1 = timer()
                print(
                    '\tWorking on {0} frame, time passed: {1:.1f} secs'.format(
                        i,
                        timer() - t0))
            Amoving[i, :] /= Amean
            Amoving[i, :] -= 1.0
    else:
        for i in np.arange(A.shape[0]):
            A[i, :] /= Amean
            A[i, :] -= 1.0

    print("dfof normalization: {0} sec".format(timer() - t0))
    if reshape:
        if win_size != None:
            A = np.reshape(Amoving, sz)
        else:
            A = np.reshape(A, sz)
    print("A type: {0}".format(A.dtype))
    print("A shape (t,x,y): {0}\n".format(A.shape))

    return A


def rescale(array,
            low=3,
            high=7,
            cap=True,
            mean_std=None,
            mask=None,
            maskval=1,
            verbose=True,
            min_max=None,
            return_scale=False):
    '''
    determine upper and lower limits of colormap for playing movie files. 
    limits based on standard deviation from mean.  low, high are defined 
    in terms of standard deviation.  Image is updated in-place, 
    and doesn't have to be returned.
    '''

    if verbose:
        print('\nRescaling Movie\n-----------------------')

    # Mask the region if mask provided
    if mask is not None:
        copy = array.copy()
        array = get_masked_region(array, mask, maskval)

    if np.isnan(array).any():
        array[np.where(np.isnan(array))] = 0

    # if unmasked color image, add extra temporary first dimension
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

        newMin = mean - low * std
        newMax = mean + high * std
        if verbose:
            print('mean:', mean, 'low:', low, 'high:', high, 'std:', std)
    else:
        assert len(min_max) == 2
        newMin = min_max[0]
        newMax = min_max[1]

    if verbose:
        print('newMin:', newMin)
        print('newMax', newMax)

    if cap:  # don't reduce dynamic range
        if verbose:
            print('array min', np.nanmin(array))
        if newMin < array.min():
            newMin = array.min()
            if verbose:
                print('array min scaled', newMin)

        if verbose:
            print('amax', np.nanmax(array))
        if newMax > array.max():
            newMax = array.max()
            if verbose:
                print('amax scaled', newMax)

    newSlope = 255.0 / (newMax - newMin)
    if verbose:
        print('newSlope:', newSlope)
    array = array - newMin
    array = array * newSlope

    if mask is not None:
        array = insert_masked_region(copy, A, mask, maskval)

    array[np.where(array > 255)] = 255
    array[np.where(array < 0)] = 0

    if array.shape[0] == 1:  #if was converted to one higher dimension
        array = array[0, :]

    if verbose:
        print('\n')

    if not return_scale:
        return array
    else:
        return array, {'scale': newSlope, 'min': newMin, 'max': newMax}


def play(array,
         textsavepath=None,
         min_max=None,
         preprocess=True,
         overlay=None,
         toolbarsMinMax=False,
         rescale_movie=True,
         cmap='default',
         loop=True):
    '''
    play movie in opencv after normalizing display range
    array is a numpy 3-dimensional movie
    newMinMax is an optional tuple of length 2, the new display range

    Note: if preprocess is set to true, the array normalization is done 
    in place, thus the array will be rescaled outside scope of 
    this function
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
        #create negative image
        overlay = overlay == 0
        overlay = overlay.astype(np.uint8)

    print('\nPlaying Movie\n-----------------------')
    assert (type(array) == np.ndarray), 'array was not a numpy array'
    assert (array.ndim == 3) | (array.ndim == 4), ('array was not three or '
                                                   'four-dimensional array')

    windowname = "Press Esc to Close"
    cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
    print('preprocessing data...')
    #Create a resizable window

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
            toolbarsMinMax = False
        else:
            # mean, std not required if not rescaling
            mean = np.nanmean(A, dtype=np.float64)
            std = np.nanstd(A, dtype=np.float64)

        if preprocess:
            print('Pre-processing movie rescaling...')
            #Normalize movie range and change to uint8 before display
            if toolbarsMinMax:
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

        if toolbarsMinMax:

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
                im *= overlay  # black:
                # im[overlay==0]=255 # white
            cv2.putText(color, str(i), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255))  #draw frame text
            cv2.imshow(windowname, color)

        elif A.ndim == 4:
            if overlay is not None:
                im *= overlay  # black:
                # im[overlay==0]=255 # white
            cv2.putText(im, str(i), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255))  #draw frame text
            cv2.imshow(windowname, im.astype('uint8'))

        k = cv2.waitKey(10)
        if k == 27:  #if esc is pressed
            break
        elif (k == ord(' ')) and (toggleNext == True):
            tf = False
        elif (k == ord(' ')) and (toggleNext == False):
            tf = True
        toggleNext = tf  #toggle the switch

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
            # reset to 0 if looping, otherwise break the while loop
            if loop:
                i = 0
            else:
                break

    cv2.destroyAllWindows()
    for i in range(5):
        cv2.waitKey(1)

    print('\n')

    if toolbarsMinMax:
        return (lowB[0], highB[0])


def scale_video(array, s_factor=1, t_factor=1, verbose=True, remainder=False):
    if verbose:
        print('\nRescaling Video\n-----------------------')
    assert array.ndim == 3, 'Input was not a video'
    shape = array.shape

    if s_factor == None:
        s_factor = 1
    if t_factor == None:
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


def downsample(array, new_shape, keepdims=False):
    # reshape m by n matrix by factor f by reshaping matrix into
    # m f n f matricies, then applying sum across mxf, nxf matrices
    # if keepdims, video is downsampled, but number of pixels remains the same.

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

    return (array)


def rotate(array, n):

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
