from timeit import default_timer as timer
import tifffile
import warnings
import sys
import numpy as np
from math import ceil
import cv2

try:
    from convertcmap import get_mpl_colormap
    default_colormap = get_mpl_colormap('rainbow')
    component_colormap = get_mpl_colormap('coolwarm')

except Exception as e:
    print('reverting to similar cv2 colormaps, instead of mpl equivalents.')

    if 'cv2' in sys.modules:
        default_colormap = cv2.COLORMAP_JET
        component_colormap = cv2.COLORMAP_JET
    else:
        print('OpenCV import failed!  No default colormap available.')
        default_colormap = None

    print('\t ERROR : ', e)


def load(pathlist, downsample=False, t_downsample=False, dtype=None, 
        verbose=True, tiffloading=True, greyscale=True):
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
    
    if type(pathlist) is str: # if just one path got in
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

        if not t_downsample: t_downsample = 1
        if not downsample: downsample = 1

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
                shape = (shape[0]//t_downsample, 
                    shape[1]//downsample, shape[2]//downsample)
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
                        if downsample > 1: print(
                            '\t spatially downsampling by {0}..'.format(downsample))
                        if t_downsample > 1: print(
                            '\t temporally downsampling by {0}..'.format(t_downsample))

                        array = tif.asarray()
                        if rmarray.shape[0] > 0:
                            if verbose: print('concatenating remainder..')
                            array = np.concatenate((rmarray, array), axis=0)

                        dsarray, rmarray = scaleVideo(array, 
                            downsample, t_downsample, verbose=False, remainder=True)
                        npages = dsarray.shape[0]
                        A[i:i+npages] = dsarray

                    else:
                        array = tif.asarray()
                        if array.ndim == 3:
                            npages = array.shape[0]
                        else: # if a movie has only one frame, shape is only 2d
                            npages = 1
                        A[i:i+npages] = array

                    i += npages

                print("\t Loading file took: {0} sec".format(timer() - t0))
        except Exception as e:
            # if something failed,  try again without size aware tiff loading
            print('Size-aware tiff-loading failed!')
            print('\tException:', e)
            tiffloading=False
                
    # don't use tiff size-aware loading.  load each file and append to growing matrix
    if not tiffloading:
        print('Not using size-aware tiff loading.')
        if t_downsample: remainder=True
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
                while (fc < frameCount  and ret):
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
                print('\t temporally downsampling by {0}..'.format(t_downsample))
                if remainder is not None:
                    if (type(remainder) == np.ndarray) and \
                            (remainder.shape[0] != 0): 
                        A = np.concatenate((remainder, A), axis=0)
                    A, remainder = scaleVideo(A, downsample, t_downsample, 
                        verbose=True, remainder=True)
                else:
                    A = scaleVideo(A, downsample, t_downsample, verbose=True)

            print("Loading file took: {0} sec".format(timer() - t0))
                
            if remainder is not None:
                return remainder, A
            else:
                return A

        # load either one file, or load and concatenate list of files 
        if len(pathlist) == 1:
            A = loadFile(pathlist[0], downsample, t_downsample)
        else:
            remainder=None
            for i, path in enumerate(pathlist):
                if t_downsample > 1:
                    remainder, Atemp = loadFile(path, 
                        downsample, t_downsample, remainder=remainder)
                else:
                    Atemp = loadFile(path, downsample, 
                        t_downsample, remainder)

                t0 = timer()
                if i == 0:
                    A = Atemp
                else:
                    try:
                        A = np.concatenate([A, Atemp], axis=0)
                    except:
                        A = np.concatenate([A, Atemp[None,:,:]], axis=0)

                print("Concatenating arrays took: {0} sec\n".format(
                    timer() - t0))
    
    return A


def save(array, path, resize_factor = 1, apply_cmap = True, rescale=False,
    colormap = default_colormap, speed = 1, fps = 10, codec = None, mask=None,
    overlay=None, overlay_color=(0,0,0), save_cbar=False):
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
    assert(type(array) == np.ndarray), ('Movie to save was not a '
        'numpy array')

    if path.endswith('.tif') | path.endswith('.tiff'):
        print('Saving to: ' + path)
        t0 = timer()
        if array.shape[2] == 3:
            # convert RGB to BGR
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
            print('Converted RGB image to BGR.')
        with tifffile.TiffWriter(path) as tif:
            tif.save(array)
        print("Save file: {0} sec\n".format(timer()-t0))
        save_cbar = False

    elif path.endswith('.png'):
        assert array.ndim <= 3, 'File was not an image'

        if rescale:
            array, scale = rescaleMovie(array, return_scale=True)

        array = array.astype('uint8')

        if apply_cmap and (array.ndim == 2):
            array = cv2.applyColorMap(array.copy(), colormap)

        if mask is not None:
            ind = np.where(mask == 0)

            if ind[0].size > 0:
                alpha = np.ones((array.shape[0], array.shape[1]), dtype='uint8')*255
                
                alpha[ind] = 0
                array = np.concatenate((array, alpha[:,:,None]), axis=2)

        cv2.imwrite(path, array)

    elif path.endswith('.avi') | path.endswith('.mp4'):
        sz = array.shape

        if rescale:
            array, scale = rescaleMovie(array, return_scale=True)
        
        array = array.astype('uint8')

        if codec == None: # no codec specified
            if path.endswith('.avi'):
                    codec = 'MJPG'
            elif path.endswith('.mp4'):
                if os.name is 'posix':
                    codec = 'X264'
                elif os.name is 'nt':
                    # codec = 'H264'
                    codec = 'XVID'

        # check codec and dimensions
        if array.ndim == 3:
            if apply_cmap == False:
                sz = array.shape
                array = array.reshape(sz[0], sz[1]*sz[2])
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
                'cannot be written in this format.'.format(array.ndim))

        print('Movie will be written in {0} using the {1} codec'.format(
            movietype, codec))
        print('Saving to: ' + path)
    
        # Set up resize
        w = int(ceil(sz[1] * resize_factor))
        h = int(ceil(sz[2] * resize_factor))

        # initialize movie writer
        display_speed = fps * speed
        fourcc = cv2.VideoWriter_fourcc(*codec) 
        out = cv2.VideoWriter(path, fourcc, display_speed, (h,w), True)

        if overlay is not None:
            if resize_factor != 1:
                overlay = cv2.resize(overlay, (h, w), interpolation = cv2.INTER_AREA)
            overlayindices = np.where(overlay > 0)

        for i in range(sz[0]):
            frame = array[i].copy()

            if (frame.ndim == 2) & (resize_factor != 1):
                frame = cv2.resize(frame, (h, w), interpolation = cv2.INTER_AREA)
            elif (frame.ndim == 3) & (resize_factor != 1):
                frame = cv2.resize(frame, (h, w, 3), interpolation = cv2.INTER_AREA)

            frame = frame.astype('uint8')

            if apply_cmap:
                frame = cv2.applyColorMap(frame, colormap)

            if overlay is not None:
                frame[overlayindices[0], overlayindices[1],:] = overlay_color 

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
        saveColorbar(scale, cbarpath, colormap=colormap)

    return