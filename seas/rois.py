import zipfile
import re
import numpy as np
from io import BytesIO
import cv2
from typing import Tuple, List


def get_masked_region(array: np.ndarray,
                      mask: np.ndarray,
                      maskval: float = None):
    '''
    Extract a spatially masked array where the mask == 1 or 
    mask == maskval. Reinsert masked region using insert_masked_region function.
    
    Arguments:
        array: a (t,x,y) numpy array or an (x,y,c) numpy array

    Returns:
        masked_array: the masked array in (t,xy) or (xy,c) format.

    Raises:
        Exception: if (x,y) mask indices did not match the shape of the input array.
    '''
    if maskval == None:
        maskind = np.where(mask == 1)
    else:
        maskind = np.where(mask == maskval)

    if array.shape[0:2] == mask.shape:  # Check if dimensions align for masking.
        masked_array = array[maskind]
    elif (array.shape[1], array.shape[2]) == mask.shape:
        masked_array = array.swapaxes(0, 1).swapaxes(1, 2)[maskind]
        masked_array = masked_array.swapaxes(0, 1)
    else:
        raise Exception(
            'Unknown mask indices with the following '
            'dimensions:\n',
            'Array: {0} Mask: {1}'.format(array.shape, mask.shape))

    return masked_array


def insert_masked_region(array: np.ndarray,
                         masked_array: np.ndarray,
                         mask: np.ndarray,
                         maskval: float = 1):
    '''
    Insert a spatially masked array from get_masked_region.  
    Masked array is inserted where the mask == 1 or mask == maskval. 
    Accepts masked array in (t,xy) or (xy,c) format.
    Accepts (t,x,y) arrays or (x,y,c) arrays, returns them in the 
    same format.

    Arguments:
        array: A (t,x,y) numpy array or an (x,y,c) numpy array
        masked_array: The masked array in (t,xy) or (xy,c) format.
            Usually extracted using get_masked_region.
        mask: The mask used to extract the masked array originally, 
            will be used for reinsertion.
        maskval: The value used to extracted the region, if not the default of 1.

    Returns:
        array: The array with the masked values reinserted.

    '''
    maskind = np.where(mask == maskval)

    if array.shape[0:2] == mask.shape:  #check if dimensions align for masking
        array[maskind] = masked_array
    elif (array.shape[1], array.shape[2]) == mask.shape:
        masked_array = masked_array.swapaxes(0, 1)
        array = array.swapaxes(0, 1).swapaxes(1, 2)
        array[maskind] = masked_array
        array = array.swapaxes(1, 2).swapaxes(0, 1)
    else:
        raise Exception('Unknown mask indices with the following '
                        'dimensions:\n'
                        'Array: {0}, Mask: {1}'.format(A.shape, mask.shape))

    return array


def roi_loader(path: str, verbose: bool = True) -> dict:
    print('\nLoading Rois\n-----------------------')
    '''
    Load ROIs from a path.  
    Rois must be in a zip file containing multiple polygon roi files, 
    created by FIJI/ImageJ.

    Arguments:
        path: The path to the zip file of FIJI ROIs.
        verbose: Whether to produce verbose output.

    Returns:
        rois: A dictionary containing lists of coordinates, 
            which should be bounded polygons.
    '''

    def load_roi_file(fileobj):
        '''
        This is based on a gist from Luis Pedro:
        https://gist.github.com/luispedro/3437255

        ROIfile is a .roi file view.  
        It must first be opened as a bitstring through BytesIO to allow 
        for seeking through the bitstring.
        Read ImageJ's ROI format. Points are returned in a nx2 array. Each row
        is in (x,y) order.
        This function may not work for float32 formats, or with images that 
        have subpixel resolution.
        '''

        def get8():
            s = fileobj.read(1)
            if not s:
                raise IOError('readroi: Unexpected EOF')
            return ord(s)

        def get16():
            b0 = get8()
            b1 = get8()
            return (b0 << 8) | b1

        assert fileobj.read(4) == b'Iout'
        version = get16()
        roi_type = get8()
        get8()

        top = get16()
        left = get16()
        bottom = get16()
        right = get16()

        n_coordinates = get16()
        fileobj.seek(64)  # seek to after header, where coordinates start

        points = np.empty((n_coordinates, 2), dtype=np.int16)
        points[:, 0] = [get16() for i in range(n_coordinates)]  # X coordinate
        points[:, 1] = [get16() for i in range(n_coordinates)]  # Y coordinate
        points[:, 1] += top
        points[:, 0] += left

        return points

    rois = dict()

    # Load a .zip file of .roi files.
    if path.endswith('.zip'):
        f = zipfile.ZipFile(path)
        roifilelist = f.namelist()

        for roifile in roifilelist:
            if verbose:
                print('Loading ROIs at: ', roifile)
            roiname = re.sub('.roi', '', roifile)

            roidata = BytesIO(f.read(roifile))
            points = load_roi_file(roidata)
            rois[roiname] = points
            roidata.close()

    # Not a valid file extension.
    else:
        raise Exception('{0} does not have a valid roi '
                        'file extension.  Accepted file format is '
                        '.zip.'.format(path))

    return rois


def make_mask(polylist: np.ndarray, shape: Tuple[int, int], bounding_box=None):
    '''
    Makes mask from coordinates of polygon(s).  
    
    Polylist is a list of (x,y) numpy arrays, 

    Arguments:
        polylist: A list of (x,y) numpy arrays returned under a roi key by roi_loader, 
            each representing a closed polygon to draw..
        shape: (x,y) coordinate of shape to draw the polygons onto.
        bounding_box: If provided, offset the polygons by this amount to take into account 
            a previous cropping of the input image/video.

    Returns:

    '''
    assert (len(shape) == 2), 'Shape was not 2D'

    if bounding_box is not None:
        polylist = polylist - bounding_box[:, 0][::-1]

    roimask = np.zeros(shape, dtype='uint8')
    cv2.fillPoly(roimask, [polylist.astype(np.int32).reshape((-1, 1, 2))], 1)

    return roimask


def draw_bounding_box(image: np.ndarray,
                      required: bool = True) -> List[List[int]]:
    '''
    Draw a bounding box on a two dimensional image.  
    Image should already be scaled to 0-255 in uint8.

    Arguments:
        image: A numpy array to draw the bounding box on.
        required: Whether to raise an error if the process is 
            exited without a valid bounding box.

    Returns:
        ref_coord: An ROI bounding box with shape [[x0,x1],[y0,y1]].
    '''
    print('\nDrawing Bounding Box\n-----------------------')

    assert (len(image.shape) == 2), '''The image is not two dimensional.  
        Shape: '''.format(image.shape)

    global refPt, cropping
    # Initialize the list of reference points
    # and boolean indicating whether cropping is being performed.
    window_name = "Draw bounding box on image.  \
    'r' = reset, 'a' = all, s' = save, +/- = zoom"

    refPt = []
    cropping = False
    zoom = 1
    zfactor = 5 / 4

    def click_and_crop(event, x, y, flags, param):
        global refPt, cropping
        # If the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed.
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt = [(x, y)]
            cropping = True

        # Check to see if left mouse button was released.
        elif event == cv2.EVENT_LBUTTONUP:
            # Record the ending (x,y) coordinates and indicate that
            # the cropping operation is finished.
            refPt.append((x, y))
            cropping = False

            # draw a rectangle around the point of interest
            cv2.rectangle(draw, refPt[0], refPt[1], 255, 2)
            cv2.imshow(window_name, draw)

    cv2.destroyAllWindows()
    for i in range(1, 5):
        cv2.waitKey(1)

    # Load the image, clone it to draw, and setup the mouse.
    image = image.astype('uint8')
    draw = image.copy()
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_and_crop)

    # Keep looping until the 'q' key is pressed.
    while True:
        # Display the image and wait for a keypress.
        cv2.imshow(window_name, draw)
        key = cv2.waitKey(1) & 0xFF

        # If the '=' key is pressed, zoom in.
        if key == ord("="):
            draw = cv2.resize(draw,
                              None,
                              fx=zfactor,
                              fy=zfactor,
                              interpolation=cv2.INTER_CUBIC)
            zoom = zoom * zfactor

        # If the '-' key is pressed, zoom out.
        if key == ord("-"):
            draw = cv2.resize(draw,
                              None,
                              fx=1 / zfactor,
                              fy=1 / zfactor,
                              interpolation=cv2.INTER_CUBIC)
            zoom = zoom * 1 / zfactor

        # If the 'r' key is pressed, reset the cropping region.
        if key == ord("r"):
            draw = image.copy()
            zoom = 1

        if key == ord("a"):
            print('Taking entire image')
            refPt = [(0, 0), (image.shape[0], image.shape[1])]
            break

        # If the 's' key is pressed, break from the loop and save ROI.
        elif key == ord("s"):
            break

    # If there are two reference points, then crop the region of interestd.

    if len(refPt) == 2:
        ref_coord = np.array([
            sorted([refPt[0][1], refPt[1][1]]),
            sorted([refPt[0][0], refPt[1][0]])
        ])

        # Unzoom reference coordinates.
        for i in range(2):
            ref_coord[i] = [round(y / zoom) for y in ref_coord[i]]

    else:
        print('Exiting without bounding box')
        ref_coord = None

    cv2.destroyAllWindows()
    for i in range(5):
        cv2.waitKey(1)

    if (ref_coord is None) and (required):
        assert (ref_coord is not None), 'Exited with no bounding box'
        print('Bounding box: x:{0}, y:{1}\n'.format(ref_coord[0], ref_coord[1]))
        print('\n')

    return ref_coord
