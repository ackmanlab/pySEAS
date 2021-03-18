import zipfile
import re
import numpy as np
from io import BytesIO
import cv2

def get_masked_region(A, mask, maskval=None):
    '''
    Extract a spatially masked array where the mask == 1 or 
    mask == maskval. 
    
    Accepts (t,x,y) arrays or (x,y,c) arrays.

    Returns the masked array in (t,xy) or (xy,c) format.

    Reinsert masked region using insert_masked_region function.
    '''

    if maskval == None:
        maskind = np.where(mask == 1)
    else:
        maskind = np.where(mask == maskval)

    if A.shape[0:2] == mask.shape:  #check if dimensions align for masking
        M = A[maskind]
    elif (A.shape[1], A.shape[2]) == mask.shape:
        M = A.swapaxes(0, 1).swapaxes(1, 2)[maskind]
        M = M.swapaxes(0, 1)
    else:
        raise Exception(
            'Unknown mask indices with the following '
            'dimensions:\n', 'Array: {0} Mask: {1}'.format(A.shape, mask.shape))

    return M


def insert_masked_region(A, M, mask, maskval=1):
    '''
    Insert a spatially masked array from get_masked_region.  
    Masked array is inserted where the mask == 1 or mask == maskval. 
    Accepts masked array in (t,xy) or (xy,c) format.
    Accepts (t,x,y) arrays or (x,y,c) arrays, returns them in the 
    same format.
    '''
    maskind = np.where(mask == maskval)

    if A.shape[0:2] == mask.shape:  #check if dimensions align for masking
        A[maskind] = M
    elif (A.shape[1], A.shape[2]) == mask.shape:
        M = M.swapaxes(0, 1)
        A = A.swapaxes(0, 1).swapaxes(1, 2)
        A[maskind] = M
        A = A.swapaxes(1, 2).swapaxes(0, 1)
        # A.swapaxes(0,1).swapaxes(1,2)[maskind] = M
    else:
        raise Exception('Unknown mask indices with the following '
                        'dimensions:\n'
                        'Array: {0}, Mask: {1}'.format(A.shape, mask.shape))

    return A


def roi_loader(path, verbose=True):
    print('\nLoading Rois\n-----------------------')

    def load_roi_file(fileobj):
        '''
        points = roiLoader(ROIfile)
        ROIfile is a .roi file view.  
        It must first be opened as a bitstring through BytesIO to allow 
        for seeking through the bitstring.
        Read ImageJ's ROI format. Points are returned in a nx2 array. Each row
        is in (x,y) order.
        This function may not work for float32 formats, or with images that 
        have subpixel resolution.

        This is based on a gist from luis pedro:
        https://gist.github.com/luispedro/3437255
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

    # Load a .zip file of .roi files
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

    # Not a valid file extension
    else:
        raise Exception('{0} does not have a valid roi '
                        'file extension.  Accepted file format is '
                        '.zip.'.format(path))

    return rois


def make_mask(polylist, shape, bounding_box=None):
    '''
    Makes mask from coordinates of polygon(s).  
    
    Polylist is a list of numpy arrays, each representing a 
    closed polygon to draw.
    '''
    assert (len(shape) == 2), 'Shape was not 2D'

    if bounding_box is not None:
        polylist = polylist - bounding_box[:, 0][::-1]

    roimask = np.zeros(shape, dtype='uint8')
    cv2.fillPoly(roimask, [polylist.astype(np.int32).reshape((-1, 1, 2))], 1)

    return roimask
