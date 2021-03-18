import pytest

import tempfile
import numpy as np
import os

import seas.video
import seas.colormaps

def test_tiff_loading_and_saving():
    test_video = np.random.random((3,10,10))
    test_video = (test_video * 255).astype('uint8') 

    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = tempdir

        save_path = os.path.join(tempdir, 'test.tiff')

        print(os.path.exists(tempdir))
        seas.video.save(test_video, save_path)

        loaded_video = seas.video.load(save_path)

    assert np.allclose(test_video, loaded_video)


# need to test other formats.  test loading and saving colorbars, saving 3 color vs greyscale, etc. 

# def test_avi_loading_and_saving():

#     test_video = np.random.random((3,10,10))
#     test_video = (test_video * 255).astype('uint8') 

#     with tempfile.TemporaryDirectory() as tempdir:
#         tempdir = tempdir

#         save_path = os.path.join(tempdir, 'test.avi')

#         print(os.path.exists(tempdir))
#         pyseas.video.save(test_video, save_path, rescale=False, apply_cmap=False)
            ### for some reason, this isn't scaling properly.  maybe cv2 function in saving.

#         loaded_video = pyseas.video.load(save_path)

#     print(test_video.shape)
#     print(loaded_video.shape)

#     print(test_video[0])
#     print(loaded_video[0])

#     assert np.allclose(test_video, loaded_video)


## needs tests:
## dfof, rescale, play(?), scale_video, downsample