import pytest

import tempfile
import numpy as np
import os

from seas.experiment import Experiment


# def test_tiff_loading_and_saving():
#     test_video = np.random.random((3,10,10))
#     test_video = (test_video * 255).astype('uint8') 

#     with tempfile.TemporaryDirectory() as tempdir:
#         tempdir = tempdir

#         save_path = os.path.join(tempdir, 'test.tiff')
#         seas.video.save(test_video, save_path)

#         loaded_video = seas.video.load(save_path)

#     assert np.allclose(test_video, loaded_video)
