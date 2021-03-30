import os
import tempfile
import pytest
import numpy as np

import seas.colormaps

# create a temporary folder for saving files
TEMP_FOLDER_HANDLE = tempfile.TemporaryDirectory()
TEMP_FOLDER = TEMP_FOLDER_HANDLE.name

TEST_SCALE = {'min': 0, 'max': 1}


def test_colormap_conversion():
    cv2_cmap = seas.colormaps.get_mpl_colormap('rainbow')
    assert cv2_cmap.shape == (256, 1, 3)

    cv2_cmap = seas.colormaps.get_mpl_colormap('rainbow_r')
    assert cv2_cmap.shape == (256, 1, 3)


def test_invalid_cmap_name_conversion():
    with pytest.raises(ValueError):
        cv2_cmap = seas.colormaps.get_mpl_colormap('invalid_colormap_name')


def test_saving_colorbar_pdf_file():
    cb_location = os.path.join(TEMP_FOLDER, 'test.pdf')
    seas.colormaps.save_colorbar(TEST_SCALE, cb_location)

    assert os.path.exists(cb_location)
    os.remove(cb_location)


def test_saving_specific_colormap():
    cb_location = os.path.join(TEMP_FOLDER, 'test.pdf')
    seas.colormaps.save_colorbar(TEST_SCALE, cb_location,
                                 seas.colormaps.COMPONENT_COLORMAP)

    assert os.path.exists(cb_location)
    os.remove(cb_location)


def test_saving_plt_named_colormap():
    cb_location = os.path.join(TEMP_FOLDER, 'test.pdf')
    seas.colormaps.save_colorbar(TEST_SCALE, cb_location, 'rainbow_r')

    assert os.path.exists(cb_location)
    os.remove(cb_location)


def test_saving_plt_invalid_named_colormap():
    cb_location = os.path.join(TEMP_FOLDER, 'test.pdf')
    with pytest.raises(ValueError):
        seas.colormaps.save_colorbar(TEST_SCALE, cb_location,
                                     'invalid_colormap_name')

    assert not os.path.exists(cb_location)


def test_saving_colorbar_fails_txt_file():
    cb_location = os.path.join(TEMP_FOLDER, 'test.txt')
    with pytest.raises(ValueError):
        seas.colormaps.save_colorbar(TEST_SCALE, cb_location)

    assert not os.path.exists(cb_location)


def test_apply_colormap():

    test_shape = (3, 10, 10)
    test_video = np.random.random(test_shape)
    colored_video = seas.colormaps.apply_colormap(test_video)

    assert colored_video.shape == (test_shape[0], test_shape[1], test_shape[2],
                                   3)
