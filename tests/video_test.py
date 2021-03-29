import pytest

import tempfile
import numpy as np
import os

import seas.video
import seas.colormaps


def test_tiff_loading_and_saving():
    test_video = np.random.random((3, 10, 10))
    test_video = (test_video * 255).astype('uint8')

    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = tempdir

        save_path = os.path.join(tempdir, 'test.tiff')
        seas.video.save(test_video, save_path)

        loaded_video = seas.video.load(save_path)

    assert np.allclose(test_video, loaded_video)


def test_rotate_image():
    test_image = np.arange(5 * 5).reshape((5, 5))

    # image rotated by 2 is the same as image flipped vertically and horizontally
    assert np.allclose(np.fliplr(np.flipud(test_image)),
                       seas.video.rotate(test_image, 2))

    assert np.allclose(test_image, seas.video.rotate(test_image, 4))


def test_rotate_video():
    test_image = np.arange(3 * 5 * 5).reshape((3, 5, 5))
    assert np.allclose(test_image, seas.video.rotate(test_image, 4))
