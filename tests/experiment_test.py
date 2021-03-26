import pytest
from unittest.mock import patch  #, MagicMock

import tempfile
import numpy as np
import os

import seas
from seas.experiment import Experiment
from seas.video import save

# create a temporary folder for test files
TEMP_FOLDER_HANDLE = tempfile.TemporaryDirectory()
TEMP_FOLDER = TEMP_FOLDER_HANDLE.name

test_video_shape = (10, 100, 100)
test_video = np.random.random((test_video_shape))
test_video = (test_video * 255).astype('uint8')

# save a test file to load as an experiment
test_video_path = os.path.join(TEMP_FOLDER, '123456_78.tiff')
save(test_video, test_video_path)

TEST_ROI_DICT = {'roi': np.array([[0, 10], [30, 50], [20, 80]], dtype=np.int16)}


@patch('seas.experiment.roi_loader', return_value=TEST_ROI_DICT)
def test_load_experiment(mock_rois):
    exp = Experiment(pathlist=test_video_path)
    assert not exp.downsample
    assert not exp.downsample_t

    assert exp.n_rotations == 0

    exp.load_rois('fake_roi_path')

    # bounding box is originally the whole video
    assert np.allclose(
        exp.bounding_box,
        np.array([[0, test_video_shape[1]], [0, test_video_shape[2]]]))

    # after defining mask boundaries, it is now smaller
    exp.define_mask_boundaries()

    assert not np.allclose(
        exp.bounding_box,
        np.array([[0, test_video_shape[0]], [0, test_video_shape[1]]]))


@patch('seas.experiment.roi_loader', return_value=TEST_ROI_DICT)
def test_ica_decompose_with_rois_n_components(mock_rois):
    exp = Experiment(pathlist=test_video_path)
    exp.load_rois('fake_roi_path')
    exp.define_mask_boundaries()
    exp.ica_filter(n_components=10)

    print(os.listdir(TEMP_FOLDER))
    raise SystemError


def test_ica_decompose_without_rois_n_components():
    exp = Experiment(pathlist=test_video_path)
    exp.ica_filter(n_components=10)


@patch('seas.experiment.roi_loader', return_value=TEST_ROI_DICT)
def test_ica_decompose_wit_rois_auto_components(mock_rois):
    exp = Experiment(pathlist=test_video_path)
    exp.load_rois('fake_roi_path')
    exp.define_mask_boundaries()
    exp.ica_filter()


def test_ica_decompose_without_rois_auto_components():
    exp = Experiment(pathlist=test_video_path)
    exp.ica_filter()

# def test_ica_naming