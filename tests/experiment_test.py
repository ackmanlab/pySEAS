import pytest
from unittest.mock import patch  #, MagicMock

import tempfile
import numpy as np
import os

import seas

# create a temporary folder for test files
TEMP_FOLDER_HANDLE = tempfile.TemporaryDirectory()
TEMP_FOLDER = TEMP_FOLDER_HANDLE.name

test_video_shape = (10, 100, 100)
test_video = np.random.random((test_video_shape))
test_video = (test_video * 255).astype('uint8')

# save a test file to load as an experiment
test_video_path = os.path.join(TEMP_FOLDER, '123456_78.tiff')
seas.video.save(test_video, test_video_path)

TEST_ROI_DICT = {'roi': np.array([[0, 10], [30, 50], [20, 80]], dtype=np.int16)}


@pytest.fixture(autouse=True)
def clear_ica_files():
    # Code that runs before each test
    for file in os.listdir(TEMP_FOLDER):
        assert not file.endswith('.hdf5')
        if file.endswith('.hdf5'):
            os.remove(os.path.join(TEMP_FOLDER, file))
    yield

    # code that runs after each test
    for file in os.listdir(TEMP_FOLDER):
        if file.endswith('.hdf5'):
            os.remove(os.path.join(TEMP_FOLDER, file))


@patch('seas.experiment.roi_loader', return_value=TEST_ROI_DICT)
def test_load_experiment(mock_rois):
    exp = seas.experiment.Experiment(pathlist=test_video_path)
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
    exp = seas.experiment.Experiment(pathlist=test_video_path)
    exp.load_rois('fake_roi_path')
    exp.define_mask_boundaries()
    results = exp.ica_project(n_components=10)

    assert 'eig_mix' in results

def test_ica_decompose_without_rois_n_components():
    exp = seas.experiment.Experiment(pathlist=test_video_path)
    results = exp.ica_project(n_components=10)

    assert 'eig_mix' in results
