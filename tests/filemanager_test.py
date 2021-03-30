import pytest
from unittest.mock import patch

import tempfile
import numpy as np
import os

import seas.filemanager

# create a temporary folder for test files
TEMP_FOLDER_HANDLE = tempfile.TemporaryDirectory()
TEMP_FOLDER = TEMP_FOLDER_HANDLE.name

MOCK_FOLDER_CONTENTS = ['170721_02_RoiSet.zip', '170721_02.tif']
EXPNAME = '170721_02'


def test_find_files():
    files = seas.filemanager.find_files(TEMP_FOLDER, '*')

    assert len(files) == 0

    newfile = os.path.join(TEMP_FOLDER, 'test.extension')
    with open(newfile, 'w') as f:
        f.write(' ')

    files = seas.filemanager.find_files(TEMP_FOLDER, 'extension', suffix=True)
    print(files)
    assert len(files) == 1

    os.remove(newfile)


@patch('seas.filemanager.os.listdir', return_value=MOCK_FOLDER_CONTENTS)
def test_experiment_sorter(mock_listdir):

    files = seas.filemanager.experiment_sorter(TEMP_FOLDER, EXPNAME)

    assert len(files['movies']) == 1
    assert len(files['roi']) == 1
