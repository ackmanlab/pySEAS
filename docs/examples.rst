Example Usage
============


Example scripts are available for download from the project `Github Page <https://github.com/ackmanlab/pySEAS/tree/master/examples>`_.


Play dFoF Movie
^^^^^^^^^^^^^

.. code-block:: bash

    python play_dfof_movie.py -m [DIRECTORY]/testmovie.tiff -d 10


for our movies, use the following rotations:

older:

.. code-block:: bash

    python ica_project.py -f /home/sydney/Lab/testfiles/ -e 170721_02 -d 10 -r 3

newer:

.. code-block:: bash

    python ica_project.py -f /home/sydney/Lab/testfiles/ -e 200220_01 -d 10 -r 1 -rr 2



ICA Project
^^^^^^^^^^^^^

The following function ICA projects an experiment listed with -e at directory -f. -d sets an integer downsample value.  Rois can be included with argument -r.  Rois are created and saved from FIJI/ImageJ.

The folder needs to have videos in format DATE_EXPNO.tif and rois in format DATE_EXPNO_RoiSet.zip.  If there are metadata files in format DATE_EXPNO_meta.yaml, they will be loaded and stored in ica.hdf5 file.

.. code-block:: bash

    python ica_project.py -f [DIRECTORY] -e TEST_VID -d 10

The output will be in format: DATE_EXPNO\_[downsample factor]\_ica.hdf5.  This will be used as the output for future scripts.

View Components
^^^^^^^^^^^^^

Launches a GUI to view ica components, and domain maps and timecourses if they exist.  This is where ica components can be manually sorted into signal or artifacts.  

Launch the gui with:

.. code-block:: bash

    python view_components.py -i INPUT_ica.hdf5 [-r optional rotation]

Once it's open, youll land on main page.  Change pages through the drop down menu 'view' tab, or with the F-keys (f1, f2, etc).  Click components to assign as signal/artifact on main page.

To change pages of components, or component focus on later pages, use arrow keys.  For domain correlation pages (only available after domain ROI loading), click on map to change focus.

To exit the GUI, DO NOT USE THE X BUTTON.   Your terminal will become unresponsive.  Exit using file>exit or file>save.  You can also use 'Esc' or 's' keys.  

Saving will save 'artifact_components' to hdf5 file, and allow domain_map creation.

Domain Map
^^^^^^^^^^^^^
Create the domain map after assigning components as artifacts, or using force flag if artifacts should be included in map:

.. code-block:: bash

    python domain_map.py -i INPUT_ica.hdf5

If you want to make domain map figures, also include the --figures flag.

If you want to save mosaic movies, use --mosaic_movie flag.  This defaults to creating an mp4 file, which may not work based on your set video codecs.  If so, try again with the addition of the -avi flag.

HDF5 Manager
^^^^^^^^^^^^^

For use manually inspecting or modifying hdf5 files.  Try viewing them with 

.. code-block:: bash

    python hdf5manager.py -i INPUT_ica.hdf5


