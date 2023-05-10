.. install-notes:

Installation Notes
============

This package should work with most python3 versions.  It is currently being tested and developed on python 3.8.

AV Libraries
^^^^^^^^^^^^

If some ffmpeg bindings aren't set correctly, mp4 videos may be difficult to load.  if so, use the --avi flag to save as avis instead.

Virtual Environments
^^^^^^^^^^^^^^^^^^^^

May help to create venv with system site packages for things like tk:
    
.. code-block:: bash

    python3.[X] -m venv <nameofyourenv> 

Manually install all dependencies with:

.. code-block:: bash

    pip install -r seas

Right now the dependencies are flexible on versioning of the required packages.  Please report if you run into any issues and we will make them more stringent.

Tkinter
^^^^^^^

GUIs should work by default with base python installation, but you may need to install tk dependencies.

I've experienced issues getting tkinter to work on a python installation that isn't the most up to date version.  i.e. if you have 3.7 and 3.8 installed, tkinter on 3.7 might not work--try using 3.8.

Debian-based linux:

.. code-block:: bash

    sudo apt-get install python3.8 python3-tk python3.8-tk

If you experience problems with the GUI, check your tk version with the following terminal command:

.. code-block:: bash

    python -m tkinter

This will launch a GUI that tells you your tk version.  Make sure it is >=8.6
