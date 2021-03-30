# pySEAS

python Signal Extraction and Segmentation

---

# Installation Notes

should work with most python3s.  Currently being tested and developed on 3.8.

if some ffmpeg bindings aren't set correctly, mp4 videos may be difficult to load.  if so, use avis instead.

## Venvs

May help to create venv with system site packages for things like tk:
    
    python3.[X] -m venv <nameofyourenv> 

Manually install all dependencies with:

    pip install -r requirements.txt

Note the system site packages flag.  This allows the venv installation to access your systemwide tk installation.


## Tkinter

GUIs should work by default with base python installation, but you may need to install tk dependencies.

I've experienced issues getting tkinter to work on a python installation that isn't the most up to date version.  i.e. if you have 3.7 and 3.8 installed, tkinter on 3.7 might not work--try using 3.8.

Debian-based linux:

    sudo apt-get install python3.8 python3-tk python3.8-tk

If you experience problems with the GUI, check your tk version with the following terminal command:
    python -m tkinter
This will launch a GUI that tells you your tk version.  Make sure it is >=8.6


# TODO:

---

* fix rotate with better implementation of np.rot90, using axes key.

* fix the following warning:

    Saving Colorbar to:/home/sydney/Lab/testfiles/200220_01_ica_mosaic_movie_colorbar.pdf
    ../seas/colormaps.py:79: UserWarning: FixedFormatter should only be used together with FixedLocator
      cb.ax.set_yticklabels(ticks)

* fix rotation issue in experiment

* clean up ROIs and other things in exp

* sharing test data?

* documentation

* clean up and unify script parameters

* explicitly return ica_project output to bash.

# Developer Notes:

Using sphinx docstring documentation.

To load html docs locally:

from docs, run:

    make html

before committing, be sure to:

    make clean


To generate function help from PEP257-formatted docstrings, run from the project root:

    sphinx-apidoc -o docs seas

this only has to be done once.  


Settings are all in docs/conf.py, including modifying the system path to see the seas module. 

https://dev.to/dev0928/how-to-generate-professional-documentation-with-sphinx-4n78
https://www.sphinx-doc.org/en/master/usage/quickstart.html
