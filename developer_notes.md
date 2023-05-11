# Developer Notes:

## Documentation

Using sphinx docstring documentation.

To load html docs locally:

from ./docs, run:

    make html

this will generate a bunch of html files in ./docs/\_build/html.  Open up index.html in your favorite browser to view.

before committing, be sure to:

    make clean


To gather all modules and functions from the package, run from the project root:

    sphinx-apidoc -o docs seas

this only has to be done once (and has already been done).  If modules were changed, delete the old ./docs/seas.rst and ./docs/modules.rst, and rerun sphinx-apidoc.

Default configurations have been changed for modules, so take note of the module formatting before rerunning.


Settings are all in docs/conf.py, including modifying the system path to see the seas module. 

https://dev.to/dev0928/how-to-generate-professional-documentation-with-sphinx-4n78
https://www.sphinx-doc.org/en/master/usage/quickstart.html
https://stackoverflow.com/questions/2701998/sphinx-autodoc-is-not-automatic-enough
https://sphinx-automodapi.readthedocs.io/en/latest/


## Making and locally installing the python package

Test and install the python package locally:

from the project root dir, generate package formatting:

    hatch build

and install it to environment:

    pip install -e .

To check if the built wheel passes all checks: 

    twine check dist/*

## Deploying the python package