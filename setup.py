import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="seas",
    version="0.0.2",
    author="Sydney Weiser",
    author_email="scweiser@ucsc.edu",
    description="python Signal Extraction and Segmentation",
    long_description="Data-driven filtration and segmentation of mesoscale neural dynamics",
    long_description_content_type="text/markdown",
    url="https://github.com/ackmanlab/pySEAS",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'h5py',
        'matplotlib',
        'numpy',
        'opencv-python',
        'PyYAML',
        'scikit-learn',
        'scipy',
        'sklearn',
        'tifffile',
        'tk'
    ],
    python_requires='>=3.5',
)
