import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="seas",
    version="0.0.1",
    author="Sydney Weiser",
    author_email="scweiser@ucsc.edu",
    description="python Signal Extraction and Segmentation",
    long_description="Data-driven filtration and segmentation of mesoscale neural dynamics",
    long_description_content_type="text/markdown",
    url="https://github.com/ackmanlab/pySEAS",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
