import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

    
with open("microtubule_catastrophe/__init__.py", "r") as f:
    init = f.readlines()

for line in init:
    if '__author__' in line:
        __author__ = line.split("'")[-2]
    if '__email__' in line:
        __email__ = line.split("'")[-2]
    if '__version__' in line:
        __version__ = line.split("'")[-2]
    
    
setuptools.setup(
    name='example',
    version='0.0.1',
    author='BE/Bi 103a Team 25',
    author_email='lhchen@caltech.edu',
    description='Utilities for analyzing and modeling microtubule catastrophe.',
    long_description=long_description,
    long_description_content_type='ext/markdown',
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
)

