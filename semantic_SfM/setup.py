from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='semantic_sfm',
    version='0.0.1',
    description='x segmentation on photogrammetry point cloud',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/ZhiangChen/semantic_SfM',
    author='Zhiang Chen',
    author_email='zhiang.chen@caltech.edu',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'laspy>=2.1.2',
        'numpy>=1.19.5',
        'matplotlib>=3.0.3',
        'opencv-python>=4.8.1'
        'scipy>=1.5.4',
        'numba>=0.53.1',
    ],
)