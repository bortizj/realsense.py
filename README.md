# Testing camera Intel Real sense

Repository to develop code to test the Intel Real sense camera (D415).

## Python environment

### Using Anaconda Prompt

Anaconda can be downloaded from the [Anaconda website](https://www.anaconda.com/products/individual).

After installing Anaconda, open a Anaconda prompt and create a new environment. Here we named it pyrs, but the name can be anything.

```shell
conda create --name pyrs python=3.10
# and to activate
conda activate pyrs
# to deactivate
conda deactivate
```

To remove the environment, if you want/need to start fresh.

```shell
conda env remove --name pyrs
```

## Installation

Once you have your environment setup you can install the package and its requirements.

- [RealSense](https://dev.intelrealsense.com/docs/python2), [Numpy](https://numpy.org/), [opencv](https://opencv.org/), ...
```shell
pip install pyrealsense2 numpy scipy opencv-python open3d tqdm packaging psutil pandas matplotlib
```
and
- the package itself
```shell
cd path/to/repo/realsense.py
pip install -e .
```
