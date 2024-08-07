# semantic_SfM
Instance segmentation for SfM point cloud. 

## Installation
Option 1 (recommended): 
```
git clone https://github.com/ZhiangChen/semantic_SfM.git
cd semantic_SfM/semantic_SfM/ssfm/
python3 -m pip install -e .
```

Option 2:
```
python3 -m pip install semantic-SfM
```

**Hardware requirements**: This project utilizes the Segment Anything Model (SAM) for panoptic segmentation. GPUs are not required for SAM but are recommended to expedite inference speed. The other parts of this project use only CPUs. As a point cloud will be stored in memory, memory usage depends on the point cloud size. 

## SSfM Architecture
[files.py](./semantic_SfM/ssfm/files.py) provides utility functions for file operation, such as point cloud and image files.

[image_segmentation.py](./semantic_SfM/ssfm/image_segmentation.py) allows to select deep learning models for instance segmentation and panoptic segmentation. 

[probabilistic_projection.py](./semantic_SfM/ssfm/probabilistic_projection.py) projects point clouds to images and creates a probablistic semantics.

[object_registration.py](./semantic_SfM/ssfm/object_registration.py) registers objects (instances and stuffs) in point clouds.



## Support Platforms and Models
This repository supports SfM products from the following platforms: 
- [WebODM](https://opendronemap.org/webodm/)
- [Agisoft](https://www.agisoft.com/)

Refer to the [docs/projection_tutorial.md](docs/projection_tutorial.md) to obtain point clouds and camera parameters from WebODM or Agisoft. The point clouds should use the format of [.las](https://laspy.readthedocs.io/en/latest/intro.html). 

For image segmentation, we have used Segment Anything Model from Meta: 
- [SAM](https://github.com/facebookresearch/segment-anything)


## Tutorials
1. Projecting Point Clouds onto Image Planes: [docs/projection_tutorial.md](docs/projection_tutorial.md)

2. Data Structure and Algorithms for Object Registration: [docs/object_registration.md](docs/object_registration.md)

3. Depth Image Rendering from Mesh and Depth Filtering: [docs/depth_filtering.md](docs/depth_filtering.md)

4. Camera Calibration Optimization: [docs/camera_calibration_optimization.md](docs/camera_calibration_optimization.md)


## Examples
We provide jupyter notebooks (Semantic_SfM/projects/) as examples to use the semantic SfM workflow.

## QnA

1. *When I import point clouds into CloudCompare, the points look sparse with strips. How can I fix it?* Such issues occur when point coordinates in CloudCompare are large. To fix the visualization issues, you can choose to automatically re-center the points when importing a point cloud. 

2. *My semantics in a resulting point cloud are shifted. Why does this happen?* During bundle adjustment in SfM (e.g., photo alignment in Agisoft), cameras are automatically calibrated. However, the calibration is not always optimal. When the resulted $c_x$, $c_y$ are large, the projected semantics on point clouds would be shifted. Please refer to [this tutorial](docs/camera_calibration_optimization.md) to optimize camera calibration. 

