# semantic_SfM
Instance/panoptic segmentation for photogrammetry point cloud. 

## Installation
Option 1 (recommended): 
```
git clone https://github.com/ZhiangChen/semantic_SfM.git
cd semantic_SfM/semantic_SfM/ssfm/
pip3 install .
```

Option 2 (TBD):
```
pip3 install ssfm
```

**Hardware requirements**: This project utilizes the Segment Anything Model (SAM) for panoptic segmentation. GPUs are not required for SAM but are recommended to expedite inference speed. The other parts of this project use only CPUs. As a point cloud will be stored in memory, memory usage depends on the point cloud size. 

## SSfM Architecture
[files.py](./semantic_SfM/ssfm/files.py) provides utility functions for file operation, such as point cloud and image files.

[image_segmentation.py](./semantic_SfM/ssfm/image_segmentation.py) allows to select deep learning models for instance segmentation and panoptic segmentation. 

[probabilistic_projection.py](./semantic_SfM/ssfm/probabilistic_projection.py) projects point clouds to images and creates a probablistic semantics.

[object_registration.py](./semantic_SfM/ssfm/object_registration.py) registers objects (instances and stuffs) in point clouds.

[workflow.py](./semantic_SfM/ssfm/workflow.py) combines the processes and provides the users an interface to use semantic_SfM. 

## Support interface
SfM: [WebODM](https://opendronemap.org/webodm/) and [Agisoft](https://www.agisoft.com/).

Image segmentation: [SAM](https://github.com/facebookresearch/segment-anything).

Point clouds: .las format. Refer to the [docs/projection_tutorial.md](docs/projection_tutorial.md) to obtain point clouds and camera parameters from WebODM or Agisoft. 


 
## Tutorials
1. Project point clouds to image planes: [docs/projection_tutorial.md](docs/projection_tutorial.md)
