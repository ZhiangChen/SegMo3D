# Benchmark Datasets for Image-based 3D segmentation


### Benchmark datasets
1. ScanNet

ScanNet v2 (http://www.scan-net.org/) is an RGB-D video dataset containing 2.5 million views in more than 1500 scans, annotated with 3D camera poses, surface reconstructions, and instance-level semantic segmentations.

2. ScanNet++

ScanNet ++ (https://kaldir.vc.in.tum.de/scannetpp/) contains approximately 3.7 million RGB-D frames captured with iPhones, along with 280,000 high-resolution images taken with DSLR cameras. ScanNet++ is an improved and extended version of the original ScanNet dataset, offering better quality data, more detailed annotations, and additional object classes.

3. KITTI

KITTI (https://www.cvlibs.net/datasets/kitti/index.php) is a large outdoor dataset for autonomous driving research, containing high-resolution stereo images captured from dual front-facing cameras, detailed 3D point clouds from a Velodyne LiDAR sensor, and precise localization data from GPS and IMU sensors. Additionally, it includes vehicle odometry data to track movement over time. The dataset offers extensive annotations, including 2D and 3D bounding boxes for object detection, pixel-wise labels for semantic segmentation, multi-frame data for object tracking, and optical flow data for understanding pixel motion between frames. 

4. KITTI-360

KITTI-360 (https://www.cvlibs.net/datasets/kitti-360/) is an extension of KITTI with 300 suburban scens, which consists of 320k images and 100k laser scans obtained through a mobile platform in driving distance of 73.7 km. 

5. SemanticKITTI

SemanticKITTI (http://www.semantic-kitti.org/dataset.html) extends the original KITTI dataset with semantic labels for each 3D point in the LiDAR scans. It provides 3D point-wise semantic labels and 2D semantic segmentation labels.

6. Cityscapes

Cityscapes (https://www.cityscapes-dataset.com/dataset-overview/) focuses on urban street scenes, providing high-quality pixel-level annotations for semantic segmentation. It provides 2D annotations for semantic segmentation and instance segmentation, and 3D annotations for depth maps. This dataset is useful for developing and benchmarking algorithms in urban scene understanding.

## Dataset generators

1. Kubric

Kubric (https://kubric.readthedocs.io/en/latest/) is an open-source dataset generation pipeline designed to create high-quality synthetic datasets for computer vision and machine learning research. It leverages Blender for 3D rendering and Bullet for physics simulation to produce diverse and realistic multi-modal data, including RGB images, depth maps, optical flow, and segmentation masks.

