# Tutorial: Data Structures and Algorithms for Object Registration

Zhiang Chen, Dec 22, 2023

Object (instances and stuffs) registration for point clouds considers the overlap between two objects from different camera poses. When the overlap is greater than a pre-defined threshold, the two objects will be merged. Otherwise, the objects will be registered independently. We use Intersection over Union (IoU) to quantify the overlap. Given two camera-object sets, $\{T_1, O_1\}$ and $\{T_2, O_2\}$, we calculate the IoU as follows: 

$intersection = Points(T_2O_1) \cap Points(T_1O_2) $,

$union = Points(T_2O_1) \cup Points(T_1O_2) $.

We have designed data structures and algorithms to improve the efficiency of the object registration. 

## Data Structures

### 1. Segmentation
Before object registration, images are segmented to create 2D masks with the same dimension as the images. In each mask, value -1 indicates that the corresponding pixel is not classified. Classified pixels have values ranging from 0 to N. 

### 2. Point2pixel association
For each image, the point-pixel association is represented by a 2D array with shape (N, 2), where N is the number of points in the point cloud. A pixel (u, v) is indexed by an integer i. u is the axis of height and v is the axis of width. If a point does not associate with any pixels, its corresponding pixel coordinates are (-1, -1). 

The point2pixel association is precomputed in parallel processing, as implemented by `ssfm.probabilistic_projection.parallel_batch_projection`. 

The point2pixel association is implemented by a camera projection with a depth filter. Segmentation is not considered in the point2pixel association.

### 3. Pixel2point association
A pixel2point association is a 2D array with shape (H, W), where H is the image height and W is image width. If a pixel does not associate with any points, its corresponding value is -1. Otherwise, the corresponding value in the pixel represents the point index. 

The pixel2point association is precomputed in parallel processing, as implemented by `ssfm.probabilistic_projection.parallel_batch_projection`. 

The pixel2point association is implemented by a camera projection with a depth filter. Segmentation is not considered in the pixel2point association.

### 4. keyimage association
The keyimage association is a 2D boolean array with shape (N, M), where N is the number of images and M is the number of total points in the point cloud. In each row, points with valid point-pixel associations and segmentation have values of True. 

The keyimage association is precomputed in `ssfm.probabilistic_projection.build_associations_keyimage`.

In `ssfm.object_registration.object_registration`, when registering a new object, K number of key images are selected from registered image candidates with the largest number of True values within the masks of the object.

### 5. Probabilistic segmented point cloud
We use two 2D arrays to represent a probabilistic segmented point cloud, `pc_segmentation_ids` and `pc_segmentation_probs`, both with the shape of (N, M), where N is the number of points in the point cloud and M is a pre-defined number to indicate the largest number of unique segmentation ids for each point. For example, if M is 5, each point has at most 5 unique segmentation ids associated with that point. `pc_segmentation_ids` includes the ids the segmentation ids. `pc_segmentation_probs` includes the likelihood of the corresponding ids. Note that the likelihood are accumulated from many images and not normalized, because if the likelihood are normalized, extra processing is needed to consider the weight of new image with respect to the previous registered ids. 

### 6. Object manager
*Probabilistic projection* is employed to (1) compensate for camera projection distortion and (2) segmentation errors. Each pixel in image segmentation has a list of probabilistic object ids from the pixel's neighbors. The probability of a neighbor is calculated from a Gaussian decaying function. 

For object registration, if immediately registering an new object by merging or creating a new object, we also need to update the probabilistic object ids for the points from the new object. However, registering a new object immediately on the fly may raise a problem, when the new object includes pixels (usually those pixels at object boundaries) with semantics of new objects that have not been registered. First, we don't know the ids of the unregistered new objects. Second, we don't know if these unregistered new objects will be registered or not, which determines their object ids. Therefore, to address this issue, instead of registering an new object on the fly, we introduce an idea of object manager. 

The object manager associates the object ids in the image to be registered and the registered object ids in the pointcloud segmentation. The object manager is a dictionary where the key is object ids in the image and the value is a list of the object ids in the pointcloud segmentation. If an object is not merged to any registered ones, the corresponding value will be an empty list. The reason of having a list for the value (instead of just one value for the registered object id) is due to the twin problem in object registration. The twin problem in 2D instance registration is explained in https://github.com/ZhiangChen/rsisa.  

During registering each object in a new image, only the object manager will be updated. After all objects are registered, we will use the object manager to update the pointcloud segmentation. 


## Algorithms

### 1. Object registration
```
for image in images:
    for object1 in objects:
        keyimages = get_keyimages(object1):
        for keyimage in keyimages:
            object2 = search_object2(object1, keyimage)
            point_object1_image2 = get_point_object1_image2()
            point_object2_image2 = get_point_object2_image2()
            point_object2_image1 = get_point_object2_image1()
            iou = calculate_IoU(point_object1_image2, point_object2_image1)
            update_object_manager(object1, point_object2_image2, iou)
    update_segmentation_pointcloud()
```

### 2. Search $O_2$
```
input: segmented_objects_image2, pixel_object1_image2

object_ids_object1_image2 = segmented_objects_image2[pixel_object1_image2]  # O(N)
object_id = sort_count(object_ids_object1_image2)  # O(N log N)
pixel_object2_image2 = segmented_objects_images.get_object(object_id)  # O(N)
```

### 3. Update object manager

```
if iou >= iou_threshold:
    registered_object_id = get_object_id(point_object2_image2)
    object_manager[object_id].append(registered_object_id)
```

The `get_object_id(.)` returns the object id with the largest summed likelihood. 

### 4. Purge pointcloud segmentation
When an object id in a new image has multiple unique object ids, which indicates the twin registration problem, we need to purge the repeated, registered object ids.  

### 5. Update pointcloud segmentation

