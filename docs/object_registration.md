# Tutorial: Data Structures and Algorithms for Object Registration

Zhiang Chen, Dec 22, 2023

Object (instances and stuffs) registration for point clouds considers the overlap between two objects from different camera poses. When the overlap is greater than a pre-defined threshold, the two objects will be merged. Otherwise, the objects will be registered independently. We use Intersection over Union (IoU) to quantify the overlap. Given two camera-object sets, $\{T_1, O_1\}$ and $\{T_2, O_2\}$, we calculate the IoU as follows: 

$intersection = Points(T_2O_1) \cap Points(T_1O_2) $,

$union = Points(T_2O_1) \cup Points(T_1O_2) $.

Designing data structures and algorithms is important to improve the efficiency of the object registration. For example, there are several computational challenges. First, exhaustively comparing any two objects is time-consuming, with a time complexity of $O(NM)$, where $N$ represents the number of registered images and $M$ denotes the maximum number of objects in any image. Second, because of the large of points, we want to avoid algorithms that have time complexity worse than $O(N log N)$. Ideally, we want the algorithms with constant time complexity. Third, if time-consuming algorithms cannot be avoided, we prefer parallel processing and preprocessing. Fourth, we need to design an efficient to method to calculate the IoU for 3D points.  

## Data Structures

### 1. Point-pixel association
For each image, the point-pixel association is represented by a 2D array of shape (N, 3), where N is the number of valid points that are projected onto an image. For each point(u, v, point_index), the first two elements are the u, v coordinate of the point in the image and the third element is the index of the point in the original points array. u is the axis of width and v is the axis of height.

The point-pixel association is precomputed in parallel processing, as implemented by `ssfm.probabilistic_projection.parallel_batch_projection`. 

### 2. Point-image association
The point-image association, `ObjectRegistration.association_p2i`, is a dictionary where the key is the point index and the value is a list of images that include the projection of the point. 


### 3. Image segmentation
The image segmentation, `ObjectRegistration.segmented_objects_image1`, is a 2D array with the same size as the image. Each pixel in the array is the index of the mask that the pixel belongs to. The valid index starts from 0. If the pixel does not belong to any mask, the value is -1.
        

### 4. Pointcloud segmentation
The pointcloud segmentation, `ObjectRegistration.pc_segmentation`, is a dictionary where the key is the point index and the value is a 2D array representing normalized object probabilities. The first column in the 2D array denotes the object index; the second column represents normalized probabilities. 

### 5. Object manager
*Probabilistic projection* is employed to (1) compensate for camera projection distortion and (2) segmentation errors. Each pixel in image segmentation has a list of probabilistic object ids from the pixel's neighbors. The probability of a neighbor is calculated from a Gaussian decaying function. 

For object registration, if immediately registering an new object by merging or creating a new object, we also need to update the probabilistic object ids for the points from the new object. However, registering a new object immediately on the fly may raise a problem, when the new object includes pixels (usually those pixels at object boundaries) with semantics of new objects that have not been registered. First, we don't know the ids of the unregistered new objects. Second, we don't know if these unregistered new objects will be registered or not, which determines their object ids. Therefore, to address this issue, instead of registering an new object on the fly, we introduce an idea of object manager. 

The object manager associates the object ids in the image to be registered and the registered object ids in the pointcloud segmentation. The object manager is a dictionary where the key is object ids in the image and the value is a list of the object ids in the pointcloud segmentation. If an object is not merged to any registered ones, the corresponding value will be an empty list. The reason of having a list for the value (instead of just one value for the registered object id) is due to the twin problem in object registration. The twin problem in 2D instance registration is explained in https://github.com/ZhiangChen/rsisa.  

During registering each object in a new image, only the object manager will be updated. After all objects are registered, we will use the object manager to update the pointcloud segmentation. 


## Algorithms

### 1. Object registration
```
pc_segmentation = initialize_pc_segmentation()  # initialize pointcloud segmentation
association_p2i = initialize_association_p2i()  # initialize point-image association
object_manager = initialize_object_manager()  # initialize object manager

M_images = len(segmentations)  # M is the number of images
for j in range(M_images):
    segmented_objects_image1 = segmentations[j]
    N_objects = len(segmented_objects_image1)
    association1 = associations[j]  # point-pixel association
    
    for i in range(N_objects):
        pixel_object1_image1 = segmented_objects[i]  # pixels of object 1 in image 1
        points_object1_image1 = get_points_from_pixel(association1, pixel_object1_image1) # points of object 1 from image 1
        key_images = get_key_images(association_p2i, points_object1_image1)
        for id_image2 in key_images:
            association2 = associations[id_image2]
            inv_association2 = inv(association2)
            pixel_object1_image2 = association2[points_object1_image1] # pixels of object 1 on image 2
            points_object1_image2 = inv_association2[pixel_object1_image2] # Point(T_2 O_1)

            segmented_objects_image2 = segmentations[id_image2]
            pixel_object2_image2 = object2_search(segmented_objects_image2, pixel_object1_image2)
            points_object2_image2 = get_points_from_pixel(association2, pixel_object2_image2)  # points of object 2 from image 2
            pixel_object2_image1 = association1[points_object2_image2]
            points_object2_image1 = get_points_from_pixel(association1, pixel_object2_image1) # Point(T_1 O_2)

            IoU = calculate_IoU(points_object1_image2, points_object2_image1)

            if IoU > threshold:
                update_object_manager(object_manager, pixel_object1_image1, segmented_objects_image1, points_object2_image2, merge=True)
            else:
                update_object_manager(object_manager, pixel_object1_image1, segmented_objects_image1, points_object2_image2=None, merge=False)

    update_association_p2i(association_p2i, j)
    purge_pointcloud_segmentation(object_manager)
    update_pointcloud_segmentation(pc_segmentation, object_manager, association1)
```

**Note**:  
All the indexing operations [.] have constant time complexity.   
association_p2i.get_key_images(.) has linear time complexity w.r.t. the number of valid points.  
inv(.) has linear time complexity w.r.t. the number of points.  
object2_search(.) has linearithmic time complexity w.r.t. the len(pixel_object1_image2).  
calculate_IoU(.) has linearithmic time complexity w.r.t. len(points_object1_image2) + len(points_object2_image1).  
update_object_manager has linearithmic time complexity w.r.t. len(points_object2_image2).   
update_pointcloud_segmentation(.) has time complexity linear to the number of points. 

### 2. Search $O_2$
```
input: segmented_objects_image2, pixel_object1_image2

object_ids_object1_image2 = segmented_objects_image2[pixel_object1_image2]  # O(N)
object_id = sort_count(object_ids_object1_image2)  # O(N log N)
pixel_object2_image2 = segmented_objects_images.get_object(object_id)  # O(N)
```

### 3. Update object manager

```
input: pixel_object1_image1, segmented_objects_image1, points_object2_image2, merge

if merge:
    object_id = segmented_objects_image1[pixel_object1_image1]
    registered_object_id = points_object2_image2.get_object_id()
    object_manager[object_id].append(registered_object_id)
else:
    pass
```

The `get_object_id(.)` returns the object id with the largest summed likelihood. 

### 4. Purge pointcloud segmentation
When an object id in a new image has multiple unique object ids, which indicates the twin registration problem, we need to purge the repeated, registered object ids.  

### 5. Update pointcloud segmentation
```
for pixel, point_id in association1:
    probabilistic_object_ids = calculate_probabilistic_object_ids(pixel)
    updated_probabilistic_object_ids = replace_object_id(probabilistic_object_ids, object_manager)
    if pc_segmentation[point_id] is Empty:
        pc_segmentation[point_id] = updated_probabilistic_object_ids
    else:
        registered_probabilistic_object_ids = pc_segmentation[point_id]
        pc_segmentation[point_id] = normalize_probabilistic_object_ids(registered_probabilistic_object_ids, updated_probabilistic_object_ids)

```
