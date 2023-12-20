# Tutorial: Data Structures and Algorithms for Object Registration

Zhiang Chen, Dec 19, 2023

Object (instances and stuffs) registration for point clouds considers the overlap between two objects from different camera poses. When the overlap is greater than a pre-defined threshold, the two objects will be merged. Otherwise, the objects will be registered independently. We use Intersection over Union (IoU) for the overlap. Given two camera-object sets, $\{T_1, O_1\}$ and $\{T_2, O_2\}$, we calculate the IoU as follows:

$intersection = Points(T_2O_1) \cap Points(T_1O_2) $,

$union = Points(T_2O_1) \cup Points(T_1O_2) $.

Designing data structures and algorithms is important to improve the efficiency of the object registration. For example, there are several computation challenges. First, comparing any objects is time-consuming, $O(NM)$, where $N$ is the number of registered images and $M$ is the maximum number of objects in any image. Second, we need to design an efficient to method to calculate the IoU. 

## Data Structures

### 1. Point-pixel association
For each image, the point-pixel association is represented by a 2D array of shape (N, 2), where N is the number of points. The index is the sequence number of the point and the value is the u, v coordinate of the point in the image. If the point is not projected onto the image, the value is [-1, -1]. The time complexity is O(N) where N is the number of points.

We also create a counterpart of the point-pixel association. The pixel-point association is a dictionary where the key is the u, v coordinate of the point in the image and the value is the index of the point. The time complexity is O(N) where N is the number of points. 

Query:  
point index -> pixel coordinates, O(1).
pixel coordinates -> point index, O(1).

### 2. Point-image association
The point-image association is dictionary where the key is the point index and the value is a list of images that include the projection of the point. 

Query:  
point index -> image list, O(1). 

### 3. Image segmentation
The image segmentation is a 2D array with the same size as the image, which might be downsized because of inference efficiency. Each pixel in the array is the index of the mask that the pixel belongs to. The valid index starts from 0. If the pixel does not belong to any mask, the value is -1.
        

Query:  
object index -> pixel coordinates of the object: python filtering   

### 4. Pointcloud segmentation
The pointcloud segmentation is a dictionary where the key is the point index and the value is a list of instance probabilities. An instance probability is a dictionary where the key is instance index and the value is its corresponding normalized probability. The pointcloud segmentation is the final result we expect from the object registration algorithms. 


### 5. Object manager
*Probabilistic projection* is employed to (1) compensate for camera projection distortion and (2) segmentation errors. Each pixel in image segmentation has a list of probabilistic object ids from the pixel's neighbors. The probability of a neighbor is calculated from a Gaussian decaying function. 

For object registration, if immediately registering an new object by neigher merging or creating a new object, we also need to update the probabilistic object ids for the points from the new object. However, if there are any object ids that have not been registered, it may raise problems of registering them with new object ids, because later on the object ids may be mergered to other registered objects. Therefore, to address this issue, we introduce an idea of object manager. 

The object manager associates the object ids in the image to be registered and the registered object ids in the pointcloud segmentation. The object manager is a dictionary where the key is object ids in the image and the value is the object ids in the pointcloud segmentation. If an object is not merged to any registered ones, the corresponding value will be assigned as None. 

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
    inv_association1 = inv(association1)  # pixel-point association
    
    for i in range(N_objects):
        pixel_object1_image1 = segmented_objects[i]  # pixels of object 1 in image 1
        points_object1_image1 = inv_association1[pixel_object1_image1] # points of object 1 from image 1
        key_images = association_p2i.get_key_images(points_object1_image1)
        for id_image2 in key_images:
            association2 = associations[id_image2]
            inv_association2 = inv(association2)
            pixel_object1_image2 = association2[points_object1_image1] # pixels of object 1 on image 2
            points_object1_image2 = inv_association2[pixel_object1_image2] # Point(T_2 O_1)

            segmented_objects_image2 = segmentations[id_image2]
            pixel_object2_image2 = object2_search(segmented_objects_image2, pixel_object1_image2)
            points_object2_image2 = inv_association2[pixel_object2_image2]  # points of object 2 from image 2
            pixel_object2_image1 = association1[points_object2_image2]
            points_object2_image1 = inv_assocation1[pixel_object2_image1] # Point(T_1 O_2)

            IoU = calculate_IoU(points_object1_image2, points_object2_image1)

            if IoU > threshold:
                update_instance_manager(instance_manager, points_object1_image1, segmented_objects_image1, points_object2_image2, merge=True)
            else:
                update_instance_manager(instance_manager, points_object1_image1, segmented_objects_image1, points_object2_image2=None, merge=False)

    update(association_p2i, j)
    update_pointcloud_segmentation(pc_segmentation, instance_manager)

```

**Note**:  
All the indexing operations [.] have constant time complexity.   
association_p2i.get_key_images(.) has linear time complexity w.r.t. the number of valid points.  
inv(.) has linear time complexity w.r.t. the number of points.  
object2_search(.) has linearithmic time comlexity.  
calculate_IoU(.) has  
update_pointcloud_segmentation(.) has  

### 2. Search $O_2$
```
object_ids_object1_image2 = segmented_objects_image2[pixel_object1_image2]  # O(N)
object_id = sort_count(object_ids_object1_image2)  # O(N log N)
pixel_object2_image2 = segmented_objects_images.get_instance(object_id)  # O(N)
```

### 3. Calculate IoU
```
points1 = filter_valid_points(points_object1_image2)
points2 = filter_valid_points(points_object2_image1)
points1_hash = hash(points1)
points2_hash = hash(points2)
points_hash = concatenate(points1_hash, points2_hash)
intersection = isin(points1_hash, points2_hash)
union = unique(points_hash)
IoU = len(intersection) / len(union)
```

The Numpy functions $isin()$ and $unique()$ have efficient implementations by utilizing merge sort algorithms, both with time complexity of $O(P\cdot log_2(P))$.

### 4. Update object manager
object manager

```

update_pointcloud_segmentation(pc_segmentation, points_object1_image1, segmented_objects_image1, points_object2_image2, merge=True)


if merge:
    

else:



```

