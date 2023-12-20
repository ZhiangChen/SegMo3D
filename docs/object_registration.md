# Tutorial: Data Structure and Algorithm for Object Registration

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
The pointcloud segmentation is a dictionary where the key is the point index and the value is a list of instance probabilities. An instance probability is a dictionary where the key is instance index and the value is its corresponding normalized probability. 

## Algorithms

### 1. Object registration
```
association_p2i = initlize()  # initilize point-image association

M_images = len(segmentations)  # M is the number of images
for j in range(M_images):
    segmented_objects_image1 = segmentations[j]
    N_objects = len(segmented_objects_image1)
    association1 = associations[j]  # point-pixel association
    inv_association1 = inv(association1)  # pixel-point association
    update(association_p2i, j)

    for i in range(N_objects):
        pixel_object1_image1 = segmented_objects[i]  # pixels of object 1 in image 1
        points_object1_image1 = inv_association1[pixel_object1_image1] # points of object 1 from image 1
        key_images = association_p2i[points_object1_image1]
        for id_image2 in key_images:
            association2 = associations[id_image2]
            inv_association2 = inv(association2)
            pixel_object1_image2 = association2[points_object1_image1] # pixels of object 1 on image 2
            points_object1_image2 = inv_association2[pixel_object1_image2] # Point(T_2 O_1)

            segmented_objects_image2 = segmentations[id_image2]
            pixel_object2_image2 = object2_search(segmented_objects_images2, pixel_object1_image2)
            points_object2_image2 = inv_association2[pixel_object2_image2]  # points of object 2 from image 2
            pixel_object2_image1 = association1[points_object2_image2]
            points_object2_image1 = inv_assocation1[pixel_object2_image1] # Point(T_1 O_2)

            IoU = calculate_IoU(points_object1_image2, points_object2_image1)

            if IoU > threshold:
                update_pointcloud_segmentation(points_object1_image1, segmented_objects_image1, points_object2_image2, merge=True)
            else:
                update_pointcloud_segmentation(points_object1_image1, segmented_objects_image1, points_object2_image2=None, merge=False)

```

**Note**:  
All the indexing operations [.] have constant time complexity.   
inv(.) has linear time complexity.  
object2_search(.) has linearithmic time comlexity.  
calculate_IoU(.) has  
update_pointcloud_segmentation(.) has  

### 2. Search $O_2$

### 3. Calculate IoU

### 4. Update pointcloud segmentation


