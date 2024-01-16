# to do:
# 1. read inv_associations
# maybe it's better to use (u, v, point_index) to combine inv_associations and associations. maybe this is more efficient to filter out points. 
# 2. more efficient on the process in object_registration

from ssfm.files import *
from ssfm.image_segmentation import *
from ssfm.probabilistic_projection import *

import os
import time
import numpy as np
from collections import defaultdict
import laspy


def group_lists(lists):
    """
    Group lists that share common elements.
    
    Parameters
    ----------
    lists : list of lists
    
    Returns
    -------
    grouped_lists : list of lists
    
    
    Example
    -------
    lists = [
        [17, 2, 3],
        [3, 4, 5],
        [5, 6, 7],
        [8, 9, 10],
        [10, 11, 2],
        [12, 13]
        ]
    grouped_lists = group_lists(lists)
    print(grouped_lists)
    [[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 17], [12, 13]]
    """
    element_to_list_map = defaultdict(list)
    for list_index, elements in enumerate(lists):
        for element in elements:
            element_to_list_map[element].append(list_index)

    # Find connected components of the list indices graph
    def dfs(list_index, visited, group):
        visited[list_index] = True
        group.append(list_index)
        for neighbour in adjacency_list[list_index]:
            if not visited[neighbour]:
                dfs(neighbour, visited, group)

    # Create an adjacency list for the graph where each node represents a list
    # and an edge connects lists that share at least one element
    adjacency_list = defaultdict(set)
    for indices in element_to_list_map.values():
        for list_index in indices:
            adjacency_list[list_index].update(indices)
            adjacency_list[list_index].remove(list_index)

    # Use DFS to find all connected components of the graph
    visited = [False] * len(lists)
    groups = []
    for list_index in range(len(lists)):
        if not visited[list_index]:
            group = []
            dfs(list_index, visited, group)
            groups.append(group)

    # Group the lists according to connected components
    grouped_lists = []
    for group in groups:
        grouped_list = set()
        for list_index in group:
            grouped_list.update(lists[list_index])
        grouped_lists.append(sorted(grouped_list))

    return grouped_lists


@jit(nopython=True)
def create_boolean_mask(mask, pixel_objects_image1):
    for pixel in pixel_objects_image1:
                mask[pixel[0], pixel[1]] = True
    return mask

class ObjectRegistration(object):
    def __init__(self, pointcloud_path, segmentation_folder_path, association_folder_path):
        self.pointcloud_path = pointcloud_path
        self.segmentation_folder_path = segmentation_folder_path
        self.association_folder_path = association_folder_path

        # Load segmentation files (.npy) and sort them
        # check if the folder exists
        assert os.path.exists(self.segmentation_folder_path), 'Segmentation folder path does not exist.'
        self.segmentation_file_paths = [os.path.join(self.segmentation_folder_path, f) for f in os.listdir(self.segmentation_folder_path) if f.endswith('.npy')]
        self.segmentation_file_paths.sort()   

        # Load association files (.npy) and sort them
        # check if the folder exists
        assert os.path.exists(self.association_folder_path), 'Association folder path does not exist.'
        self.association_file_paths = [os.path.join(self.association_folder_path, f) for f in os.listdir(self.association_folder_path) if f.endswith('.npy')]
        self.association_file_paths.sort()

        # Check if the number of segmentation files and association files are the same
        assert len(self.segmentation_file_paths) == len(self.association_file_paths), 'The number of segmentation files and association files are not the same.'

        # create segmentation-association pairs
        self.segmentation_association_pairs = []
        for i in range(len(self.segmentation_file_paths)):
            self.segmentation_association_pairs.append((self.segmentation_file_paths[i], self.association_file_paths[i]))

        # print the number of segmentation-association pairs
        print('Number of segmentation-association pairs: {}'.format(len(self.segmentation_association_pairs)))

        # initialize data structures
        self.association_p2i = dict()  # the key is the point index and the value is a list of images that include the projection of the point.
        self.pc_segmentation = dict()  # the key is the point index and the value is a list of object probabilities.
        self.latest_registered_id = -1  # the latest registered object id
        self.associations_pixel2point = []
        self.associations_point2pixel = []
        self.masks = []

        # pre-compute gaussian weights
        self.radius = 0
        self.decaying = 2
        self.likelihoods = compute_gaussian_likelihood(radius=self.radius, decaying=self.decaying)

        # debug

    
    def update_association_p2i(self, point_object1_image1, image_index):
        """
        Update association_p2i where the key is the point index and the value is a list of images that include the projection of the point.

        Parameters
        ----------
        point_object1_image1 : list of point indices
        image_index : int

        Returns
        -------
        None
        """
        for point in point_object1_image1:
            if point not in self.association_p2i.keys():
                self.association_p2i[point] = [image_index]
            else:
                self.association_p2i[point].append(image_index)

    def search_object2(self, segmented_objects_image2, pixel_object1_image2):
        """
        Within pixels of object1 in image2, search for object2 that has the largest number of semantics ids. 

        Parameters
        ----------
        segmented_objects_image2 : 2D array of shape (width, height), where each element is an object id
        pixel_object1_image2 : 2D array of shape (N_pixels, 2), where each row is a pixel coordinate (u, v)

        Returns
        -------
        pixel_object2_image2 : list of point indices
        """
        pixel_object1_image2_ = np.array(pixel_object1_image2)
        # Convert pixel_object1_image2 to a tuple of arrays for advanced indexing
        pixel_indices = (pixel_object1_image2_[:, 0], pixel_object1_image2_[:, 1])

        # Get object IDs directly using advanced indexing
        object_ids_object1_image2 = segmented_objects_image2[pixel_indices]

        # Find the object ID with the maximum count
        unique_ids, counts = np.unique(object_ids_object1_image2, return_counts=True)
        max_count_id = unique_ids[np.argmax(counts)]

        # Get pixel coordinates of the object with the maximum count
        pixel_object2_image2 = np.argwhere(segmented_objects_image2 == max_count_id)

        return pixel_object2_image2

    def calculate_3D_IoU(self, point_object1_image2, point_object2_image1):
        """
        Calculate 3D IoU between object1 in image2 and object2 in image1.

        Parameters
        ----------
        point_object1_image2 : list of point indices
        point_object2_image1 : list of point indices

        Returns
        -------
        iou : float
        """
        intersection = np.count_nonzero(np.in1d(point_object1_image2, point_object2_image1, assume_unique=True))
        union = len(point_object1_image2) + len(point_object2_image1) - intersection
        iou = intersection / union
        return iou

    def update_object_manager(self, pixel_object1_image1, segmented_objects_image1, point_object2_image2):
        """
        Update object_manager where the key is the object id from segmented_objects_image1 with pixel_object1_image1 
        and the value is a list of registered object ids that are associated with the object id. If point_object2_image2 
        is None, the value is []. Otherwise, the value is a list of the unique object ids with the maximum probabilities.
        
        Parameters
        ----------
        pixel_object1_image1 : 2D array of shape (N_pixels, 2), where each row is a pixel coordinate (u, v)
        segmented_objects_image1 : 2D array of shape (width, height), where each element is an object id
        point_object2_image2 : list of point indices
        
        Returns
        -------
        None
        """
        # get object id from pixel_object1_image1 and segmented_objects_image1
        object_id = int(segmented_objects_image1[tuple(pixel_object1_image1[0])]) 
        print('object_id: {}'.format(object_id))

        if point_object2_image2 == None:  # if point_object2_image2 is None, update the object_id with None
            registered_objects_id = None
        else:  # if point_object2_image2 is not None, get the object id with the maximum probability
            point_object_prob_sum = dict()
            for point in point_object2_image2:
                point_object_probs = self.pc_segmentation[point]  # dictionary of object probabilities for a point, where the key is the object id and the value is the normalized object probability
                for object_id_, prob in point_object_probs.items():
                    if object_id_ not in point_object_prob_sum.keys():
                        point_object_prob_sum[object_id_] = prob
                    else:
                        point_object_prob_sum[object_id_] += prob
            
            registered_objects_id = max(point_object_prob_sum, key=point_object_prob_sum.get)
        
        # if there is no object id in object_manager, add the object id and registered_objects_id to object_manager
        if object_id not in self.object_manager.keys():
            if registered_objects_id == None:
                self.object_manager[object_id] = []
            else:
                self.object_manager[object_id] = [registered_objects_id]
        else:
            if registered_objects_id == None:
                pass
            else:
                if registered_objects_id not in self.object_manager[object_id]:
                    self.object_manager[object_id].append(registered_objects_id)
                else:
                    pass
    
    def purge_pc_segmentation(self):
        """
        Purge self.pc_segmentation where the key is the point index and the value is a list of object probabilities. 
        We use the object_manager to purge pc_segmentation.
        object_manager is a dictionary where the key is the object id and the value is a list of registered object ids. 
        When the value is the list of different registered object ids, we combine and purge the registered object ids. 
        At the same time, the probabilities of the registered object ids are combined and normalized.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        ---------------- 
        object_manager: the key is the object id and the value is a list of registered object ids.
        self.pc_segmentation: the key is the point index and the value is a list of object probabilities.
        """
        registered_object_id_list = list(self.object_manager.values())
        group_registered_object_id_list = group_lists(registered_object_id_list)

        purge_object_id_map = dict()  # the key is the object id and the value is a list of registered object ids that need to be purged.
        for group_registered_object_id in group_registered_object_id_list:
            if len(group_registered_object_id) <= 1:
                pass
            else:
                for object_id in group_registered_object_id[1:]:
                    purge_object_id_map[object_id] = group_registered_object_id[0]

        # iterate over self.pc_segmentation
        for point, object_probs in self.pc_segmentation.items():
            # get object ids from object_probs
            object_ids = list(object_probs.keys())
            # iterate over object_ids
            for object_id in object_ids:
                if object_id in purge_object_id_map.keys():
                    # get purge_object_prob
                    purge_object_prob = object_probs[object_id]
                    # get purge_object_id
                    purge_object_id = purge_object_id_map[object_id]
                    # update object_probs
                    if purge_object_id not in object_probs.keys():
                        object_probs[purge_object_id] = purge_object_prob
                    else:
                        object_probs[purge_object_id] += purge_object_prob
                    del object_probs[object_id]
                else:
                    pass
        
        # purge self.object_manager
        for object_id, registered_object_ids in self.object_manager.items():
            registered_object_ids_copy = registered_object_ids.copy()
            for registered_object_id in registered_object_ids_copy: 
                if registered_object_id in purge_object_id_map.keys(): 
                    keep_object_id = purge_object_id_map[registered_object_id] 
                    if keep_object_id in registered_object_ids:
                        registered_object_ids.remove(registered_object_id)
                    else:
                        registered_object_ids.append(keep_object_id)
                        registered_object_ids.remove(registered_object_id)
                else:
                    pass
        
        # check the length of values in self.object_manager
        for object_id, registered_object_ids in self.object_manager.items():
            if len(registered_object_ids) > 1:
                raise ValueError('The length of registered_object_ids is not 1.')
            else:
                pass
        
    def update_pc_segmentation(self, associations1_pixel2point, segmented_objects_image1):
        """
        Update self.pc_segmentation where the key is the point index and the value is a list of object probabilities. 
        We use the object_manager to update pc_segmentation and use probabilistic projection to update probabilities.

        Parameters
        ----------
        associations1_pixel2point : dictionary where the key is the pixel coordinate (u, v) and the value is the point index
        segmented_objects_image1 : 2D array of shape (width, height), where each element is an object id
        
        Returns
        -------
        None
        ---------------- 
        object_manager: the key is the object id and the value is a list of registered object ids.
        self.pc_segmentation: the key is the point index and the value is a list of object probabilities.
        """
        # iterate over associations1_pixel2point
        for pixel, point in associations1_pixel2point.items():
            # get object id from segmented_objects_image1
            object_id = int(segmented_objects_image1[pixel])
            if object_id == -1:  # this pixel is not associated with semantics
                continue
            else:
                # get normalized_likelihoods
                u, v = pixel
                normalized_likelihoods = inquire_semantics(v, u, self.padded_segmentation, self.normalized_likelihoods, self.likelihoods, radius=self.radius)
                                    
                # get registered object ids from object_manager
                registered_objects_id = self.object_manager[object_id]

                new_likelihoods_dict = dict()
                for i in range(len(normalized_likelihoods)):
                    if normalized_likelihoods[i] > 0:
                        registered_object_ids_kernel = self.object_manager[i]
                        if registered_object_ids_kernel == []:
                            new_likelihoods_dict[i+self.latest_registered_id] = normalized_likelihoods[i]
                        else:
                            if registered_object_ids_kernel[0] not in new_likelihoods_dict.keys():
                                new_likelihoods_dict[registered_object_ids_kernel[0]] = normalized_likelihoods[i]
                            else:
                                new_likelihoods_dict[registered_object_ids_kernel[0]] += normalized_likelihoods[i]
                    else:
                        pass

                if registered_objects_id == []: # this object is not registered
                    self.pc_segmentation[point] = new_likelihoods_dict

                else:  # this object is registered
                    if point not in self.pc_segmentation.keys():
                        self.pc_segmentation[point] = new_likelihoods_dict
                    else:
                        original_likelihoods_dict = self.pc_segmentation[point]  
                        # combine original_likelihoods_dict and new_likelihoods_dict
                        for object_id, prob in new_likelihoods_dict.items():
                            if object_id not in original_likelihoods_dict.keys():
                                original_likelihoods_dict[object_id] = prob
                            else:
                                original_likelihoods_dict[object_id] += prob
                        # normalize original_likelihoods_dict
                        total_prob = sum(original_likelihoods_dict.values())
                        for object_id, prob in original_likelihoods_dict.items():
                            original_likelihoods_dict[object_id] = prob / total_prob

                        self.pc_segmentation[point] = original_likelihoods_dict
        
        # update self.latest_registered_id
        self.latest_registered_id += len(self.normalized_likelihoods)

    def extract_semantic_pointcloud(self, save_las_path):
        """
        Extract semantic pointcloud from self.pc_segmentation (.las). Read pointcloud from self.pointcloud_path. Iterate over points in pointcloud. When a point is in self.pc_segmentation, extract the semantics id with the maximum probability from self.pc_segmentation and add the semantics id to the attributes of the point. If a point is not in self.pc_segmentation, add -1 to the attributes of the point. Save the semantic pointcloud to a .las file.

        Parameters
        ----------
        save_las_path : str, path to save the semantic pointcloud
        
        Returns
        -------
        None
        """
        # read pointcloud from self.pointcloud_path
        points, colors = read_las_file(self.pointcloud_path)
        semantics = []
        # iterate over points
        N = points.shape[0]
        for i in range(N):
            if i in self.pc_segmentation.keys():
                # get semantics id with the maximum probability
                semantics_id = max(self.pc_segmentation[i], key=self.pc_segmentation[i].get)
                semantics.append(semantics_id)
            else:
                semantics.append(-1)
        
        semantics = np.array(semantics)

        # construct a .las file
        hdr = laspy.LasHeader(version="1.2", point_format=3)
        hdr.scale = [0.0001, 0.0001, 0.0001]  # Example scale factor, adjust as needed
        hdr.offset = np.min(points, axis=0)

        # Create a LasData object
        las = laspy.LasData(hdr)

        # Add points
        las.x = points[:, 0]
        las.y = points[:, 1]
        las.z = points[:, 2]

        # Add colors
        las.red = colors[:, 0]
        las.green = colors[:, 1]
        las.blue = colors[:, 2]

        # Add semantics
        las.user_data = semantics

        # Write the LAS file
        las.write(save_las_path)

    def object_registration(self, iou_threshold=0.75):
        M_images = len(self.segmentation_association_pairs)

        # iterate over all images
        for i in range(M_images):
            # load segmentation and association files
            segmentation_file_path, association_file_path = self.segmentation_association_pairs[i]
            print(segmentation_file_path, association_file_path)
            t1 = time.time()
            segmented_objects_image1 = np.load(segmentation_file_path).transpose(1, 0).astype(np.int16)  # transpose from (height, width) to (width, height)
            associations1 = np.load(association_file_path)  # association1 is 2D array for point-pixel association, A 2D array of shape (N, 3) where N is the number of valid points that are projected onto the image. Each row is (u, v, point_index). 
            t2 = time.time()
            print('Time to load segmentation and association files: {}'.format(t2 - t1))

            self.point_count = np.zeros(segmented_objects_image1.max() + 1, dtype=np.int32)
            
            # get image height and width of segmented_objects_image1
            image_height, image_width = segmented_objects_image1.shape
            # pad segmentation image by the size of radius with -1
            self.padded_segmentation = -np.ones((2*self.radius+image_height+2, 2*self.radius+image_width+2))
            self.padded_segmentation[self.radius+1:self.radius+image_height+1, self.radius+1:self.radius+image_width+1] = segmented_objects_image1

            # Allocate normalized_likelihoods outside the loop
            self.normalized_likelihoods = np.zeros(int(segmented_objects_image1.max() + 1), dtype=np.float64)

            N_objects = int(segmented_objects_image1.max() + 1)
            print('Number of objects in image {}: {}'.format(i, N_objects))

            pixel_objects_image1 = associations1[:, :2]  # 2D array of shape (N, 2), where each row is a pixel coordinate (u, v)

            t2 = time.time()
            # create a dictionary for associations1_pixel2point where the key is the pixel coordinate (u, v) and the value is the point index
            # create a dictionary for associations1_point2pixel where the key is the point index and the value is the pixel coordinate (u, v)
            """associations1_pixel2point = dict()
            associations1_point2pixel = dict()
            for association in associations1:
                associations1_pixel2point[tuple(association[:2])] = association[2]
                associations1_point2pixel[association[2]] = tuple(association[:2])"""
            associations1_pixel2point = {tuple(association[:2]): association[2] for association in associations1}
            associations1_point2pixel = {association[2]: tuple(association[:2]) for association in associations1}
            self.associations_pixel2point.append(associations1_pixel2point)
            self.associations_point2pixel.append(associations1_point2pixel)
            t3 = time.time()
            print('Time to create dictionaries for associations1: {}'.format(t3 - t2))

            # create a boolean mask for pixel_objects_image1
            t2 = time.time()
            pixels_array = np.array(list(pixel_objects_image1))
            rows, cols = pixels_array[:, 0], pixels_array[:, 1]
            mask1 = np.zeros(segmented_objects_image1.shape, dtype=bool)
            mask1[rows, cols] = True
            self.masks.append(mask1)
            t3 = time.time()
            print('Time to create a boolean mask for pixel_objects_image1: {}'.format(t3 - t2))

            self.object_manager = dict()  # the key is the object id and the value is a list of registered object ids.

            
            # iterate over all objects in image1
            for j in range(N_objects):
                # get pixel coordinates of segmented_objects_image1 == j
                t4 = time.time()
                pixel_object1_image1 = np.argwhere(segmented_objects_image1 == j)  # 2D array of shape (N_pixels, 2), where each row is a pixel coordinate (u, v)
                t5 = time.time()
                print('Time to get pixel coordinates of segmented_objects_image1 == {}: {}'.format(j, t5 - t4))
                

                # get point indices of pixel_object1_image1 using association1. However, not all pixels in pixel_object1_image1 have a corresponding point index.
                point_object1_image1 = [associations1_pixel2point[tuple((p[0], p[1]))] for p in pixel_object1_image1 if mask1[p[0], p[1]]]
                
                self.point_count[j] = len(point_object1_image1)

                t6 = time.time()
                print('Time to get point indices of pixel_object1_image1: {}'.format(t6 - t5))

                # get key images of point_object1_image1
                key_images_lists = [self.association_p2i[point] for point in point_object1_image1 if point in self.association_p2i.keys()]
                key_images = np.unique([item for sublist in key_images_lists for item in sublist])
            
                t7 = time.time()
                print('Time to get key images: {}'.format(t7 - t6))

                t7 = time.time()
                # update association_p2i
                self.update_association_p2i(point_object1_image1, i)
                t8 = time.time()
                print('Time to update association_p2i: {}'.format(t8 - t7))

                if len(key_images) == 0:
                    # update object_manager
                    self.update_object_manager(pixel_object1_image1, segmented_objects_image1, None)
                    t9 = time.time()
                    print('Time to update object_manager: {}'.format(t9 - t8))
                else:
                    # iterate over all key images
                    for key_image in key_images:
                        # get associations2 for key_image
                        associations2_pixel2point = self.associations_pixel2point[key_image]
                        associations2_point2pixel = self.associations_point2pixel[key_image]
                        mask2 = self.masks[key_image]

                        t1 = time.time()
                        pixel_object1_image2 = [associations2_point2pixel[point] for point in point_object1_image1 if point in associations2_point2pixel.keys()]
                        point_object1_image2 = [associations2_pixel2point[tuple((p[0], p[1]))] for p in pixel_object1_image2 if  mask2[p[0], p[1]]]
                        t2 = time.time()
                        print('Time to get point_object1_image2: {}'.format(t2 - t1))

                        segmentation_file_path, _ = self.segmentation_association_pairs[key_image]
                        segmented_objects_image2 = np.load(segmentation_file_path).transpose(1, 0).astype(np.int16)  # transpose from (height, width) to (width, height)
                        t1 = time.time()
                        pixel_object2_image2 = self.search_object2(segmented_objects_image2, pixel_object1_image2)
                        t2 = time.time()
                        print('Time to search object2: {}'.format(t2 - t1))

                        t1 = time.time()
                        point_object2_image2 = [associations2_pixel2point[tuple((p[0], p[1]))] for p in pixel_object2_image2 if mask2[p[0], p[1]]]
                        t2 = time.time()
                        print('Time to get point_object2_image2: {}'.format(t2 - t1))

                        pixel_object2_image1 = [associations1_point2pixel[point] for point in point_object2_image2 if point in associations1_point2pixel.keys()]
                        t3 = time.time()
                        print('Time to get pixel_object2_image1: {}'.format(t3 - t2))

                        point_object2_image1 = [associations1_pixel2point[tuple((p[0], p[1]))] for p in pixel_object2_image1 if mask1[p[0], p[1]]]
                        t4 = time.time()
                        print('Time to get point_object2_image1: {}'.format(t4 - t3))

                        iou = self.calculate_3D_IoU(point_object1_image2, point_object2_image1)
                        print('iou: {}'.format(iou))
                        t5 = time.time()
                        print('Time to calculate 3D IoU: {}'.format(t5 - t4))

                        if iou >= iou_threshold:
                            # update object_manager
                            self.update_object_manager(pixel_object1_image1, segmented_objects_image1, point_object2_image2)
                            t5 = time.time()
                            print('Time to update object_manager: {}'.format(t5 - t4))
                        else:
                            # update object_manager
                            self.update_object_manager(pixel_object1_image1, segmented_objects_image1, None)
                            t5 = time.time()
                            print('Time to update object_manager: {}'.format(t5 - t4))

                print('---------------------------------------------------')

            # purge self.pc_segmentation using self.object_manager
            self.purge_pc_segmentation()
            # print object_manager
            print('object_manager: {}'.format(self.object_manager))
            # update self.pc_segmentation
            self.update_pc_segmentation(associations1_pixel2point, segmented_objects_image1)


            print('===================================================')
        


                


if __name__ == "__main__":
    # Set paths
    pointcloud_path = '../../data/model.las'
    segmentation_folder_path = '../../data/mission_2_segmentations_test'
    image_folder_path = '../../data/mission_2'
    association_folder_path = '../../data/mission_2_associations_test'

    # Create object registration object
    t1 = time.time()
    object_registration = ObjectRegistration(pointcloud_path, segmentation_folder_path, association_folder_path)
    t2 = time.time()
    print('Time to create object registration object: {}'.format(t2 - t1))
    object_registration.object_registration()
    object_registration.extract_semantic_pointcloud('../../data/semantic_pointcloud.las')


"""
semantics = []
N = points.shape[0]
for i in range(N):
    if i in object_registration.pc_segmentation.keys():
        # get semantics id with the maximum probability
        semantics_id = max(object_registration.pc_segmentation[i], key=object_registration.pc_segmentation[i].get)
        semantics.append(semantics_id)
    else:
        semantics.append(-1)

semantics = np.array(semantics)
"""