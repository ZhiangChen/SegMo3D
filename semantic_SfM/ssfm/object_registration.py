from ssfm.files import *
from ssfm.image_segmentation import *
from ssfm.probabilistic_projection import *

import os
import time
import numpy as np
from collections import defaultdict
import logging
import yaml


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
def numba_update_pc_segmentation(associations1_point2pixel, segmented_objects_image1, object_manager_array, latest_registered_id, M_segmentation_ids, radius, padded_segmentation, normalized_likelihoods, likelihoods, pc_segmentation_ids, pc_segmentation_probs):
    N_points = associations1_point2pixel.shape[0]
    for point_id in range(N_points):
        u,v = associations1_point2pixel[point_id]
        if u != -1:
            object_id = segmented_objects_image1[u, v]
            if object_id == -1:
                continue
            normalized_likelihoods = inquire_semantics(u, v, padded_segmentation, normalized_likelihoods, likelihoods, radius)
            
            # construct an array where the first column is the object id and the second column is the normalized object probability
            new_likelihoods_array = np.zeros((len(normalized_likelihoods), 2), dtype=np.float32)

            for i in range(len(normalized_likelihoods)):
                if normalized_likelihoods[i] > 0.001:
                    registered_object_ids_kernel = object_manager_array[i, 1]
                    if registered_object_ids_kernel == -1:
                        new_likelihoods_array[i, 0] = i+latest_registered_id
                        new_likelihoods_array[i, 1] = normalized_likelihoods[i]
                    else:
                        new_likelihoods_array[i, 0] = registered_object_ids_kernel
                        new_likelihoods_array[i, 1] = normalized_likelihoods[i]
                else:
                    pass

            # obtain the top M_segmentation_ids object ids with the highest probabilities in new_likelihoods_array
            sorted_new_likelihoods_array = new_likelihoods_array[new_likelihoods_array[:, 1].argsort()[::-1]]
            sorted_new_likelihoods_array = sorted_new_likelihoods_array[:M_segmentation_ids]
            
            # obtain original_segmentation_ids and original_segmentation_probs
            original_segmentation_ids = pc_segmentation_ids[point_id]
            original_segmentation_probs = pc_segmentation_probs[point_id]

            # combine original_segmentation_ids and original_segmentation_probs with sorted_new_likelihoods_array
            for object_id, prob in sorted_new_likelihoods_array:
                if object_id in original_segmentation_ids:
                    index = np.where(original_segmentation_ids == object_id)
                    original_segmentation_probs[index] += prob
                else:
                    if -1 in original_segmentation_ids:
                        index = np.where(original_segmentation_ids == -1)[0][0]
                        original_segmentation_ids[index] = object_id
                        original_segmentation_probs[index] = prob
                    else:
                        index = np.argmin(original_segmentation_probs)
                        if prob > original_segmentation_probs[index]:
                            original_segmentation_ids[index] = object_id
                            original_segmentation_probs[index] = prob
                        else:
                            pass

            # normalize original_segmentation_probs
            # original_segmentation_probs /= np.sum(original_segmentation_probs)

            # update pc_segmentation_ids and pc_segmentation_probs
            #pc_segmentation_ids[point_id] = original_segmentation_ids
            #pc_segmentation_probs[point_id] = original_segmentation_probs


class ObjectRegistration(object):
    def __init__(self, pointcloud_path, segmentation_folder_path, association_folder_path, keyimage_associations_file_name=None, keyimage_yaml_name=None, loginfo=True):
        self.pointcloud_path = pointcloud_path
        self.segmentation_folder_path = segmentation_folder_path
        self.association_folder_path = association_folder_path
        self.loginfo = loginfo

        if self.loginfo:
            logging.basicConfig(filename="object_registration.log", 
                                format='%(asctime)s %(message)s', 
                                filemode='w') 

            self.logger = logging.getLogger() 
            self.logger.setLevel(logging.DEBUG) 

        # check if the pointcloud file exists
        assert os.path.exists(self.pointcloud_path), 'Pointcloud path does not exist.'

        if pointcloud_path.endswith('.las'):
            with laspy.open(self.pointcloud_path) as las_file:
                header = las_file.header
                self.N_points = header.point_count
        elif pointcloud_path.endswith('.npy'):
            mesh_vertices_color = np.load(pointcloud_path)
            points = mesh_vertices_color[0]
            colors = mesh_vertices_color[1]
            self.N_points = points.shape[0]

        # load keyimage association files (.npy)
        if keyimage_associations_file_name is None:
            keyimage_associations_file_path = os.path.join(self.association_folder_path, 'associations_keyimage.npy')
            assert os.path.exists(keyimage_associations_file_path), 'Keyimage association file path does not exist.'
            self.keyimage_associations = np.load(keyimage_associations_file_path, allow_pickle=True)
        else:
            keyimage_associations_file_path = os.path.join(self.association_folder_path, keyimage_associations_file_name)
            assert os.path.exists(keyimage_associations_file_path), 'Keyimage association file path does not exist.'
            self.keyimage_associations = np.load(keyimage_associations_file_path, allow_pickle=True)


        # Load pixel2point association files (.npy) and sort them
        if keyimage_yaml_name is None:
            # check if the folder exists
            self.associations_pixel2point_path = os.path.join(self.association_folder_path, 'pixel2point')
            assert os.path.exists(self.associations_pixel2point_path), 'Association pixel2point folder path does not exist.'
            self.associations_pixel2point_file_paths = [os.path.join(self.associations_pixel2point_path, f) for f in os.listdir(self.associations_pixel2point_path) if f.endswith('.npy')]
            self.associations_pixel2point_file_paths.sort()

            # Load point2pixel association files (.npy) and sort them
            # check if the folder exists
            self.associations_point2pixel_path = os.path.join(self.association_folder_path, 'point2pixel')
            assert os.path.exists(self.associations_point2pixel_path), 'Association point2pixel folder path does not exist.'
            self.associations_point2pixel_file_paths = [os.path.join(self.associations_point2pixel_path, f) for f in os.listdir(self.associations_point2pixel_path) if f.endswith('.npy')]
            self.associations_point2pixel_file_paths.sort()

            # Load segmentation files (.npy) and sort them
            # check if the folder exists
            assert os.path.exists(self.segmentation_folder_path), 'Segmentation folder path does not exist.'
            self.segmentation_file_paths = [os.path.join(self.segmentation_folder_path, f) for f in os.listdir(self.segmentation_folder_path) if f.endswith('.npy')]
            self.segmentation_file_paths.sort()   

        else:
            keyimage_yaml_path = os.path.join(self.association_folder_path, keyimage_yaml_name)
            assert os.path.exists(keyimage_yaml_path), 'Keyimage yaml path does not exist.'
            with open(keyimage_yaml_path, 'r') as file:
                keyimage_list = yaml.load(file, Loader=yaml.FullLoader)
            
            self.associations_pixel2point_path = os.path.join(self.association_folder_path, 'pixel2point')
            assert os.path.exists(self.associations_pixel2point_path), 'Association pixel2point folder path does not exist.'
            associations_pixel2point_files = [f for f in os.listdir(self.associations_pixel2point_path) if f.endswith('.npy')]
            self.associations_pixel2point_file_paths = [os.path.join(self.associations_pixel2point_path, f) for f in keyimage_list if f in associations_pixel2point_files]

            self.associations_point2pixel_path = os.path.join(self.association_folder_path, 'point2pixel')
            assert os.path.exists(self.associations_point2pixel_path), 'Association point2pixel folder path does not exist.'
            associations_point2pixel_files = [f for f in os.listdir(self.associations_point2pixel_path) if f.endswith('.npy')]
            self.associations_point2pixel_file_paths = [os.path.join(self.associations_point2pixel_path, f) for f in keyimage_list if f in associations_point2pixel_files]

            assert os.path.exists(self.segmentation_folder_path), 'Segmentation folder path does not exist.'
            segmentation_files = [f for f in os.listdir(self.segmentation_folder_path) if f.endswith('.npy')]
            self.segmentation_file_paths = [os.path.join(self.segmentation_folder_path, f) for f in keyimage_list if f in segmentation_files]

        # Check if the number of segmentation files and association files are the same
        assert len(self.segmentation_file_paths) == len(self.associations_pixel2point_file_paths), 'The number of segmentation files and pixel2point association files are not the same.'
        assert len(self.segmentation_file_paths) == len(self.associations_point2pixel_file_paths), 'The number of segmentation files and point2pixel association files are not the same.'

        # create segmentation-association pairs
        self.segmentation_association_pairs = []
        for i in range(len(self.segmentation_file_paths)):
            # check if the segmentation file name and association file names are the same
            segmentation_file_name = os.path.basename(self.segmentation_file_paths[i]).split('.')[0]
            associations_pixel2point_file_name = os.path.basename(self.associations_pixel2point_file_paths[i]).split('.')[0]
            associations_point2pixel_file_name = os.path.basename(self.associations_point2pixel_file_paths[i]).split('.')[0]
            assert segmentation_file_name == associations_pixel2point_file_name, 'The segmentation file name and association pixel2point file name are not the same.'
            assert segmentation_file_name == associations_point2pixel_file_name, 'The segmentation file name and association point2pixel file name are not the same.'
            self.segmentation_association_pairs.append((self.segmentation_file_paths[i], self.associations_pixel2point_file_paths[i], 
                                                        self.associations_point2pixel_file_paths[i]))

        # log the number of segmentation-association pairs
        if self.loginfo:
            self.logger.info('Number of segmentation-association pairs: {}'.format(len(self.segmentation_association_pairs)))

        # initialize data structures
        self.latest_registered_id = 0  # the latest registered object id
        self.associations_pixel2point = []
        self.associations_point2pixel = []
        self.segmented_objects_images = []

        # pre-compute gaussian weights
        self.radius = 2
        self.decaying = 2
        self.likelihoods = compute_gaussian_likelihood(radius=self.radius, decaying=self.decaying)

    def update_object_manager(self, object_id, intersected_points):
        if intersected_points is None:
            registered_objects_id = None
        else:  
            # if intersected_points is not None, get the object id with the maximum probability
            ids = self.pc_segmentation_ids[intersected_points, :]
            probs = self.pc_segmentation_probs[intersected_points, :]

            # Flatten the arrays and compute the sum of probabilities for each unique id
            unique_ids, indices = np.unique(ids, return_inverse=True)
            summed_probs = np.bincount(indices, weights=probs.flatten())

            # Find the id with the maximum probability sum
            max_prob_index = np.argmax(summed_probs)
            registered_objects_id = unique_ids[max_prob_index]

            # log registered_objects_id
            if self.loginfo:
                self.logger.info('    registered_objects_id: {}'.format(registered_objects_id))

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

    def update_pc_segmentation(self, associations1_point2pixel, segmented_objects_image1):
        registered_object_id_list = list(self.object_manager.values())
        group_registered_object_id_list = group_lists(registered_object_id_list)

        purge_object_id_map = dict()  # the key is the registered object id to be purged and the value is the registered object id to be kept
        for group_registered_object_id in group_registered_object_id_list:
            if len(group_registered_object_id) <= 1:
                pass
            else:
                for object_id in group_registered_object_id[1:]:
                    purge_object_id_map[object_id] = group_registered_object_id[0]

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
            
        for purge_object_id, keep_object_id in purge_object_id_map.items():
            purge = self.pc_segmentation_ids == purge_object_id
            keep = self.pc_segmentation_ids == keep_object_id
            purge_any = np.any(purge, axis=1)
            keep_any = np.any(keep, axis=1)
            purge_any_intersect_keep_any = np.logical_and(purge_any, keep_any)

            purge_exclude_keep_any = purge.copy()
            purge_exclude_keep_any[purge_any_intersect_keep_any] = False

            purge_intersect_keep_any = purge.copy()
            purge_intersect_keep_any[~purge_any_intersect_keep_any] = False

            keep_intersect_purge_any = keep.copy()
            keep_intersect_purge_any[~purge_any_intersect_keep_any] = False

            self.pc_segmentation_ids[purge_exclude_keep_any] = keep_object_id
            self.pc_segmentation_ids[purge_intersect_keep_any] = -1
            

            self.pc_segmentation_probs[keep_intersect_purge_any] += self.pc_segmentation_probs[purge_intersect_keep_any]
            
            self.pc_segmentation_probs[purge_intersect_keep_any] = 0


        # construct object_manager_array where the first column is the object id and the second column is the registered object id
        object_manager_array = -np.ones((len(self.object_manager), 2), dtype=np.int32)
        for i, (object_id, registered_object_ids) in enumerate(self.object_manager.items()):
            object_manager_array[i, 0] = object_id
            if registered_object_ids == []:
                object_manager_array[i, 1] = -1
            else:
                object_manager_array[i, 1] = registered_object_ids[0]

        numba_update_pc_segmentation(associations1_point2pixel, 
                                    segmented_objects_image1, 
                                    object_manager_array, 
                                    self.latest_registered_id, 
                                    self.M_segmentation_ids, 
                                    self.radius, 
                                    self.padded_segmentation, 
                                    self.normalized_likelihoods, 
                                    self.likelihoods, 
                                    self.pc_segmentation_ids, 
                                    self.pc_segmentation_probs)
        
        # test
        # make a copy of self.object_manager
        object_manager_copy = self.object_manager.copy()
        for i, (object_id, registered_object_ids) in enumerate(object_manager_copy.items()):
            if registered_object_ids == []:
                object_manager_copy[object_id] = [object_id + self.latest_registered_id]
            else:
                object_manager_copy[object_id] = registered_object_ids

        # log object_manager_copy
        if self.loginfo:
            self.logger.info('    object_manager_copy: {}'.format(object_manager_copy))

        # test end
                
        self.latest_registered_id += len(self.normalized_likelihoods)

    def save_semantics(self, save_semantics_path):
        max_prob_indices = np.argmax(self.pc_segmentation_probs, axis=1)
        semantics = self.pc_segmentation_ids[np.arange(len(max_prob_indices)), max_prob_indices]
        np.save(save_semantics_path, semantics)

    def save_semantic_pointcloud(self, save_las_path):
        max_prob_indices = np.argmax(self.pc_segmentation_probs, axis=1)
        semantics = self.pc_segmentation_ids[np.arange(len(max_prob_indices)), max_prob_indices]

        if self.pointcloud_path.endswith('.las'):
            points, colors = read_las_file(self.pointcloud_path)
        elif self.pointcloud_path.endswith('.npy'):
            points, colors = read_mesh_file(self.pointcloud_path)
            colors = colors * 255
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
        las.intensity = semantics

        # Write the LAS file
        las.write(save_las_path)


    def search_object2(self, key_image, pixel_object1_image2):
        """
        Within pixels of object1 in image2, search for object2 that has the largest number of semantics ids. 

        Parameters
        ----------
        key_image : int, the key image id
        pixel_object1_image2 : 2D array of shape (N_pixels, 2), where each row is a pixel coordinate (u, v)

        Returns
        -------
        max_count_id : int, the object id of object2
        pixel_object2_image2 : list of point indices
        """
        segmented_objects_image2 = self.segmented_objects_images[key_image]
        object_ids_object1_image2 = segmented_objects_image2[pixel_object1_image2[:, 0], pixel_object1_image2[:, 1]]
        #logging.info('    object_ids_object1_image2: {}'.format(object_ids_object1_image2))
        if len(object_ids_object1_image2) == 0:
            return -1, None
        unique_ids, counts = np.unique(object_ids_object1_image2, return_counts=True)
        max_count_id = unique_ids[np.argmax(counts)]
        pixel_object2_image2 = np.argwhere(segmented_objects_image2 == max_count_id)
        return max_count_id, pixel_object2_image2
    
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
        intersected_points = np.intersect1d(point_object1_image2, point_object2_image1, assume_unique=True)
        intersection = np.count_nonzero(intersected_points)
        union = len(point_object1_image2) + len(point_object2_image1) - intersection
        iou = intersection / union
        return iou, intersected_points

    def object_registration(self, iou_threshold=0.75, M_segmentation_ids=5, M_keyimages=5, save_semantics=False, save_semantic_las=False):
        """
        Register objects in the point cloud.

        Parameters
        ----------
        iou_threshold : float, the threshold for 3D IoU
        M_segmentation_ids : int, the maximum number of segmentation ids for each point
        M_keyimages : int, the maximum number of key images for each object
        save_semantics : bool, whether to save semantics for each image

        Returns
        -------
        None
        """
        N_images = len(self.segmentation_association_pairs)

        self.M_keyimages = M_keyimages
        self.M_segmentation_ids = M_segmentation_ids

        self.pc_segmentation_ids = -np.ones((self.N_points, self.M_segmentation_ids), dtype=np.int32)
        self.pc_segmentation_probs = np.zeros((self.N_points, self.M_segmentation_ids), dtype=np.float32)

        for image_id in range(N_images):
            # logging the current, total number images, and image name
            if self.loginfo:
                self.logger.info(f'Processing image {image_id+1}/{N_images}: {os.path.basename(self.segmentation_association_pairs[image_id][0])}')
                print(f'Processing image {image_id+1}/{N_images}: {os.path.basename(self.segmentation_association_pairs[image_id][0])}')    

            t1 =   time.time()
            # load segmentation and association files
            segmentation_file_path = self.segmentation_association_pairs[image_id][0]
            associations_pixel2point_file_path = self.segmentation_association_pairs[image_id][1]
            associations_point2pixel_file_path = self.segmentation_association_pairs[image_id][2]

            segmented_objects_image1 = np.load(segmentation_file_path, allow_pickle=True).astype(np.int16)
            associations1_pixel2point = np.load(associations_pixel2point_file_path, allow_pickle=True)
            associations1_point2pixel = np.load(associations_point2pixel_file_path, allow_pickle=True)

            self.segmented_objects_images.append(segmented_objects_image1)
            self.associations_pixel2point.append(associations1_pixel2point)
            self.associations_point2pixel.append(associations1_point2pixel)

            # create a segmentation mask for segmentation_objects_image1 with boolean data type, where True indicates the object id is not -1
            #segmentation_mask1 = segmented_objects_image1 != -1

            # pre-compute padded segmentation and normalized likelihoods
            image_height, image_width = segmented_objects_image1.shape
            self.padded_segmentation = -np.ones((2*self.radius+image_height+2, 2*self.radius+image_width+2)).astype(np.int16)
            self.padded_segmentation[self.radius+1:self.radius+image_height+1, self.radius+1:self.radius+image_width+1] = segmented_objects_image1
            self.normalized_likelihoods = np.zeros(int(segmented_objects_image1.max() + 1), dtype=np.float32)

            N_objects = int(segmented_objects_image1.max() + 1)

            self.object_manager = dict()  # the key is the object id and the value is a list of registered object ids.

            for object_id in range(N_objects):

                pixel_object1_image1_bool = segmented_objects_image1 == object_id
                
                point_object1_image1 = associations1_pixel2point[pixel_object1_image1_bool] # point_object1_image1 is a list of point ids
                # remove -1 from point_object1_image1
                point_object1_image1 = point_object1_image1[point_object1_image1 != -1]
                point_object1_image1_bool = np.zeros(self.N_points, dtype=bool)
                point_object1_image1_bool[point_object1_image1] = True

                # get the keyimages of object_id
                if image_id == 0:
                    keyimages = []
                else:
                    keyimages = self.keyimage_associations[point_object1_image1_bool, :image_id]
                    keyimages = np.sum(keyimages, axis=0) 
                    descending_indices = np.argsort(keyimages)[::-1]
                    nonzero_indices = np.argwhere(keyimages > 0).reshape(-1)
                    if len(nonzero_indices) > self.M_keyimages:
                        descending_indices = descending_indices[:self.M_keyimages]
                    else:
                        descending_indices = nonzero_indices
                    
                    keyimages = descending_indices.tolist()

                # process keyimages
                if len(keyimages) == 0:
                    self.update_object_manager(object_id, None)
                else:
                    # iterate over all key images
                    for key_image in keyimages:
                        associations2_pixel2point = self.associations_pixel2point[key_image]
                        associations2_point2pixel = self.associations_point2pixel[key_image]

                        pixel_object1_image2 = associations2_point2pixel[point_object1_image1]
                        pixel_object1_image2 = pixel_object1_image2[pixel_object1_image2[:, 0] != -1]
                        point_object1_image2 = associations2_pixel2point[pixel_object1_image2[:, 0], pixel_object1_image2[:, 1]]  # point_object1_image2 is a list of point ids

                        object2_id_image2, pixel_object2_image2 = self.search_object2(key_image, pixel_object1_image2)

                        if object2_id_image2 == -1:
                            self.update_object_manager(object_id, None)
                        else:
                            point_object2_image2 = associations2_pixel2point[pixel_object2_image2[:, 0], pixel_object2_image2[:, 1]]  # point_object2_image2 is a list of point ids
                            point_object2_image2 = point_object2_image2[point_object2_image2 != -1]
                            pixel_object2_image1 = associations1_point2pixel[point_object2_image2]
                            pixel_object2_image1 = pixel_object2_image1[pixel_object2_image1[:, 0] != -1]
                            point_object2_image1 = associations1_pixel2point[pixel_object2_image1[:, 0], pixel_object2_image1[:, 1]]

                            iou, intersected_points = self.calculate_3D_IoU(point_object1_image2, point_object2_image1)

                            if self.loginfo:
                                self.logger.info("    object_id: {}, key_image: {}, object2_id_image2: {}, iou: {}".format(object_id, key_image, object2_id_image2, iou))

                            if iou >= iou_threshold:
                                self.update_object_manager(object_id, intersected_points)
                            else:
                                self.update_object_manager(object_id, None)
            
            t3 = time.time()
            if self.loginfo:
                self.logger.info("    time elapsed for updating object_manager {}: {}".format(image_id+1, t3-t1))
                #print("time elapsed for updating object_manager {}: {}".format(image_id+1, t3-t1))

                # logging self.object_manager
                self.logger.info('    object_manager: {}'.format(self.object_manager))

            self.update_pc_segmentation(associations1_point2pixel, segmented_objects_image1)
            t4 = time.time()
            
            if self.loginfo:
                self.logger.info("    time elapsed for updating pc_segmentation {}: {}".format(image_id+1, t4-t3))
                #print("time elapsed for updating pc_segmentation {}: {}".format(image_id+1, t4-t3))

            t2 = time.time()
            if self.loginfo:
                self.logger.info("    time elapsed for image {}: {}".format(image_id+1, t2-t1))
                #print("time elapsed for image {}: {}".format(image_id+1, t2-t1))
        
            if save_semantics:
                # create a folder to save semantics under self.association_folder_path
                semantics_folder_path = os.path.join(self.association_folder_path, 'semantics')
                if not os.path.exists(semantics_folder_path):
                    os.makedirs(semantics_folder_path)
                save_semantics_path = os.path.join(semantics_folder_path, 'semantics_{}.npy'.format(image_id))
                self.save_semantics(save_semantics_path)

                if save_semantic_las:
                    save_las_path = os.path.join(self.association_folder_path, 'semantics', 'semantics_{}.las'.format(image_id))
                    add_semantics_to_pointcloud(self.pointcloud_path, save_semantics_path, save_las_path)   
                    

if __name__ == "__main__":
    # Set paths
    pointcloud_path = '../../data/box_canyon_park/SfM_products/agisoft_model.las'
    segmentation_folder_path = '../../data/box_canyon_park/segmentations'
    image_folder_path = '../../data/box_canyon_park/DJI_photos'
    association_folder_path = '../../data/box_canyon_park/associations'

    object_registration_flag = True
    add_semantics_to_pointcloud_flag = False
    if object_registration_flag:
        # Create object registration
        t1 = time.time()
        obr = ObjectRegistration(pointcloud_path, segmentation_folder_path, association_folder_path)
        t2 = time.time()
        print('Time elapsed for creating object registration: {}'.format(t2-t1))

        # Run object registration
        obr.object_registration(iou_threshold=0.5, save_semantics=True)
        #obr.object_registration(iou_threshold=0.5)

    if add_semantics_to_pointcloud_flag:
        # Add semantics to the point cloud
        image_id = 50
        semantics_folder_path = os.path.join(association_folder_path, 'semantics', 'semantics_{}.npy'.format(image_id))
        save_las_path = os.path.join(association_folder_path, 'semantics', 'semantics_{}.las'.format(image_id))
        add_semantics_to_pointcloud(pointcloud_path, semantics_folder_path, save_las_path)   

