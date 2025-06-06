from ssfm.files import *
from ssfm.image_segmentation import *
from ssfm.probabilistic_projection import *

import os
import time
import numpy as np
from collections import defaultdict
import logging
import yaml
from collections import Counter
from tqdm import tqdm
import networkx as nx

from joblib import Parallel, delayed
from numba import njit, prange, jit
import uuid

segmented_objects_images_memmap = None
associations_pixel2point_memmap = None
associations_point2pixel_memmap = None
keyimage_associations_memmap = None


@njit
def apply_remap_and_merge(ids, probs, id_map, purge_keys, keep_keys):
    """
    Apply remapping and merge overlapping probabilities.
    Parameters
    ----------
    ids : 2D array of shape (N, M), where N is the number of points and M is the number of segmentation ids
    probs : 2D array of shape (N, M), where N is the number of points and M is the number of segmentation ids
    id_map : 1D array of shape (max_id + 1,), where max_id is the maximum id in ids
    purge_keys : 1D array of shape (K,), where K is the number of keys to purge
    keep_keys : 1D array of shape (K,), where K is the number of keys to keep
    Returns
    -------
    ids : 2D array of shape (N, M), where N is the number of points and M is the number of segmentation ids
    probs : 2D array of shape (N, M), where N is the number of points and M is the number of segmentation ids
    """
    N, M = ids.shape

    # Step 1: Apply remapping
    for i in range(N):
        for j in range(M):
            val = ids[i, j]
            if val != -1:
                ids[i, j] = id_map[val]

    # Step 2: Merge overlapping probabilities
    for i in range(N):
        for k in range(len(purge_keys)):
            purge_id = purge_keys[k]
            keep_id = keep_keys[k]
            purge_idx = -1
            keep_idx = -1
            for j in range(M):
                if ids[i, j] == purge_id:
                    purge_idx = j
                elif ids[i, j] == keep_id:
                    keep_idx = j
            if purge_idx != -1 and keep_idx != -1:
                probs[i, keep_idx] += probs[i, purge_idx]
                ids[i, purge_idx] = -1
                probs[i, purge_idx] = 0
    return ids, probs


def convert_to_numpy_memmap(segmented_objects_images, associations_pixel2point, associations_point2pixel, keyimage_associations):
    uid = uuid.uuid4().hex
    # segmented_objects_images
    N = len(segmented_objects_images)
    H, W = segmented_objects_images[0].shape
    dtype = segmented_objects_images[0].dtype
    segmented_objects_images_path = os.path.join("/dev/shm", "segmented_objects_images_{}.dat".format(uid))
    segmented_objects_images_memmap = np.memmap(segmented_objects_images_path, dtype=dtype, mode='w+', shape=(N, H, W))
    for i in range(N):
        segmented_objects_images_memmap[i] = segmented_objects_images[i]

    # associations_pixel2point
    N = len(associations_pixel2point)
    H, W = associations_pixel2point[0].shape
    dtype = associations_pixel2point[0].dtype
    associations_pixel2point_path = os.path.join("/dev/shm", "associations_pixel2point_{}.dat".format(uid))
    associations_pixel2point_memmap = np.memmap(associations_pixel2point_path, dtype=dtype, mode='w+', shape=(N, H, W))
    for i in range(N):
        associations_pixel2point_memmap[i] = associations_pixel2point[i]

    # associations_point2pixel
    N = len(associations_point2pixel)
    H, W = associations_point2pixel[0].shape
    dtype = associations_point2pixel[0].dtype
    associations_point2pixel_path = os.path.join("/dev/shm", "associations_point2pixel_{}.dat".format(uid))
    associations_point2pixel_memmap = np.memmap(associations_point2pixel_path, dtype=dtype, mode='w+', shape=(N, H, W))
    for i in range(N):
        associations_point2pixel_memmap[i] = associations_point2pixel[i]
    
    # keyimage_associations, note that keyimage_associations is a 2D array
    N = keyimage_associations.shape[0]
    M = keyimage_associations.shape[1]
    dtype = keyimage_associations.dtype
    keyimage_associations_path = os.path.join("/dev/shm", "keyimage_associations_{}.dat".format(uid))
    keyimage_associations_memmap = np.memmap(keyimage_associations_path, dtype=dtype, mode='w+', shape=(N, M))
    keyimage_associations_memmap[:] = keyimage_associations[:]

    # flush the memmap files to disk
    segmented_objects_images_memmap.flush()
    associations_pixel2point_memmap.flush()
    associations_point2pixel_memmap.flush()
    keyimage_associations_memmap.flush()
    # close the memmap files
    del segmented_objects_images_memmap
    del associations_pixel2point_memmap
    del associations_point2pixel_memmap
    del keyimage_associations_memmap

    # return the paths of the memmap files
    return segmented_objects_images_path, associations_pixel2point_path, associations_point2pixel_path, keyimage_associations_path


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

    # Build adjacency list
    adjacency_list = defaultdict(set)
    for indices in element_to_list_map.values():
        for list_index in indices:
            adjacency_list[list_index].update(i for i in indices if i != list_index)

    visited = [False] * len(lists)
    groups = []

    # Iterative DFS
    for list_index in range(len(lists)):
        if not visited[list_index]:
            stack = [list_index]
            group = []
            visited[list_index] = True

            while stack:
                current = stack.pop()
                group.append(current)
                for neighbor in adjacency_list[current]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)

            groups.append(group)

    # Combine lists based on groups
    grouped_lists = []
    for group in groups:
        merged = set()
        for idx in group:
            merged.update(lists[idx])
        grouped_lists.append(sorted(merged))

    return grouped_lists


@jit(nopython=True)
def numba_update_pc_segmentation(sequence_id, object_manager_array, latest_registered_id, M_segmentation_ids, radius, padded_segmentation, normalized_likelihoods, likelihoods, pc_segmentation_ids, pc_segmentation_probs):
    global segmented_objects_images_memmap
    global associations_point2pixel_memmap

    associations1_point2pixel = associations_point2pixel_memmap[sequence_id]
    segmented_objects_image1 = segmented_objects_images_memmap[sequence_id]
    
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

            # fill the new_likelihoods_array with object ids and normalized object probabilities
            for i in range(len(normalized_likelihoods)):
                if normalized_likelihoods[i] > 0.00001:
                    new_likelihoods_array[i, 0] = object_manager_array[i, 1]
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

    return pc_segmentation_ids, pc_segmentation_probs


class SegMo3D(object):
    def __init__(self, pointcloud_path, segmentation_folder_path, association_folder_path, keyimage_associations_file_name=None, image_list=None, loginfo=True, using_graph=False, radius=2, decaying=1, scene_name="scene"):
        """
        Parameters
        ----------
        pointcloud_path : str, the path to the point cloud file
        segmentation_folder_path : str, the path to the segmentation folder
        association_folder_path : str, the path to the association folder
        keyimage_associations_file_name : str, the name of the keyimage associations file
        image_list : list of str, the list of image file names
        loginfo : bool, whether to log information
        using_graph : bool, whether to use graph
        radius : int, the radius for gaussian likelihood
        decaying : int, the decaying factor for gaussian likelihood
        scene_name : str, the name of the scene
        """
        global keyimage_associations
        
        self.pointcloud_path = pointcloud_path
        self.segmentation_folder_path = segmentation_folder_path
        self.association_folder_path = association_folder_path
        self.loginfo = loginfo

        if self.loginfo:
            logging.basicConfig(filename=scene_name + '_segmo3d.log', 
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
            keyimage_associations = np.load(keyimage_associations_file_path, allow_pickle=True)
        else:
            keyimage_associations_file_path = os.path.join(self.association_folder_path, keyimage_associations_file_name)
            assert os.path.exists(keyimage_associations_file_path), 'Keyimage association file path does not exist.'
            keyimage_associations = np.load(keyimage_associations_file_path, allow_pickle=True)


        # Load pixel2point association files (.npy) and sort them
        if image_list is None:
            # check if the folder exists
            self.associations_pixel2point_path = os.path.join(self.association_folder_path, 'pixel2point')
            assert os.path.exists(self.associations_pixel2point_path), 'Association pixel2point folder path does not exist.'
            self.associations_pixel2point_file_paths = [os.path.join(self.associations_pixel2point_path, f) for f in os.listdir(self.associations_pixel2point_path) if f.endswith('.npy')]
            # sort the association files based on the number in the file name
            self.associations_pixel2point_file_paths = sorted(self.associations_pixel2point_file_paths, key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))

            # Load point2pixel association files (.npy) and sort them
            # check if the folder exists
            self.associations_point2pixel_path = os.path.join(self.association_folder_path, 'point2pixel')
            assert os.path.exists(self.associations_point2pixel_path), 'Association point2pixel folder path does not exist.'
            self.associations_point2pixel_file_paths = [os.path.join(self.associations_point2pixel_path, f) for f in os.listdir(self.associations_point2pixel_path) if f.endswith('.npy')]
            # sort the association files based on the number in the file name
            self.associations_point2pixel_file_paths = sorted(self.associations_point2pixel_file_paths, key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))

            # Load segmentation files (.npy) and sort them
            # check if the folder exists
            assert os.path.exists(self.segmentation_folder_path), 'Segmentation folder path does not exist.'
            self.segmentation_file_paths = [os.path.join(self.segmentation_folder_path, f) for f in os.listdir(self.segmentation_folder_path) if f.endswith('.npy')]
            # sort the segmentation files based on the number in the file name
            self.segmentation_file_paths = sorted(self.segmentation_file_paths, key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))

        else: 
            keyimage_list = [f.replace('.jpg', '.npy').replace('.png', '.npy').replace('.JPG', '.npy') for f in image_list]
                       
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

        # load graph 
        self.using_graph = using_graph
        if using_graph:
            graph_file_path = os.path.join(self.association_folder_path, 'graph.graphml')
            assert os.path.exists(graph_file_path), 'Graph file path does not exist.'
            self.graph = nx.read_graphml(graph_file_path)
            # convert edges to dictionary
            self.edges = dict()
            for edge in self.graph.edges:
                i = int(edge[0])
                j = int(edge[1])
                if i not in self.edges.keys():
                    self.edges[i] = [j]
                else:
                    self.edges[i].append(j)
                if j not in self.edges.keys():
                    self.edges[j] = [i]
                else:
                    self.edges[j].append(i)

            # check if key is in value list and sort the values
            for key, values in self.edges.items():
                if key in values:
                    raise ValueError('key is in values.')
                
                self.edges[key] = sorted(values)

        # log the number of segmentation-association pairs
        if self.loginfo:
            self.logger.info('Number of segmentation-association pairs: {}'.format(len(self.segmentation_association_pairs)))

        # initialize data structures
        self.latest_registered_id = 0  # the latest registered object id

        # pre-compute gaussian weights
        self.radius = radius
        self.decaying = decaying
        self.likelihoods = compute_gaussian_likelihood(radius=self.radius, decaying=self.decaying)

        self.registered_object_manager = dict()  # the key is the image id and object id; the value is the registered object id
        self.image_id = 0
        self.sequence_id = 0

    
    def update_object_manager(self, object_id, key_image_id, object2_id_image2, intersected_points):
        if key_image_id is None:
            registered_objects_id = None
        elif key_image_id == -1:
            registered_objects_id = -1
        else:
            registered_objects_id = self.registered_object_manager[(key_image_id, object2_id_image2)]

        if object_id not in self.object_manager.keys():
            if registered_objects_id == None:
                self.object_manager[object_id] = []
            elif registered_objects_id == -1:
                self.object_manager[object_id] = [-1]
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

        

    def update_pc_segmentation(self):
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


        # update registered_object_manager
        for object_id, registered_object_ids in self.object_manager.items():
            # check the length of values in self.object_manager
            if len(registered_object_ids) > 1:
                raise ValueError('The length of registered_object_ids is not 1.')
            else:
                if len(registered_object_ids) == 0:
                    registered_object_id = object_id + self.latest_registered_id
                else:
                    registered_object_id = registered_object_ids[0]
                self.registered_object_manager[(self.image_id, object_id)] = registered_object_id


        max_seg_id = np.max(self.pc_segmentation_ids)
        max_purge_id = max(purge_object_id_map.keys(), default=max_seg_id)
        id_map_size = max(max_seg_id, max_purge_id) + 2  # Buffer
        id_map = np.arange(id_map_size, dtype=np.int32)   # Default sequential mapping
        for purge_id, keep_id in purge_object_id_map.items():
            id_map[purge_id] = keep_id  # IndexError: index 752138 is out of bounds for axis 0 with size 752128

        # Prepare key arrays
        purge_keys = np.array(list(purge_object_id_map.keys()), dtype=np.int32)
        keep_keys = np.array(list(purge_object_id_map.values()), dtype=np.int32)

        # Apply optimized remapping & merging
        self.pc_segmentation_ids, self.pc_segmentation_probs = apply_remap_and_merge(
            self.pc_segmentation_ids, self.pc_segmentation_probs, id_map, purge_keys, keep_keys
        )


        # construct object_manager_array where the first column is the object id and the second column is the registered object id
        object_manager_array = -np.ones((len(self.object_manager), 2), dtype=np.int32)
        for i, (object_id, registered_object_ids) in enumerate(self.object_manager.items()):
            object_manager_array[i, 0] = object_id
            if registered_object_ids == []:
                object_manager_array[i, 1] = object_id + self.latest_registered_id
            else:
                object_manager_array[i, 1] = registered_object_ids[0]

        self.pc_segmentation_ids, self.pc_segmentation_probs = numba_update_pc_segmentation(
                                    self.sequence_id,
                                    object_manager_array, 
                                    self.latest_registered_id, 
                                    self.M_segmentation_ids, 
                                    self.radius, 
                                    self.padded_segmentation, 
                                    self.normalized_likelihoods, 
                                    self.likelihoods,
                                    self.pc_segmentation_ids,
                                    self.pc_segmentation_probs)


        self.latest_registered_id += len(self.normalized_likelihoods)

    def save_prob_semantics(self):
        pc_segmentation_save_path = os.path.join(self.association_folder_path, 'semantics', str(self.image_id) + '_segmentation_ids.npy')
        np.save(pc_segmentation_save_path, self.pc_segmentation_ids)
        pc_segmentation_probs_save_path = os.path.join(self.association_folder_path, 'semantics', str(self.image_id) + '_segmentation_probs.npy')
        np.save(pc_segmentation_probs_save_path, self.pc_segmentation_probs)

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


    def search_object2(self, key_image_id, pixel_object1_image2):
        """
        Within pixels of object1 in image2, search for object2 that has the largest number of semantics ids. 

        Parameters
        ----------
        key_image_id : int, the key image id
        pixel_object1_image2 : 2D array of shape (N_pixels, 2), where each row is a pixel coordinate (u, v)

        Returns
        -------
        max_count_id : int, the object id of object2
        pixel_object2_image2 : list of point indices
        """
        global segmented_objects_images_memmap
        key_sequence_id = np.where(self.keyimage_idx_list == key_image_id)[0][0]
        segmented_objects_image2 = segmented_objects_images_memmap[key_sequence_id]
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

    def segmo3d(self, iou_threshold=0.75, M_segmentation_ids=5, M_keyimages=5, save_semantics=False, save_semantic_las=False, save_semantics_all=False, explicit_background=False):
        """
        Register objects in the point cloud.

        Parameters
        ----------
        iou_threshold : float, the threshold for 3D IoU
        M_segmentation_ids : int, the maximum number of segmentation ids for each point
        M_keyimages : int, the maximum number of key images for each object
        save_semantics : bool, whether to save semantics for each image
        save_semantic_las : bool, whether to save semantic point cloud
        save_semantics_all : bool, whether to save all semantics ids and probabilities
        explicit_background : bool, whether to use explicit background. When True, the background has to be set to 0 in the segmentation file before running the code.

        Returns
        -------
        None
        """

        global segmented_objects_images_memmap
        global associations_pixel2point_memmap
        global associations_point2pixel_memmap
        global keyimage_associations_memmap


        N_images = len(self.segmentation_association_pairs)

        self.M_segmentation_ids = M_segmentation_ids

        self.pc_segmentation_ids = -np.ones((self.N_points, self.M_segmentation_ids), dtype=np.int32)
        self.pc_segmentation_probs = np.zeros((self.N_points, self.M_segmentation_ids), dtype=np.float32)

        self.keyimage_idx_list = np.where(np.sum(keyimage_associations, axis=0) > 0)[0]

        self.explicit_background = explicit_background

        segmented_objects_images = []
        associations_pixel2point = []
        associations_point2pixel = []


        for image_id in self.keyimage_idx_list:
            segmentation_file_path = self.segmentation_association_pairs[image_id][0]
            associations_pixel2point_file_path = self.segmentation_association_pairs[image_id][1]
            associations_point2pixel_file_path = self.segmentation_association_pairs[image_id][2]

            segmented_objects_image1 = np.load(segmentation_file_path, allow_pickle=True).astype(np.int16)
            associations1_pixel2point = np.load(associations_pixel2point_file_path, allow_pickle=True)
            associations1_point2pixel = np.load(associations_point2pixel_file_path, allow_pickle=True)

            segmented_objects_images.append(segmented_objects_image1)
            associations_pixel2point.append(associations1_pixel2point)
            associations_point2pixel.append(associations1_point2pixel)

        print(f"Memory size of segmented_objects_images: {sum(arr.nbytes for arr in segmented_objects_images) / 1024**3:.3f} GB")
        print(f"Memory size of associations_pixel2point: {sum(arr.nbytes for arr in associations_pixel2point) / 1024**3:.3f} GB")
        print(f"Memory size of associations_point2pixel: {sum(arr.nbytes for arr in associations_point2pixel) / 1024**3:.3f} GB")
        print(f"Memory size of pc_segmentation_ids: {self.pc_segmentation_ids.nbytes / 1024**3:.3f} GB")
        print(f"Memory size of pc_segmentation_probs: {self.pc_segmentation_probs.nbytes / 1024**3:.3f} GB")

        segmented_objects_images_path, associations_pixel2point_path, associations_point2pixel_path, keyimage_associations_path = convert_to_numpy_memmap(
            segmented_objects_images, 
            associations_pixel2point, 
            associations_point2pixel, 
            keyimage_associations)

        # load segmented_objects_images, associations_pixel2point, associations_point2pixel, keyimage_associations
        segmented_objects_images_memmap = np.memmap(segmented_objects_images_path, dtype=segmented_objects_images[0].dtype, mode='r', shape=(len(segmented_objects_images), segmented_objects_images[0].shape[0], segmented_objects_images[0].shape[1]))
        associations_pixel2point_memmap = np.memmap(associations_pixel2point_path, dtype=associations_pixel2point[0].dtype, mode='r', shape=(len(associations_pixel2point), associations_pixel2point[0].shape[0], associations_pixel2point[0].shape[1]))
        associations_point2pixel_memmap = np.memmap(associations_point2pixel_path, dtype=associations_point2pixel[0].dtype, mode='r', shape=(len(associations_point2pixel), associations_point2pixel[0].shape[0], associations_point2pixel[0].shape[1]))
        keyimage_associations_memmap = np.memmap(keyimage_associations_path, dtype=keyimage_associations.dtype, mode='r', shape=(keyimage_associations.shape[0], keyimage_associations.shape[1]))

        
        for sequence_id, image_id in tqdm(enumerate(self.keyimage_idx_list), total=len(self.keyimage_idx_list), desc="Processing images"):
            self.sequence_id = sequence_id
            self.image_id = image_id
            # logging the current, total number images, and image name
            if self.loginfo:
                self.logger.info(f'Processing image {image_id+1}/{N_images}: {os.path.basename(self.segmentation_association_pairs[image_id][0])}')
                #print(f'Processing image {image_id+1}/{N_images}: {os.path.basename(self.segmentation_association_pairs[image_id][0])}')    

            t1 = time.time()

            segmented_objects_image1 = segmented_objects_images_memmap[self.sequence_id]

            image_height, image_width = segmented_objects_image1.shape
            self.padded_segmentation = -np.ones((2*self.radius+image_height+2, 2*self.radius+image_width+2)).astype(np.int16)
            self.padded_segmentation[self.radius+1:self.radius+image_height+1, self.radius+1:self.radius+image_width+1] = segmented_objects_image1
            self.normalized_likelihoods = np.zeros(int(segmented_objects_image1.max() + 1), dtype=np.float32)

            N_objects = int(segmented_objects_image1.max() + 1)

            self.object_manager = dict()  # the key is the object id and the value is a list of registered object ids.
            
            # parallel processing using joblib
            results_list = Parallel(n_jobs=8, prefer='threads')(delayed(self.process_single_object)(
                object_id,
                self.N_points,
                M_keyimages,
                iou_threshold
                ) for object_id in range(N_objects))

            t2 = time.time()
            if self.loginfo:
                self.logger.info("    time elapsed for processing all objects: {}".format(t2-t1))


            for results in results_list:
                for result in results:
                    self.update_object_manager(result[0], result[1], result[2], result[3])

            t3 = time.time()
            if self.loginfo:
                self.logger.info("    time elapsed for updating object_manager {}: {}".format(image_id+1, t3-t2))
                #print("time elapsed for updating object_manager {}: {}".format(image_id+1, t3-t1))
                

                # print registered_object_manager
                #print('    registered_object_manager: {}'.format(self.registered_object_manager))

                #print('    object_manager: {}'.format(self.object_manager))

                # logging self.object_manager
                self.logger.info('    object_manager: {}'.format(self.object_manager))

            self.update_pc_segmentation()
            t4 = time.time()
            
            if self.loginfo:
                self.logger.info("    time elapsed for updating pc_segmentation {}: {}".format(image_id+1, t4-t3))
                #print("time elapsed for updating pc_segmentation {}: {}".format(image_id+1, t4-t3))

            t5 = time.time()
            if self.loginfo:
                self.logger.info("    time elapsed for image {}: {}".format(image_id+1, t5-t1))
                #print("time elapsed for image {}: {}".format(image_id+1, t2-t1))
        
            if save_semantics:
                # create a folder to save semantics under self.association_folder_path
                semantics_folder_path = os.path.join(self.association_folder_path, 'semantics')
                if not os.path.exists(semantics_folder_path):
                    os.makedirs(semantics_folder_path)
                save_semantics_path = os.path.join(semantics_folder_path, 'semantics_{}.npy'.format(image_id))
                self.save_semantics(save_semantics_path)

                if save_semantic_las:
                    save_las_path = os.path.join(self.association_folder_path, 'semantics', 'semantics_{}.las'.format(image_id+1))
                    add_semantics_to_pointcloud(self.pointcloud_path, save_semantics_path, save_las_path) 

            if save_semantics_all:
                # create a folder to save semantics under self.association_folder_path
                semantics_folder_path = os.path.join(self.association_folder_path, 'semantics')
                if not os.path.exists(semantics_folder_path):
                    os.makedirs(semantics_folder_path)
                self.save_prob_semantics()

        # delete the memmap files
        os.remove(segmented_objects_images_path)
        os.remove(associations_pixel2point_path)
        os.remove(associations_point2pixel_path)
        os.remove(keyimage_associations_path)  
           
    def process_single_object(
        self,
        object_id, 
        N_points,
        M_keyimages,
        iou_threshold,
        edges=None):

        global keyimage_associations_memmap
        global associations_pixel2point_memmap
        global associations_point2pixel_memmap
        global segmented_objects_images_memmap

        segmented_objects_image1 = segmented_objects_images_memmap[self.sequence_id]
        associations1_pixel2point = associations_pixel2point_memmap[self.sequence_id]
        associations1_point2pixel = associations_point2pixel_memmap[self.sequence_id]

        pixel_object1_image1_bool = segmented_objects_image1 == object_id
        point_object1_image1 = associations1_pixel2point[pixel_object1_image1_bool] # point_object1_image1 is a list of point ids
        point_object1_image1 = point_object1_image1[point_object1_image1 != -1]
        point_object1_image1_bool = np.zeros(N_points, dtype=bool)
        point_object1_image1_bool[point_object1_image1] = True

        object_updates = []

        # get the keyimages of object_id
        if self.sequence_id == 0:
            keyimages = []
        else:
            if edges is not None:
                pass
            else:
                keyimages = keyimage_associations_memmap[point_object1_image1_bool, :self.image_id]
                keyimages = np.sum(keyimages, axis=0) 
                descending_indices = np.argsort(keyimages)[::-1]
                nonzero_indices = np.argwhere(keyimages > 0).reshape(-1)
                if len(nonzero_indices) > M_keyimages:
                    descending_indices = descending_indices[:M_keyimages]
                else:
                    descending_indices = nonzero_indices
                
                keyimages = descending_indices.tolist()

        if len(keyimages) == 0:
            #self.update_object_manager(object_id, None)
            object_updates.append((object_id, None, None, None))

        else:
            # iterate over all key images
            for key_image_id in keyimages:
                # find sequence_id from the index of self.keyimage_idx_list
                key_sequence_id = np.where(self.keyimage_idx_list == key_image_id)[0][0]
                associations2_pixel2point = associations_pixel2point_memmap[key_sequence_id]
                associations2_point2pixel = associations_point2pixel_memmap[key_sequence_id]

                pixel_object1_image2 = associations2_point2pixel[point_object1_image1]
                pixel_object1_image2 = pixel_object1_image2[pixel_object1_image2[:, 0] != -1]
                point_object1_image2 = associations2_pixel2point[pixel_object1_image2[:, 0], pixel_object1_image2[:, 1]]  # point_object1_image2 is a list of point ids

                object2_id_image2, pixel_object2_image2 = self.search_object2(key_image_id, pixel_object1_image2)

                if object2_id_image2 == -1:
                    #self.update_object_manager(object_id, None)
                    object_updates.append((object_id, -1, None, None))
                else:
                    if self.explicit_background:
                        if object2_id_image2 == 0:
                            if object_id != 0:
                                object_updates.append((object_id, None, None, None))
                                continue
                    point_object2_image2 = associations2_pixel2point[pixel_object2_image2[:, 0], pixel_object2_image2[:, 1]]  # point_object2_image2 is a list of point ids
                    point_object2_image2 = point_object2_image2[point_object2_image2 != -1]
                    pixel_object2_image1 = associations1_point2pixel[point_object2_image2]
                    pixel_object2_image1 = pixel_object2_image1[pixel_object2_image1[:, 0] != -1]
                    point_object2_image1 = associations1_pixel2point[pixel_object2_image1[:, 0], pixel_object2_image1[:, 1]]

                    iou, intersected_points = self.calculate_3D_IoU(point_object1_image2, point_object2_image1)

                    if self.loginfo:
                        self.logger.info("    object_id: {}, key_image_id: {}, object2_id_image2: {}, iou: {}".format(object_id, key_image_id, object2_id_image2, iou))

                    if iou >= iou_threshold:
                        #self.update_object_manager2(object_id, key_image_id, object2_id_image2, intersected_points)
                        object_updates.append((object_id, key_image_id, object2_id_image2, intersected_points))
                    else:
                        #self.update_object_manager(object_id, None)
                        object_updates.append((object_id, None, None, None))
        
        return object_updates


                    

if __name__ == "__main__":
    # Set paths
    scene_dir = '../../data/kubric_0'
    pointcloud_path = os.path.join(scene_dir, 'reconstructions', 'combined_point_cloud.las')
    associations_folder_path = os.path.join(scene_dir, 'associations')
    segmentations_folder_path = os.path.join(scene_dir, 'segmentations')
    photos_folder_path = os.path.join(scene_dir, 'photos')

    segmo3d_option = True
    save_semantics = True
    scene_name = "kubric_0_segmo3d_8"
    postprocessing_option = True


    image_list = [f for f in os.listdir(photos_folder_path) if f.endswith('.png')]
    image_list.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

    keyimage_associations_file_name = 'associations_keyimage.npy'

    if segmo3d_option:
        # Create object registration
        obr = SegMo3D(pointcloud_path, segmentations_folder_path, associations_folder_path, keyimage_associations_file_name=keyimage_associations_file_name, image_list=image_list, scene_name=scene_name)
        # Run object registration
        obr.segmo3d(iou_threshold=0.5, save_semantics=save_semantics)


    if postprocessing_option:
        from ssfm.post_processing import *
        image_id = 284
        semantics_folder_path = os.path.join(associations_folder_path, 'semantics', 'semantics_{}.npy'.format(image_id))
        save_las_path = os.path.join(associations_folder_path, 'semantics', 'semantics_{}.las'.format(image_id))
        add_semantics_to_pointcloud(pointcloud_path, semantics_folder_path, save_las_path, remove_small_N=500, nearest_interpolation=500)

