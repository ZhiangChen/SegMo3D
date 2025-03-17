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
from scipy.spatial import cKDTree

from util import *

def intersect_with_indices(a: np.ndarray, b: np.ndarray):
    """
    Find values that appear in both 'a' and 'b' (1D), plus the indices where 
    those values appear in 'a' and 'b'.

    Args:
        a, b: 1D NumPy arrays.
    Returns:
        intersect_vals: 1D NumPy array of the matched values (sorted in ascending order).
        idx_a:          1D NumPy array of indices in 'a' corresponding to each matched value.
        idx_b:          1D NumPy array of indices in 'b' corresponding to each matched value.

    Notes:
    - Duplicates are preserved. If a value appears multiple times in both arrays, 
      you may see multiple matches. 
    - If you want unique intersection (set-style), you can call np.unique on 'a' and 'b'
      before passing them in. But then you lose the original index mapping for duplicates.
    """

    # 1) Sort each array, retaining original indices
    idx_a_sorted = np.argsort(a)
    idx_b_sorted = np.argsort(b)
    a_sorted = a[idx_a_sorted]
    b_sorted = b[idx_b_sorted]

    # 2) Two-pointer approach to find common elements
    i, j = 0, 0
    out_vals = []
    out_idx_a = []
    out_idx_b = []
    while i < len(a_sorted) and j < len(b_sorted):
        if a_sorted[i] < b_sorted[j]:
            i += 1
        elif a_sorted[i] > b_sorted[j]:
            j += 1
        else:
            # Found a match
            out_vals.append(a_sorted[i])
            out_idx_a.append(idx_a_sorted[i])
            out_idx_b.append(idx_b_sorted[j])
            i += 1
            j += 1

    # 3) Convert to NumPy arrays
    if len(out_vals) == 0:
        return (np.array([], dtype=a.dtype),
                np.array([], dtype=int),
                np.array([], dtype=int))
    intersect_vals = np.array(out_vals, dtype=a.dtype)
    idx_a_final = np.array(out_idx_a, dtype=int)
    idx_b_final = np.array(out_idx_b, dtype=int)

    return intersect_vals, idx_a_final, idx_b_final

def get_matching_indices_modified(pc_dict_0, pc_dict_1):
    pc_index_0 = pc_dict_0['index']
    pc_index_1 = pc_dict_1['index']
    # find the intersection of the two indices
    intersect_vals, idx_0, idx_1 = intersect_with_indices(pc_index_0, pc_index_1)
 
    match_inds = np.stack((idx_1, idx_0), axis=1)
    # convert to list
    match_inds = match_inds.tolist()
    return match_inds

def cal_group(input_dict, new_input_dict, match_inds, ratio=0.5):
    group_0 = input_dict["group"]
    group_1 = new_input_dict["group"]

    group_1[group_1 != -1] += group_0.max() + 1

    unique_groups, group_0_counts = np.unique(group_0, return_counts=True)
    group_0_counts = dict(zip(unique_groups, group_0_counts))
    unique_groups, group_1_counts = np.unique(group_1, return_counts=True)
    group_1_counts = dict(zip(unique_groups, group_1_counts))

    # Calculate the group number correspondence of overlapping points
    group_overlap = {}
    for i, j in match_inds:
        group_i = group_1[i]
        group_j = group_0[j]
        if group_i == -1:
            group_1[i] = group_0[j]
            continue
        if group_j == -1:
            continue
        if group_i not in group_overlap:
            group_overlap[group_i] = {}
        if group_j not in group_overlap[group_i]:
            group_overlap[group_i][group_j] = 0
        group_overlap[group_i][group_j] += 1

    # Update group information for point cloud 1
    for group_i, overlap_count in group_overlap.items():
        # for group_j, count in overlap_count.items():
        max_index = np.argmax(np.array(list(overlap_count.values())))
        group_j = list(overlap_count.keys())[max_index]
        count = list(overlap_count.values())[max_index]

        # Skip invalid groups
        if group_j not in group_0_counts or group_i not in group_1_counts:
            continue  # <--- Fix here
        
        total_count = min(group_0_counts[group_j], group_1_counts[group_i]).astype(np.float32)
        # print(count / total_count)
        if count / total_count >= ratio:
            group_1[group_1 == group_i] = group_j
    return group_1

def make_open3d_point_cloud(input_dict_path, voxelize, th):
    dictionary = np.load(input_dict_path, allow_pickle=True)
    input_dict = dict(dictionary.item())
    input_dict["group"] = remove_small_group(input_dict["group"], th)
    # input_dict = voxelize(input_dict)

    xyz = input_dict["coord"]
    if np.isnan(xyz).any():
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd, input_dict

def cal_2_scenes(pcd_list, index, voxel_size, voxelize, th=50, overlapping_ratio=0.5):
    if len(index) == 1:
        pcd0, input_dict_0 = make_open3d_point_cloud(pcd_list[index[0]], voxelize, th)
        return input_dict_0
    # print(index, flush=True)
    input_dict_0 = pcd_list[index[0]]
    input_dict_1 = pcd_list[index[1]]
    pcd0, input_dict_0 = make_open3d_point_cloud(input_dict_0, voxelize, th)
    pcd1, input_dict_1 = make_open3d_point_cloud(input_dict_1, voxelize, th)

    if pcd0 == None:
        if pcd1 == None:
            return None
        else:
            return input_dict_1
    elif pcd1 == None:
        return input_dict_0

    # Cal Dul-overlap
    #match_inds = get_matching_indices(pcd1, pcd0, 1.5 * voxel_size, 1)
    match_inds = get_matching_indices_modified(input_dict_0, input_dict_1)
    pcd1_new_group = cal_group(input_dict_0, input_dict_1, match_inds, ratio=overlapping_ratio)

    #match_inds = get_matching_indices(pcd0, pcd1, 1.5 * voxel_size, 1)
    match_inds = get_matching_indices_modified(input_dict_1, input_dict_0)
    input_dict_1["group"] = pcd1_new_group
    pcd0_new_group = cal_group(input_dict_1, input_dict_0, match_inds, ratio=overlapping_ratio)

    pcd_new_group = np.concatenate((pcd0_new_group, pcd1_new_group), axis=0)
    pcd_new_group = num_to_natural(pcd_new_group)
    pcd_new_coord = np.concatenate((input_dict_0["coord"], input_dict_1["coord"]), axis=0)
    pcd_new_color = np.concatenate((input_dict_0["color"], input_dict_1["color"]), axis=0)
    pcd_new_index = np.concatenate((input_dict_0["index"], input_dict_1["index"]), axis=0)
    pcd_dict = dict(coord=pcd_new_coord, color=pcd_new_color, group=pcd_new_group, index=pcd_new_index)

    pcd_dict = voxelize(pcd_dict)
    return pcd_dict

class SAM3D(object):
    def __init__(self, pointcloud_path, segmentation_folder_path, association_folder_path, keyimage_associations_file_name=None, image_list=None):
        self.pointcloud_path = pointcloud_path
        self.segmentation_folder_path = segmentation_folder_path
        self.association_folder_path = association_folder_path

        # check if the pointcloud file exists
        assert os.path.exists(self.pointcloud_path), 'Pointcloud path does not exist.'

        if pointcloud_path.endswith('.las'):
            with laspy.open(self.pointcloud_path) as las_file:
                header = las_file.header
                self.N_points = header.point_count
            
            pc = laspy.read(self.pointcloud_path)
            self.points = np.vstack((pc.x, pc.y, pc.z)).T
            self.colors = np.vstack((pc.red, pc.green, pc.blue)).T
        elif pointcloud_path.endswith('.npy'):
            mesh_vertices_color = np.load(pointcloud_path)
            self.points = mesh_vertices_color[0]
            self.colors = mesh_vertices_color[1]
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

        basenames = [os.path.basename(f) for f in self.associations_pixel2point_file_paths]
        self.segmentation_file_paths = [os.path.join(self.segmentation_folder_path, f) for f in basenames]
        print('Number of segmentation files: ', len(self.segmentation_file_paths))
        print('Number of pixel2point association files: ', len(self.associations_pixel2point_file_paths))
        print('Number of point2pixel association files: ', len(self.associations_point2pixel_file_paths))


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

        print(self.segmentation_association_pairs[-10:])


        # create a folder to save pcd files
        self.pcd_folder_path = os.path.join(self.association_folder_path, 'pcd')
        if not os.path.exists(self.pcd_folder_path):
            os.makedirs(self.pcd_folder_path)

        self.merged_pcd_folder_path = os.path.join(self.association_folder_path, 'merged_pcd')
        if not os.path.exists(self.merged_pcd_folder_path):
            os.makedirs(self.merged_pcd_folder_path)

        self.voxel_size = 0.001
        self.voxelize = Voxelize(voxel_size=self.voxel_size, mode='train', keys=('coord', 'color', 'group', 'index'))
        self.threshold_points_number = 50
        self.overlapping_ratio = 0.5

    def get_pcd(self):
        # iterate through self.segmentation_association_pairs and add progress bar
        for i in tqdm(range(len(self.segmentation_association_pairs))):
            self.image_id = i
            self.associations_pixel2point = np.load(self.segmentation_association_pairs[i][1], allow_pickle=True)
            self.associations_point2pixel = np.load(self.segmentation_association_pairs[i][2], allow_pickle=True)

            # load the segmentation file
            segmentation = np.load(self.segmentation_association_pairs[i][0], allow_pickle=True)

            # valid points are the points with the corresponding pixel is not -1
            valid_points_index = np.where(self.associations_point2pixel != -1)[0]
            valid_semantics = segmentation[self.associations_point2pixel[valid_points_index][:, 0], self.associations_point2pixel[valid_points_index][:, 1]]

            valid_points = self.points[valid_points_index]
            valid_colors = self.colors[valid_points_index]
            group_ids = num_to_natural(valid_semantics)

            save_dict = dict(coord=valid_points, color=valid_colors, group=group_ids, index=valid_points_index)

            file_path = os.path.join(self.pcd_folder_path, os.path.basename(self.segmentation_association_pairs[i][0]))
            np.save(file_path, save_dict)

    def seg_pcd(self):
        voxel_size = self.voxel_size
        voxelize = self.voxelize
        threshold = self.threshold_points_number
        overlapping_ratio = self.overlapping_ratio

        pcd_list = [f for f in os.listdir(self.pcd_folder_path) if f.endswith('.npy')]
        # sort the pcd files based on the number in the file name
        pcd_list = sorted(pcd_list, key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))
        pcd_list = [os.path.join(self.pcd_folder_path, f) for f in pcd_list]

        # pcd_list = pcd_list[:10]
        
        while len(pcd_list) != 1:
            print(len(pcd_list), flush=True)
            new_pcd_list = []
            for indice in tqdm(pairwise_indices(len(pcd_list))):
                pcd_frame = cal_2_scenes(pcd_list, indice, voxel_size=voxel_size, voxelize=voxelize, overlapping_ratio=overlapping_ratio)
                if pcd_frame is not None:
                    # save the merged pcd
                    file_path = os.path.join(self.merged_pcd_folder_path, os.path.basename(pcd_list[indice[0]]))
                    np.save(file_path, pcd_frame)
                    new_pcd_list.append(file_path)
                    # clear GPU memory
                    del pcd_frame
                    
            pcd_list = new_pcd_list
            pcd_list = sorted(pcd_list, key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))

        pcd_path = pcd_list[0]
        print(pcd_path)
        return pcd_path

    def save_las(self, pcd_path, save_las_path, nearest_interpolation=0):
        dictionary = np.load(pcd_path, allow_pickle=True)
        input_dict = dict(dictionary.item())
        semantics = remove_small_group(input_dict["group"], self.threshold_points_number)
        print("Unique semantics: ", np.unique(semantics).shape[0])
        indices = input_dict["index"]

        points = self.points.copy()
        colors = self.colors.copy()
        

        # create a new las file
        # construct a .las file
        hdr = laspy.LasHeader(version="1.2", point_format=3)
        hdr.scale = [0.0001, 0.0001, 0.0001]  # Example scale factor, adjust as needed
        hdr.offset = np.mean(points, axis=0)

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

        # Initialize the intensity field
        las.intensity = np.zeros(self.N_points)

        # Add semantics
        if nearest_interpolation == 0:
            # make the semantics start from 0. For all semantics < 0, set them to the maximum semantics + 1
            #semantics[semantics < 0] = semantics.max() + 1
            #las.intensity = semantics
            las.intensity[indices] = semantics+2
        elif nearest_interpolation > 0:
            N = nearest_interpolation
            # labeled points are the points with semantics >=0; unlabeled points are the points with semantics < 0
            # labeled_points = points[semantics >= 0]
            # labeled_semantics = semantics[semantics >= 0]
            # unlabeled_points = points[semantics < 0]
            semantics_all = np.zeros(self.N_points) - 1
            semantics_all[indices] = semantics

            labeled_points = points[indices[semantics >= 0]]
            labeled_semantics = semantics_all[indices[semantics >= 0]]
            unlabeled_points = points[semantics_all < 0]

            # construct a KDTree
            tree = cKDTree(labeled_points)

            # find the N nearest neighbors for the unlabeled points
            N = nearest_interpolation
            distances, indices = tree.query(unlabeled_points, k=N)
            nearest_semantics = labeled_semantics[indices].astype(np.int64)
            # find the most frequent semantics
            unlabeled_semantics = np.array([np.argmax(np.bincount(nearest_semantics[i])) for i in range(unlabeled_points.shape[0])])

            # combine the labeled and unlabeled semantics
            combined_semantics = np.zeros(self.N_points)
            combined_semantics[semantics_all >= 0] = semantics_all[semantics_all >= 0]
            combined_semantics[semantics_all < 0] = unlabeled_semantics
            #combined_semantics[semantics >= 0] = labeled_semantics
            #combined_semantics[semantics < 0] = unlabeled_semantics

            las.intensity = combined_semantics
        
        else:
            raise ValueError("nearest_interpolation should be a non-negative integer")

        # Write the LAS file
        las.write(save_las_path)


def sam3d_kubric(scene_path='../../data/kubric_0'):
    pointcloud_path = os.path.join(scene_path, 'reconstructions', 'combined_point_cloud.las')
    segmentation_folder_path = os.path.join(scene_path, 'segmentations_filtered')
    association_folder_path = os.path.join(scene_path, 'associations')

    sam3d = SAM3D(pointcloud_path, segmentation_folder_path, association_folder_path)
    # sam3d.get_pcd()
    # final_pcd_path = sam3d.seg_pcd()
    #final_pcd_path = '../../data/kubric_0/associations/merged_pcd/0.npy'
    #final_pcd_path = os.path.join(scene_path, 'associations', 'merged_pcd', os.path.basename(final_pcd_path))
    final_pcd_path = os.path.join(scene_path, 'associations', 'merged_pcd', '0.npy')
    save_las_path = os.path.join(scene_path, 'associations', 'sam3d.las')
    sam3d.save_las(final_pcd_path, save_las_path, nearest_interpolation=5)

def sam3d_kubric_batch():
    scene_paths = [os.path.join('../../data', f) for f in os.listdir('../../data') if "kubric" in f]
    scene_paths_valid = []
    for scene_path in scene_paths:
        if os.path.exists(os.path.join(scene_path, 'sampro3d')):
            scene_paths_valid.append(scene_path)

    scene_paths = scene_paths_valid

    for scene_path in scene_paths:
        sam3d_kubric(scene_path=scene_path)
        remove_floor(scene_path, 500, 500)


def remove_floor(scene_path, remove_small_N, nearest_interpolation):
    print(scene_path)
     
    pointcloud_path = os.path.join(scene_path, 'reconstructions', 'combined_point_cloud.las')
    save_las_path = os.path.join(scene_path, 'associations', 'sam3d.las')
    # load the pointcloud las
    pc = laspy.read(pointcloud_path)
    points = np.vstack((pc.x, pc.y, pc.z)).T
    colors = np.vstack((pc.red, pc.green, pc.blue)).T
    # semantics_gt is the intensity field in the las file
    semantics_gt = pc.intensity

    # load the sam3d las
    sam3d_pc = laspy.read(save_las_path)
    sam3d_points = np.vstack((sam3d_pc.x, sam3d_pc.y, sam3d_pc.z)).T
    sam3d_colors = np.vstack((sam3d_pc.red, sam3d_pc.green, sam3d_pc.blue)).T
    semantics = sam3d_pc.intensity

    # unqiue semantics and their counts
    unique_semantics, counts = np.unique(semantics_gt, return_counts=True)
    # the most frequent semantics is the floor
    floor_id = unique_semantics[np.argmax(counts)]
    # get non-floor indices
    non_floor_indices = np.where(semantics_gt != floor_id)[0]
    # get floor indices
    floor_indices = np.where(semantics_gt == floor_id)[0]  
    # print semantics number before
    print("Unique semantics before: ", np.unique(semantics).shape[0])

    floor_semantic = np.max(semantics) + 1
    semantics[floor_indices] = floor_semantic

    # print semantics number after
    print("Unique semantics after: ", np.unique(semantics).shape[0])

    # construct a .las file
    hdr = laspy.LasHeader(version="1.2", point_format=3)
    hdr.scale = [0.0001, 0.0001, 0.0001]  # Example scale factor, adjust as needed
    hdr.offset = np.mean(points, axis=0)

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

    # kDTree and nearest interpolation
    labeled_points = points[semantics >= 0]
    labeled_semantics = semantics[semantics >= 0]
    unlabeled_points = points[semantics < 0]

    # construct a KDTree
    tree = cKDTree(labeled_points)

    # find the N nearest neighbors for the unlabeled points
    N = nearest_interpolation
    distances, indices = tree.query(unlabeled_points, k=N)
    nearest_semantics = labeled_semantics[indices]
    # find the most frequent semantics
    unlabeled_semantics = np.array([np.argmax(np.bincount(nearest_semantics[i])) for i in range(unlabeled_points.shape[0])])

    # combine the labeled and unlabeled semantics
    combined_semantics = np.zeros(semantics.shape)
    combined_semantics[semantics >= 0] = labeled_semantics
    combined_semantics[semantics < 0] = unlabeled_semantics

    # add floor semantics
    # las.intensity[floor_indices] = floor_semantics
    # las.intensity[non_floor_indices] = combined_semantics
    las.intensity = combined_semantics

    # print semantics number after
    print("Unique semantics after nearest interpolation: ", np.unique(las.intensity).shape[0])

    # Write the LAS file
    save_las_path_ = save_las_path.replace('.las', '_no_floor.las')
    las.write(save_las_path_)

def sort_semantic_ids(semantics, exclude_largest_semantic=False):
    # Count occurrences of each semantic class
    unique_semantics, semantic_counts = np.unique(semantics, return_counts=True)

    # Sort semantics by count
    sorted_indices = np.argsort(semantic_counts)

    # Create a mapping from old indices to new sorted indices
    index_mapping = {old: new for new, old in enumerate(unique_semantics[sorted_indices])}

    # Update semantics using the mapping
    semantics = np.vectorize(index_mapping.get)(semantics)

    if exclude_largest_semantic:
        # get the indices of the semantics with the largest counts
        unique_semantics, semantic_counts = np.unique(semantics, return_counts=True)
        largest_semantic_indices = unique_semantics[np.argmax(semantic_counts)]

        # check if the largest semantic is the background
        if largest_semantic_indices == -1:
            # get the second largest semantic
            largest_semantic_indices = unique_semantics[np.argsort(semantic_counts)[-2]]
        # get the indices of the semantics to exclude
        exclude_indices = np.where(semantics == largest_semantic_indices)
        # set the semantics to -1
        semantics[exclude_indices] = -2

    return semantics

def add_semantics_to_pointcloud(pointcloud_path, semantics_las_path, save_las_path, remove_small_N=0, nearest_interpolation=0):
    """
    Add semantics to the point cloud and save it as a .las file.

    Parameters
    ----------
    pointcloud_path : str, the path to the .las file
    semantics_path : str, the path to the semantics file
    save_las_path : str, the path to save the .las file
    remove_small_N : int, remove the semantics with numbers smaller than N
    nearest_interpolation : 0, not to use nearest interpolation to assign semantics to the unlabeled points; positive integer, the number of nearest neighbors to use for nearest interpolation

    Returns
    -------
    None
    """
    if pointcloud_path.endswith('.las'):
        points, colors = read_las_file(pointcloud_path)
    elif pointcloud_path.endswith('.npy'):
        points, colors = read_mesh_file(pointcloud_path)
        colors = colors * 255

    semantics_pc = laspy.read(semantics_las_path)
    semantics = semantics_pc.intensity

    print('number of semantics: ', semantics.shape[0])

    semantics_ids = np.unique(semantics)
    N_semantics = len(semantics_ids)

    print("Before removing small semantics: ")
    print('maximum of semantics: ', semantics.max())
    print('number of unique semantics: ', N_semantics)

    # remove the semantics with numbers smaller than a threshold
    for semantics_id in semantics_ids:
        if np.sum(semantics == semantics_id) < remove_small_N:
            semantics[semantics == semantics_id] = -1

    print("After removing small semantics: ")
    print('number of unique semantics: ', len(np.unique(semantics)))
            

    # construct a .las file
    hdr = laspy.LasHeader(version="1.2", point_format=3)
    hdr.scale = [0.0001, 0.0001, 0.0001]  # Example scale factor, adjust as needed
    hdr.offset = np.mean(points, axis=0)

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
    if nearest_interpolation == 0:
        # make the semantics start from 0. For all semantics < 0, set them to the maximum semantics + 1
        semantics[semantics < 0] = semantics.max() + 1
        las.intensity = semantics
    elif nearest_interpolation > 0:
        # labeled points are the points with semantics >=0; unlabeled points are the points with semantics < 0
        labeled_points = points[semantics >= 0]
        labeled_semantics = semantics[semantics >= 0]
        unlabeled_points = points[semantics < 0]

        # construct a KDTree
        tree = cKDTree(labeled_points)

        # find the N nearest neighbors for the unlabeled points
        N = nearest_interpolation
        distances, indices = tree.query(unlabeled_points, k=N)
        nearest_semantics = labeled_semantics[indices]
        # find the most frequent semantics
        unlabeled_semantics = np.array([np.argmax(np.bincount(nearest_semantics[i])) for i in range(unlabeled_points.shape[0])])

        # combine the labeled and unlabeled semantics
        combined_semantics = np.zeros(semantics.shape)
        combined_semantics[semantics >= 0] = labeled_semantics
        combined_semantics[semantics < 0] = unlabeled_semantics

        combined_semantics = sort_semantic_ids(combined_semantics, exclude_largest_semantic=False)

        las.intensity = combined_semantics
    
    else:
        raise ValueError("nearest_interpolation should be a non-negative integer")

    # Write the LAS file
    las.write(save_las_path)

def sam3d_gd():
    scene_path = '../../data/granite_dells'
    pointcloud_path = os.path.join(scene_path, 'SfM_products', 'granite_dells_downsampled_sam3d.las')
    segmentation_folder_path = os.path.join(scene_path, 'segmentations_classes')
    association_folder_path = os.path.join(scene_path, 'associations')

    sam3d = SAM3D(pointcloud_path, segmentation_folder_path, association_folder_path)
    sam3d.get_pcd()
    final_pcd_path = sam3d.seg_pcd()
    final_pcd_path = os.path.join(scene_path, 'associations', 'merged_pcd', 'DJI_0247.npy')
    semantics_las_path = os.path.join(scene_path, 'associations', 'sam3d.las')
    save_las_path = os.path.join(scene_path, 'associations', 'sam3d_refine.las')
    sam3d.save_las(final_pcd_path, semantics_las_path, nearest_interpolation=0)
    add_semantics_to_pointcloud(pointcloud_path, semantics_las_path, save_las_path, remove_small_N=50, nearest_interpolation=50)

if __name__ == '__main__':
    """
    scene_path='../../data/kubric_0'
    sam3d_kubric(scene_path=scene_path)
    """

    #sam3d_kubric_batch()
    sam3d_gd()

    
    