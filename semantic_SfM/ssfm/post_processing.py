from ssfm.files import *
import laspy
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


def filter_semantics(semantics, ratio, MIN_num_semantics):
    """
    Filter the semantics by keeping the semantics with the highest counts.
    
    Parameters
    ----------
    semantics : numpy array, the semantics to filter
    ratio : float, the ratio of semantics to keep
    MIN_num_semantics : int, the minimum number of semantics to keep
    
    Returns
    -------
    numpy array, the filtered semantics
    """
    # get the unique semantics and their counts
    unique_semantics, semantic_counts = np.unique(semantics, return_counts=True)
    # get the number of unique semantics
    num_unique_semantics = unique_semantics.shape[0]
    # get the number of semantics to keep
    num_semantics_to_keep = int(num_unique_semantics * ratio)
    if num_semantics_to_keep < MIN_num_semantics:
        num_semantics_to_keep = MIN_num_semantics
    # get the indices of the semantics to keep
    indices = np.argsort(semantic_counts)[-num_semantics_to_keep:]
    # get the semantics to keep
    semantics_to_keep = unique_semantics[indices]
    # get the indices of the semantics to remove
    indices = np.where(np.isin(semantics, semantics_to_keep, invert=True))
    # remove the semantics
    semantics[indices] = -1
    return semantics




def add_semantics_to_pointcloud(pointcloud_path, semantics_path, save_las_path, remove_small_N=0, nearest_interpolation=False):
    """
    Add semantics to the point cloud and save it as a .las file.

    Parameters
    ----------
    pointcloud_path : str, the path to the .las file
    semantics_path : str, the path to the semantics file
    save_las_path : str, the path to save the .las file
    remove_small_N : int, remove the semantics with numbers smaller than N
    nearest_interpolation : False, not to use nearest interpolation to assign semantics to the unlabeled points; positive integer, the number of nearest neighbors to use for nearest interpolation

    Returns
    -------
    None
    """
    if pointcloud_path.endswith('.las'):
        points, colors = read_las_file(pointcloud_path)
    elif pointcloud_path.endswith('.npy'):
        points, colors = read_mesh_file(pointcloud_path)
        colors = colors * 255

    semantics = np.load(semantics_path)

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
    if nearest_interpolation is False:
        las.intensity = semantics
    else:
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

        las.intensity = combined_semantics

    # Write the LAS file
    las.write(save_las_path)



class PostProcessing(object):
    def __init__(self, semantic_pc_file_path) -> None:
        # assert if the semantic pointcloud file exists
        assert os.path.exists(semantic_pc_file_path), "Semantic pointcloud file does not exist"
        # read the semantic pointcloud file
        pc = laspy.read(semantic_pc_file_path)

        # Get the points
        x = pc.x.scaled_array()
        y = pc.y.scaled_array()
        z = pc.z.scaled_array()
        r = pc.red
        g = pc.green
        b = pc.blue

        # Stack points: (N, 3), where N is the number of points 
        self.points = np.vstack((x, y, z)).T

        # Stack colors: (N, 3), where N is the number of points
        self.colors = np.vstack((r, g, b)).T

        # get the semantics
        self.semantics = pc.intensity

        # get the unique semantics and their counts
        self.unique_semantics, self.semantic_counts = np.unique(self.semantics, return_counts=True)

        # get the number of unique semantics
        self.num_unique_semantics = self.unique_semantics.shape[0]
        print("Number of unique semantics: ", self.num_unique_semantics)


    def shuffle_semantic_ids(self, exclude_largest_semantic=False):
        shuffled_indices = np.random.permutation(self.num_unique_semantics)
        # Create a mapping from old indices to new indices
        index_mapping = np.zeros(max(self.unique_semantics) + 1, dtype=int)
        index_mapping[self.unique_semantics] = shuffled_indices
        # Save the index of the background in semantics
        background_index = np.where(self.semantics == -1)
        # Apply the mapping to self.semantics
        self.semantics = index_mapping[self.semantics]
        # Set the background to -1
        self.semantics[background_index] = -1

        if exclude_largest_semantic:
            # get the indices of the semantics with the largest counts
            unique_semantics, semantic_counts = np.unique(self.semantics, return_counts=True)
            largest_semantic_indices = unique_semantics[np.argmax(semantic_counts)]

            # check if the largest semantic is the background
            if largest_semantic_indices == -1:
                # get the second largest semantic
                largest_semantic_indices = unique_semantics[np.argsort(semantic_counts)[-2]]
            # get the indices of the semantics to exclude
            exclude_indices = np.where(self.semantics == largest_semantic_indices)
            # set the semantics to -1
            self.semantics[exclude_indices] = -2

            
    def save_semantic_pointcloud(self, save_las_path):
        # construct a .las file
        hdr = laspy.LasHeader(version="1.2", point_format=3)
        hdr.scale = [0.0001, 0.0001, 0.0001]  # Example scale factor, adjust as needed
        hdr.offset = np.min(self.points, axis=0)

        # Create a LasData object
        las = laspy.LasData(hdr)

        # Add points
        las.x = self.points[:, 0]
        las.y = self.points[:, 1]
        las.z = self.points[:, 2]

        # Add colors
        las.red = self.colors[:, 0]
        las.green = self.colors[:, 1]
        las.blue = self.colors[:, 2]

        # Add semantics
        las.intensity = self.semantics

        # Write the LAS file
        las.write(save_las_path)


if __name__ == "__main__":
    """
    semantic_pc_file_path = '../../data/box_canyon_park/semantic_model.las'
    post_processing = PostProcessing(semantic_pc_file_path)
    post_processing.shuffle_semantic_ids_2()
    save_las_path = '../../data/box_canyon_park/semantic_model_shuffled.las'
    post_processing.save_semantic_pointcloud(save_las_path)
    """
    #"""
    pointcloud_path = '../../data/scannet/ssfm/scene0707_00/reconstructions/mesh_vertices_color.npy'
    semantics_path = '../../data/scannet/ssfm/scene0707_00/associations/semantics/semantics_276.npy'
    save_las_path = '../../data/scannet/ssfm/scene0707_00/associations/semantics/semantic_276_interpolated.las'
    add_semantics_to_pointcloud(pointcloud_path, semantics_path, save_las_path, nearest_interpolation=5, filter_random_semantics=True, ratio=0.1, MIN_num_semantics=80)

    semantic_pc_file_path = save_las_path
    post_processing = PostProcessing(semantic_pc_file_path)
    post_processing.shuffle_semantic_ids()
    save_las_path = '../../data/scannet/ssfm/scene0707_00/associations/semantics/semantic_276_interpolated_shuffled.las'
    post_processing.save_semantic_pointcloud(save_las_path)
    """
    pointcloud_path = '../../data/scene0000_00/reconstructions/mesh_vertices_color.npy'
    semantics_path = '../../data/scene0000_00/associations/semantics/semantics_613.npy'
    save_las_path = '../../data/scene0000_00/associations/semantics/semantic_613_interpolated.las'
    add_semantics_to_pointcloud(pointcloud_path, semantics_path, save_las_path, nearest_interpolation=5, filter_random_semantics=False, ratio=0.5, MIN_num_semantics=80)

    semantic_pc_file_path = save_las_path
    post_processing = PostProcessing(semantic_pc_file_path)
    post_processing.shuffle_semantic_ids()
    save_las_path = '../../data/scene0000_00/associations/semantics/semantic_613_interpolated_shuffled_filtered.las'
    post_processing.save_semantic_pointcloud(save_las_path)
    """