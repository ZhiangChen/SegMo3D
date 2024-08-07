from ssfm.files import *
import laspy
import numpy as np
from scipy.spatial import cKDTree


def add_semantics_to_pointcloud(pointcloud_path, semantics_path, save_las_path, nearest_interpolation=False):
    """
    Add semantics to the point cloud and save it as a .las file.

    Parameters
    ----------
    pointcloud_path : str, the path to the .las file
    semantics_path : str, the path to the semantics file
    save_las_path : str, the path to save the .las file
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

    print('maximum of semantics: ', semantics.max())
    print('number of unique semantics: ', len(np.unique(semantics)))

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


    def shuffle_semantic_ids(self):
        shuffled_indices = np.random.permutation(self.num_unique_semantics)

        # Create a mapping from old indices to new indices
        index_mapping = np.zeros(max(self.unique_semantics) + 1, dtype=int)
        index_mapping[self.unique_semantics] = shuffled_indices

        # Apply the mapping to self.semantics
        self.semantics = index_mapping[self.semantics]


            
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
    pointcloud_path = '../../data/scannet/ssfm/scene0707_00/reconstructions/mesh_vertices_color.npy'
    semantics_path = '../../data/scannet/ssfm/scene0707_00/associations/semantics/semantics_243.npy'
    save_las_path = '../../data/scannet/ssfm/scene0707_00/associations/semantics/semantic_243_interpolated.las'
    add_semantics_to_pointcloud(pointcloud_path, semantics_path, save_las_path, nearest_interpolation=5)

    semantic_pc_file_path = save_las_path
    post_processing = PostProcessing(semantic_pc_file_path)
    post_processing.shuffle_semantic_ids()
    save_las_path = '../../data/scannet/ssfm/scene0707_00/associations/semantics/semantic_243_interpolated_shuffled.las'
    post_processing.save_semantic_pointcloud(save_las_path)