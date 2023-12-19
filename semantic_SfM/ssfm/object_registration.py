from ssfm.files import *
from ssfm.image_segmentation import *
from ssfm.probabilistic_projection import *

import os

class ObjectRegistration(object):
    def __init__(self, pointcloud_path, segmentation_folder_path, association_folder_path):
        self.pointcloud_path = pointcloud_path
        self.segmentation_folder_path = segmentation_folder_path
        self.association_folder_path = association_folder_path

        # Load pointcloud
        self.points, self.colors = read_las_file(self.pointcloud_path)

        # Load segmentation files (.npy) and sort them
        # check if the folder exists
        assert os.path.exists(self.segmentation_folder_path), 'Segmentation folder path does not exist.'
        self.segmentation_file_paths = [f for f in os.listdir(self.segmentation_folder_path) if f.endswith('.npy')]
        self.segmentation_file_paths.sort()   

        # Load association files (.npy) and sort them
        # check if the folder exists
        assert os.path.exists(self.association_folder_path), 'Association folder path does not exist.'
        self.association_file_paths = [f for f in os.listdir(self.association_folder_path) if f.endswith('.npy')]
        self.association_file_paths.sort()

        # Check if the number of segmentation files and association files are the same
        assert len(self.segmentation_file_paths) == len(self.association_file_paths), 'The number of segmentation files and association files are not the same.'

        # create segmentation-association pairs
        self.segmentation_association_pairs = []
        for i in range(len(self.segmentation_file_paths)):
            self.segmentation_association_pairs.append((self.segmentation_file_paths[i], self.association_file_paths[i]))

    def start_registration(self):
        pass            


if __name__ == "__main__":
    # Set paths
    pointcloud_path = '../../data/model.las'
    segmentation_folder_path = '../../data/mission_2_segmentations'
    image_folder_path = '../../data/mission_2'
    association_folder_path = '../../data/mission_2_associations'

    # Create object registration object
    object_registration = ObjectRegistration(pointcloud_path, segmentation_folder_path, association_folder_path)







    

