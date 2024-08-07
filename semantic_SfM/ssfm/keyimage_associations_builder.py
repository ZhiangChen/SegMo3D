import numpy as np
import os
import yaml
from tqdm import tqdm
from joblib import Parallel, delayed

class KeyimageAssociationsBuilder(object):
    def __init__(self, read_folder_path, segmentation_folder_path):
        """
        Arguments:
            read_folder_path (str): The path to the folder containing the point2pixel files and pixel2point files.
            segmentation_folder_path (str): The path to the folder containing the segmentation files.
        """
        self.read_folder_path = read_folder_path
        self.segmentation_folder_path = segmentation_folder_path

        pixel2point_folder_path = os.path.join(read_folder_path, 'pixel2point')
        assert os.path.exists(pixel2point_folder_path), 'Pixel2point folder path does not exist.'
        self.pixel2point_file_paths = [os.path.join(pixel2point_folder_path, f) for f in os.listdir(pixel2point_folder_path) if f.endswith('.npy')]
        self.pixel2point_file_paths.sort()
        self.N_images = len(self.pixel2point_file_paths)
        
        # get the number of points
        point2pixel_folder_path = os.path.join(read_folder_path, 'point2pixel')  
        assert os.path.exists(point2pixel_folder_path), 'Point2pixel folder path does not exist.'  
        self.point2pixel_file_paths = [os.path.join(point2pixel_folder_path, f) for f in os.listdir(point2pixel_folder_path) if f.endswith('.npy')]  
        point2pixel = np.load(self.point2pixel_file_paths[0])
        self.N_points = point2pixel.shape[0]

        assert os.path.exists(segmentation_folder_path), 'Segmentation folder path does not exist.'
        self.segmentation_file_paths = [os.path.join(segmentation_folder_path, f) for f in os.listdir(segmentation_folder_path) if (f.endswith('.npy') and f in os.listdir(pixel2point_folder_path))]
        self.segmentation_file_paths.sort()

        # check if the number of images is the same
        assert len(self.segmentation_file_paths) == self.N_images, 'The number of images is not the same as the number of segmentation files.'
        self.segmentation_pixel2point_pairs = []
        for i in range(len(self.segmentation_file_paths)):
            segmentation_file_path = self.segmentation_file_paths[i]
            pixel2point_file_path = self.pixel2point_file_paths[i]
            # check if the base of the file name is the same
            assert os.path.basename(segmentation_file_path)[:-4] == os.path.basename(pixel2point_file_path)[:-4], 'The base of the file name is not the same.'
            self.segmentation_pixel2point_pairs.append((segmentation_file_path, pixel2point_file_path))

    def build_associations(self):
        # build associations
        self.associations_keyimage = np.full((self.N_points, self.N_images), False, dtype=bool)

        for k in tqdm(range(len(self.point2pixel_file_paths))):
            segmentation = np.load(self.segmentation_pixel2point_pairs[k][0])
            pixel2point = np.load(self.segmentation_pixel2point_pairs[k][1])
            pixel2point[segmentation == -1] = -1
            valid_point_ids = pixel2point[pixel2point != -1]
            self.associations_keyimage[valid_point_ids, k] = True

        # save associations
        save_file_path = os.path.join(self.read_folder_path, 'associations_keyimage.npy')
        np.save(save_file_path, self.associations_keyimage)

        # save image file names 
        image_file_names = [os.path.basename(f) for f in self.segmentation_file_paths]
        save_file_path = os.path.join(self.read_folder_path, 'keyimages.yaml')
        with open(save_file_path, 'w') as f:
            yaml.dump(image_file_names, f)
        
    def read_associations(self, associations_keyimage_file_path=None):
        """
        Read the associations_keyimage file
        """
        if associations_keyimage_file_path is not None:
            self.associations_keyimage = np.load(associations_keyimage_file_path)
        else:
            associations_keyimage_file_path = os.path.join(self.read_folder_path, 'associations_keyimage.npy')
            assert os.path.exists(associations_keyimage_file_path), 'Associations keyimage file does not exist.'
            self.associations_keyimage = np.load(associations_keyimage_file_path)
        
        # print the associations are loaded
        print(f"Loaded associations keyimage file from {associations_keyimage_file_path}")

        # assert the number of images is the same
        assert self.associations_keyimage.shape[1] == len(self.segmentation_file_paths), 'The number of images is not the same as the number of segmentation files.'
    
    def find_min_cover(self):
        """
        Find the minimum cover of the associations_keyimage
        """
        # find the number of True values in each column of the associations_keyimage
        num_true_values = np.sum(self.associations_keyimage, axis=1)
        N_points = num_true_values.shape[0]
        # find the minimum number in the num_true_values
        min_num_true_values = np.min(num_true_values)
        # get the number of elements equal to 0 in the num_true_values
        num_zeros = np.sum(num_true_values == 0)
        # get the number of points covered by less than or equal to 1 image
        num_one = np.sum(num_true_values <= 1)
        # get the number of points covered by less than or equal to 3 images
        num_three = np.sum(num_true_values <= 3)
        # get the number of points covered by less than or equal to 5 images
        num_five = np.sum(num_true_values <= 5)

        # Prepare header and row format
        header = "| {0: <60} | {1: <10} | {2: <20} |"
        row = "| {0: <60} | {1: <10d} | {2: <20.2f} |"

        # Print table header
        print(header.format("Metric", "Count", "Percentage"))
        print("-" * 100)

        # Print table rows
        print(row.format("Number of points not covered by any image                 ", num_zeros, num_zeros / N_points * 100))
        print(row.format("Number of points covered by less than or equal to 1 image ", num_one, num_one / N_points * 100))
        print(row.format("Number of points covered by less than or equal to 3 images", num_three, num_three / N_points * 100))
        print(row.format("Number of points covered by less than or equal to 5 images", num_five, num_five / N_points * 100))


    def refine(self, remove_ratio=0.3):
        N_image_per_point = np.sum(self.associations_keyimage, axis=1)
        _, N = self.associations_keyimage.shape

        importance_scarce_points_list = []
        for i in tqdm(range(N)):
            association = self.associations_keyimage[:, i]
            importance_scarce_points = 5-N_image_per_point[association & (N_image_per_point !=0)]
            importance_scarce_points = np.sum(importance_scarce_points[importance_scarce_points > 0])
            importance_scarce_points_list.append(importance_scarce_points)

        importance_scarce_points_list = np.array(importance_scarce_points_list)
        sorted_indices = np.argsort(importance_scarce_points_list)
        # remove the data with the smallest importance: self.segmentation_file_paths, self.associations_keyimage
        remove_indices = sorted_indices[:int(remove_ratio*N)]
        self.segmentation_file_paths = [self.segmentation_file_paths[i] for i in range(N) if i not in remove_indices]
        self.associations_keyimage = np.delete(self.associations_keyimage, remove_indices, axis=1)

        # save self.segmentation_file_paths and self.associations_keyimage
        image_file_names = [os.path.basename(f) for f in self.segmentation_file_paths]
        save_file_path = os.path.join(self.read_folder_path, 'refined_keyimages.yaml')
        with open(save_file_path, 'w') as f:
            yaml.dump(image_file_names, f)

        save_file_path = os.path.join(self.read_folder_path, 'refined_associations_keyimage.npy')
        np.save(save_file_path, self.associations_keyimage)

        


if __name__ == '__main__':
    smc_solver = KeyimageAssociationsBuilder('../../data/box_canyon_park/associations', '../../data/box_canyon_park/segmentations')
    #smc_solver.build_associations(image_patch_path='../../data/box_canyon_park/DJI_photos_split/image_patches.yaml')
    smc_solver.read_associations('../../data/box_canyon_park/associations/associations_keyimage.npy')
    #smc_solver.find_min_cover()
    smc_solver.refine(0.5)
    