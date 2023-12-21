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
        #self.points, self.colors = read_las_file(self.pointcloud_path)

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
        self.assocation_p2i = dict()  # the key is the point index and the value is a list of images that include the projection of the point.
        self.pc_segmentation = dict()  # the key is the point index and the value is a list of object probabilities.

    def get_key_images(self, point_object1_image1):
        key_images = []
        for point in point_object1_image1:
            if point in self.assocation_p2i.keys():
                key_images += self.assocation_p2i[point]
            else:
                pass
        return key_images
    
    def update_key_images(self, point_object1_image1, image_index):
        for point in point_object1_image1:
            if point not in self.assocation_p2i.keys():
                self.assocation_p2i[point] = [image_index]
            else:
                self.assocation_p2i[point].append(image_index)

    def search_object2(self):
        pass

    def calculate_3D_IoU(self):
        pass

    def update_object_manager(self):
        pass

    def purge_object_manager(self):
        pass

    def update_association_p2i(self):
        pass

    def update_pc_segmentation(self):
        pass

    def get_object_id_from_points(self):
        pass

    def object_registration(self):
        M_images = len(self.segmentation_association_pairs)

        # iterate over all images
        for i in range(M_images):
            # load segmentation and association files
            segmentation_file_path, association_file_path = self.segmentation_association_pairs[i]
            print(segmentation_file_path, association_file_path)
            segmented_objects_image1 = np.load(segmentation_file_path).transpose(1, 0)  # tranpose from (height, width) to (width, height)
            association1 = np.load(association_file_path)  # association1 is 2D array for point-pixel association, where each row is a point index and each column is a pixel coordinate (u, v). u is along the width and v is along the height.
            inv_association1, point_object1 = inverse_associations(association1)  # inv_association1 is a dictionary for pixel-point association

            N_objects = int(segmented_objects_image1.max() + 1)
            print('Number of objects in image {}: {}'.format(i, N_objects))

            pixel_objects_image1 = inv_association1.keys()
            # iterate over all objects in image1
            for j in range(N_objects):
                # get pixel coordinates of segmented_objects_image1 == j
                pixel_object1_image1 = np.argwhere(segmented_objects_image1 == j)  # 2D array of shape (N_pixels, 2), where each row is a pixel coordinate (u, v)

                # get point indices of pixel_object1_image1 using inv_association1. However, not all pixels in pixel_object1_image1 have a corresponding point index.
                point_object1_image1 = [inv_association1[tuple(pixel)] for pixel in pixel_object1_image1 if tuple(pixel) in pixel_objects_image1]
                
                # get key images of point_object1_image1
                key_images = self.get_key_images(point_object1_image1)

                if len(key_images) == 0:
                    pass
                    # update association_p2i
                    self.update_key_images(point_object1_image1, i)
                    # update object manager
                    self.update_object_manager(point_object1_image1, i)
                    # update pc_segmentation
                    self.update_pc_segmentation(point_object1_image1, i)

                # iterate over all key images
                for key_image in key_images:
                    pass

                return
                


                


if __name__ == "__main__":
    # Set paths
    pointcloud_path = '../../data/model.las'
    segmentation_folder_path = '../../data/mission_2_segmentations_test'
    image_folder_path = '../../data/mission_2'
    association_folder_path = '../../data/mission_2_associations_test'

    # Create object registration object
    object_registration = ObjectRegistration(pointcloud_path, segmentation_folder_path, association_folder_path)
    object_registration.object_registration()






    

