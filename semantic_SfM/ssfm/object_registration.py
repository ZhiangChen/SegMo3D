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
import numba


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
        self.association_p2i = dict()  # the key is the point index and the value is a list of images that include the projection of the point.
        self.pc_segmentation = dict()  # the key is the point index and the value is a list of object probabilities.
        self.associations_pixel2point = []
        self.associations_point2pixel = []
        self.masks = []
    
    def update_key_images(self, point_object1_image1, image_index):
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

    def calculate_3D_IoU(self):
        pass

    def update_object_manager(self, pixel_object1_image1, segmented_objects_image1, point_object2_image2, merge):
        """
        Update object_manager. If merge is False, the object_manager is not updated. If merge is True and point_object2_image2 
        is None, the object_id from pixel_object1_image1 is updated with None. If merge is True and point_object2_image2 is not 
        None, the object_manager is updated with the object_id from pixel_object1_image1 with the maximum probability.
        
        Parameters
        ----------
        pixel_object1_image1 : 2D array of shape (N_pixels, 2), where each row is a pixel coordinate (u, v)
        segmented_objects_image1 : 2D array of shape (width, height), where each element is an object id
        point_object2_image2 : list of point indices
        merge : boolean
        
        Returns
        -------
        None
        """
        
        if merge:
            # get object id from pixel_object1_image1 and segmented_objects_image1
            object_id = int(segmented_objects_image1[tuple(pixel_object1_image1[0])])

            if point_object2_image2 == None:  # if point_object2_image2 is None, update the object_id with None
                registered_objects_id = None
            else:  # if point_object2_image2 is not None, get the object id with the maximum probability
                point_object_prob_sum = dict()
                for point in point_object2_image2:
                    point_object_probs = self.pc_segmentation[point]  # dictionary of object probabilities for a point, where the key is the object id and the value is the normalized object probability
                    for object_id, prob in point_object_probs.items():
                        if object_id not in point_object_prob_sum.keys():
                            point_object_prob_sum[object_id] = prob
                        else:
                            point_object_prob_sum[object_id] += prob
                
                registered_objects_id = max(point_object_prob_sum, key=point_object_prob_sum.get)
            
            # if there is no object id in object_manager, add the object id and registered_objects_id to object_manager
            if object_id not in self.object_manager.keys():
                self.object_manager[object_id] = [registered_objects_id]
            else:
                self.object_manager[object_id].append(registered_objects_id)
        else:
            pass

    def purge_object_manager(self):
        pass

    def update_association_p2i(self):
        pass

    def update_pc_segmentation(self, association1):
        pass

    def get_object_id_from_points(self, point_object2_image2):
        if point_object2_image2 == None:
            return None
        else:
            pass

    def object_registration(self):
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
                t6 = time.time()
                print('Time to get point indices of pixel_object1_image1: {}'.format(t6 - t5))

                # get key images of point_object1_image1
                key_images_lists = [self.association_p2i[point] for point in point_object1_image1 if point in self.association_p2i.keys()]
                key_images = np.unique([item for sublist in key_images_lists for item in sublist])
            
                t7 = time.time()
                print('Time to get key images: {}'.format(t7 - t6))

                t7 = time.time()
                # update association_p2i
                self.update_key_images(point_object1_image1, i)
                t8 = time.time()
                print('Time to update association_p2i: {}'.format(t8 - t7))

                if len(key_images) == 0:
                    # update object_manager
                    self.update_object_manager(pixel_object1_image1, segmented_objects_image1, None, True)
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

                print('---------------------------------------------------')


                


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






    

