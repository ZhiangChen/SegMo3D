from ssfm.files import *
from ssfm.image_segmentation import *
from ssfm.probabilistic_projection import *

import os
import time
import numpy as np
from collections import defaultdict
import laspy
import torch
import logging
from scipy.stats import mode


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

        # load keyimage association files (.npy)
        keyimage_association_file_path = os.path.join(self.association_folder_path, 'associations_keyimage.npy')
        assert os.path.exists(keyimage_association_file_path), 'Keyimage association file path does not exist.'
        self.keyimage_associations = np.load(keyimage_association_file_path, allow_pickle=True)


        # Load pixel2point association files (.npy) and sort them
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
        logger.info('Number of segmentation-association pairs: {}'.format(len(self.segmentation_association_pairs)))

        # initialize data structures
        self.latest_registered_id = 0  # the latest registered object id
        self.associations_pixel2point = []
        self.associations_point2pixel = []
        self.segmented_objects_images = []
        self.masks = []

        # pre-compute gaussian weights
        self.radius = 2
        self.decaying = 2
        self.likelihoods = compute_gaussian_likelihood(radius=self.radius, decaying=self.decaying)

    def update_object_manager(self, object_id, segmented_objects_image1, point_object2_image2):
        pass

    def search_object2(self, key_image, pixel_object1_image2):
        """
        Within pixels of object1 in image2, search for object2 that has the largest number of semantics ids. 

        Parameters
        ----------
        key_image : int, the key image id
        pixel_object1_image2 : 2D array of shape (N_pixels, 2), where each row is a pixel coordinate (u, v)

        Returns
        -------
        pixel_object2_image2 : list of point indices
        """
        segmented_objects_image2 = self.segmented_objects_images[key_image]
        object_ids_object1_image2 = segmented_objects_image2[pixel_object1_image2[:, 0], pixel_object1_image2[:, 1]]
        unique_ids, counts = np.unique(object_ids_object1_image2, return_counts=True)
        max_count_id = unique_ids[np.argmax(counts)]
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

    def object_registration(self, iou_threshold=0.75, num_workers=4):
        N_images = len(self.segmentation_association_pairs)

        # todo: remove this
        N_points = 28515326
        self.M_keyimages = 5
        self.M_prob = 5

        times = []
        for image_id in range(N_images):
            # logging the current and total number images
            logger.info(f'Processing image {image_id+1}/{N_images}')

            t1 =   time.time()
            # load segmentation and association files
            segmentation_file_path = self.segmentation_association_pairs[image_id][0]
            associations_pixel2point_file_path = self.segmentation_association_pairs[image_id][1]
            associations_point2pixel_file_path = self.segmentation_association_pairs[image_id][2]

            segmented_objects_image1 = np.load(segmentation_file_path, allow_pickle=True)
            associations1_pixel2point = np.load(associations_pixel2point_file_path, allow_pickle=True)
            associations1_point2pixel = np.load(associations_point2pixel_file_path, allow_pickle=True)

            self.segmented_objects_images.append(segmented_objects_image1)
            self.associations_pixel2point.append(associations1_pixel2point)
            self.associations_point2pixel.append(associations1_point2pixel)

            mask1 = associations1_pixel2point != -1
            self.masks.append(mask1)

            # pre-compute padded segmentation and normalized likelihoods
            image_height, image_width = segmented_objects_image1.shape
            self.padded_segmentation = -np.ones((2*self.radius+image_height+2, 2*self.radius+image_width+2)).astype(np.int16)
            self.padded_segmentation[self.radius+1:self.radius+image_height+1, self.radius+1:self.radius+image_width+1] = segmented_objects_image1
            self.normalized_likelihoods = np.zeros(int(segmented_objects_image1.max() + 1), dtype=np.float16)

            N_objects = int(segmented_objects_image1.max() + 1)

            self.object_manager = dict()  # the key is the object id and the value is a list of registered object ids.

            for object_id in range(N_objects):
                logger.info(f'Processing object {object_id+1}/{N_objects}')

                pixel_object1_image1_bool = segmented_objects_image1 == object_id
                
                point_object1_image1 = associations1_pixel2point[pixel_object1_image1_bool] # point_object1_image1 is a list of point ids
                point_object1_image1_bool = np.zeros(N_points, dtype=bool)
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
                    self.update_object_manager(object_id, segmented_objects_image1, None)
                else:
                    # iterate over all key images
                    for key_image in keyimages:
                        associations2_pixel2point = self.associations_pixel2point[key_image]
                        associations2_point2pixel = self.associations_point2pixel[key_image]

                        pixel_object1_image2 = associations2_point2pixel[point_object1_image1]
                        pixel_object1_image2 = pixel_object1_image2[pixel_object1_image2[:, 0] != -1]
                        point_object1_image2 = associations2_pixel2point[pixel_object1_image2[:, 0], pixel_object1_image2[:, 1]]  # point_object1_image2 is a list of point ids

                        pixel_object2_image2 = self.search_object2(key_image, pixel_object1_image2)

                        point_object2_image2 = associations2_pixel2point[pixel_object2_image2[:, 0], pixel_object2_image2[:, 1]]  # point_object2_image2 is a list of point ids
                        pixel_object2_image1 = associations1_point2pixel[point_object2_image2]
                        pixel_object2_image1 = pixel_object2_image1[pixel_object2_image1[:, 0] != -1]
                        point_object2_image1 = associations1_pixel2point[pixel_object2_image1[:, 0], pixel_object2_image1[:, 1]]

                        iou = self.calculate_3D_IoU(point_object1_image2, point_object2_image1)
                        print("image_id: {}, object_id: {}, key_image: {}, iou: {}".format(image_id, object_id, key_image, iou))

                        if iou >= iou_threshold:
                            self.update_object_manager(object_id, segmented_objects_image1, point_object2_image2)
                        else:
                            self.update_object_manager(object_id, segmented_objects_image1, None)


            t2 = time.time()
            times.append(t2-t1)
            print("time elapsed for image {}: {}".format(image_id+1, t2-t1))
        
        self.times = times

                    

if __name__ == "__main__":
    #now we will Create and configure logger 
    logging.basicConfig(filename="std_test.log", 
                        format='%(asctime)s %(message)s', 
                        filemode='w') 

    #Let us Create an object 
    global logger
    logger=logging.getLogger() 

    #Now we are going to Set the threshold of logger to DEBUG 
    logger.setLevel(logging.DEBUG) 

    # Set paths
    pointcloud_path = '../../data/model.las'
    segmentation_folder_path = '../../data/mission_2_segmentations'
    image_folder_path = '../../data/mission_2'
    association_folder_path = '../../data/mission_2_associations_parallel'

    # Create object registration
    t1 = time.time()
    obr = ObjectRegistration(pointcloud_path, segmentation_folder_path, association_folder_path)
    t2 = time.time()
    print('Time elapsed for creating object registration: {}'.format(t2-t1))

    # Run object registration
    obr.object_registration()
    

