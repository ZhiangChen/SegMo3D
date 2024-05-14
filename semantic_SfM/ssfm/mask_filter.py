import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import xml.etree.ElementTree as ET
import time
from scipy import ndimage

from joblib import Parallel, delayed
from tqdm import tqdm


class SimpleMaskFilter(object):
    def __init__(self, configs) -> None:
        self.window_size = configs.get('window_size', 5)
        self.depth_folder = configs.get('depth_folder', None)
        self.output_folder = configs.get('output_folder', None)
        self.area_upper_threshold = configs.get('area_upper_threshold', 9)
        self.area_lower_threshold = configs.get('area_lower_threshold', 0.01)
        self.camera_parameter_file = configs.get('camera_parameter_file', None)
        #self.focal_length, self.pixel_size = self.read_camera_parameters_agisoft(self.camera_parameter_file)
        self.focal_length = 0.010689196654678247
        self.pixel_size = 2.005835530993016e-06
        self.erosion_kernel_size = configs.get('erosion_kernel_size', 0)
        self.erosion_iteration = configs.get('erosion_iteration', 2)
        self.background_mask = configs.get('background_mask', False)
        

    def calculate_masked_area(self, area_image, masks, id):
        mask = masks == id
        masked_area = np.sum(area_image[mask])
        return masked_area
    
    def read_camera_parameters_agisoft(self, file_path):
        # check if the file exists
        assert os.path.exists(file_path)

        cameras = dict()

        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Extract image dimensions
        image_dimensions = root.find('.//ImageDimensions')
        width = int(image_dimensions.find('Width').text)
        height = int(image_dimensions.find('Height').text)

        cameras['width'] = width
        cameras['height'] = height

        # Get camera intrinsics
        sensor_diagonal_mm = root.find('.//SensorSize').text
        focal_length_mm = root.find('.//FocalLength').text

        # Convert camera intrinsics to meters
        sensor_diagonal_m = float(sensor_diagonal_mm) / 1000
        focal_length_m = float(focal_length_mm) / 1000

        # Calculate aspect ratio
        aspect_ratio = width / height

        # Calculate sensor width and height using aspect ratio and diagonal
        sensor_height = np.sqrt(sensor_diagonal_m**2 / (1 + aspect_ratio**2))
        sensor_width = aspect_ratio * sensor_height

        # Calculate pixel size
        pixel_size_width = sensor_width / width
        pixel_size_height = sensor_height / height

        pixel_size_m = np.sqrt(pixel_size_width * pixel_size_height) 

        return focal_length_m, pixel_size_m

    def moving_average_filter(self, masks):
        mask_sizes = []
        for i in range(np.max(masks)+1):
            mask_sizes.append(np.sum(masks == i))
        mask_sizes = np.array(mask_sizes)
        window = np.ones(int(self.window_size)) / float(self.window_size)
        smoothed_mask_sizes = np.convolve(mask_sizes, window, 'same')
        valid_masks_id = np.where(mask_sizes > smoothed_mask_sizes)[0]
        valid_masks_size = mask_sizes[valid_masks_id]
        return valid_masks_id

    def area_size_filter(self, masks, file_name):
        depth_file_path = os.path.join(self.depth_folder, file_name)
        assert os.path.exists(depth_file_path), "The depth file does not exist: {}".format(depth_file_path)
        depth_image = np.load(depth_file_path)
        # replace inf with 0
        depth_image = np.nan_to_num(depth_image, nan=0, posinf=0, neginf=0)
        area_image = ((depth_image / self.focal_length) ** 2) * (self.pixel_size ** 2)
        mask_areas = []
        for i in range(np.max(masks)+1):
            mask_areas.append(self.calculate_masked_area(area_image, masks, i))
        mask_areas = np.array(mask_areas)
        valid_masks_id = np.where((mask_areas > self.area_lower_threshold) & (mask_areas < self.area_upper_threshold))[0]
        return valid_masks_id
    

    def erode_mask(self, masks):
        num_masks = np.max(masks) + 1
        kernel = np.ones((self.erosion_kernel_size, self.erosion_kernel_size), np.uint8)
        temp_masks = np.zeros_like(masks) -1 # Temporary mask storage

        for i in range(num_masks):
            mask = (masks == i).astype(np.uint8)  # Convert mask to uint8
            eroded_mask = cv2.erode(mask, kernel, iterations=self.erosion_iteration)
            temp_masks[eroded_mask == 1] = i  # Update temporary masks where eroded_mask is 1
        
        if self.background_mask:
            mask = (masks == -1).astype(np.uint8)
            eroded_mask = cv2.erode(mask, kernel, iterations=self.erosion_iteration)
            #temp_masks[eroded_mask == 1] = num_masks
            temp_masks[mask == 1] = num_masks
        
        return temp_masks

    def filter_segmentation_file(self, segmentation_file_path):
        masks = np.load(segmentation_file_path)
        file_name = os.path.basename(segmentation_file_path)
        valid_masks_id_maf = self.moving_average_filter(masks)
        if self.erosion_kernel_size > 0:
            masks = self.erode_mask(masks)
        valid_masks_id_asf = self.area_size_filter(masks, file_name)
        valid_masks_id = np.intersect1d(valid_masks_id_maf, valid_masks_id_asf)
        if self.background_mask:
            if np.max(masks) not in valid_masks_id:
                valid_masks_id = np.append(valid_masks_id, np.max(masks))
        filtered_masks = np.zeros_like(masks) - 1
        for i, mask_id in enumerate(valid_masks_id):
            filtered_masks[masks == mask_id] = i     
        output_file_path = os.path.join(self.output_folder, file_name)
        np.save(output_file_path, filtered_masks)
        

    def filter_batch_processes(self, segmentation_folder_path, num_processes=8):
        assert os.path.exists(segmentation_folder_path), "The folder does not exist"
        self.segmentation_files = [os.path.join(segmentation_folder_path, f) for f in os.listdir(segmentation_folder_path) if f.endswith('.npy')]

        # sort the files
        self.segmentation_files.sort()

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # print the number of files
        print(f"Total number of files: {len(self.segmentation_files)}")
        Parallel(n_jobs=num_processes)(delayed(self.filter_segmentation_file)(f) for f in tqdm(self.segmentation_files))


if __name__ == "__main__":
    segmentation_folder_path = "../../data/courtright/segmentations"

    configs = {
        'window_size': 5,
        'depth_folder': "../../data/courtright/associations/depth",
        'output_folder': "../../data/courtright/segmentations_filtered",
        'area_upper_threshold': 6,
        'area_lower_threshold': 0.01,
        'erosion_kernel_size': 5,
        'erosion_iteration':1,
        'camera_parameter_file': "../../data/courtright/SfM_products/agisoft_cameras.xml",
        'background_mask': True
    }

    mask_filter = SimpleMaskFilter(configs)
    
    t1 = time.time()
    mask_filter.filter_segmentation_file('../../data/courtright/segmentations/DJI_0580.npy')
    #mask_filter.filter_batch_processes(segmentation_folder_path)
    t2 = time.time()
    print(f"Time taken: {t2 - t1} seconds")

    