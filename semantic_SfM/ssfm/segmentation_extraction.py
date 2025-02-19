import os
import json
import numpy as np
import cv2
from ssfm.files import *
from joblib import Parallel, delayed
from tqdm import tqdm
from threading import Lock

class SegmentationExtraction:
    def __init__(self, image_folder, segmentation_folder, output_folder):
        # assert exists 
        assert os.path.exists(segmentation_folder), "Segmentation folder does not exist"
        assert os.path.exists(output_folder), "Output folder does not exist"
        assert os.path.exists(image_folder), "Image folder does not exist"

        # load images
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.JPG')]
        self.image_folder = image_folder
        # load image size
        image = cv2.imread(os.path.join(image_folder, self.image_files[0]))
        if image is None:
            raise ValueError("Image is None")
        self.image_height, self.image_width = image.shape[:2]

        # load segmentation
        self.segmentation_files = [f for f in os.listdir(segmentation_folder) if f.endswith('.json')]
        self.segmentation_folder = segmentation_folder
        # sort segmentation files by the number in the file name
        self.segmentation_files = sorted(self.segmentation_files, key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))

        # load output folder
        self.output_folder = output_folder

        self.distortion_params = None
        self.matrix_intrinsics = None

    def load_camera_parameters(self, camera_file_path):
        """
        Load camera parameters from Agisoft Metashape. 
        """
        cameras = read_camera_parameters_agisoft(camera_file_path)
        self.distortion_params = cameras['distortion_params']
        self.matrix_intrinsics = cameras['K']

    # def extract_segmentation(self, save_overlap=False, add_background=False):
    #     """
    #     Extracts polygon points from a JSON file and creates binary masks for each polygon.
        
    #     Args:
    #         save_overlap (bool): If True, saves an overlap image of the masks on the original image.
    #         add_background (bool): If True, adds a background class with ID 0.
        
    #     Returns:
    #         None
    #     """
    #     if self.distortion_params is None or self.matrix_intrinsics is None:
    #         raise ValueError("Camera parameters not loaded. Call load_camera_parameters() first.")

    #     for json_file in self.segmentation_files:
    #         json_file_path = os.path.join(self.segmentation_folder, json_file)
    #         print(f"Processing {json_file}")

    #         # Load JSON file
    #         with open(json_file_path, 'r') as file:
    #             data = json.load(file)
            
    #         # Extract polygons
    #         shapes = data.get("shapes", [])
    #         masks = []
            
    #         for shape in shapes:
    #             points = shape.get("points", [])
    #             if points:
    #                 # Create a blank mask
    #                 mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
    #                 polygon = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    #                 cv2.fillPoly(mask, [polygon], 1)  # Use 1 for binary mask
    #                 # undistort the mask
    #                 mask = cv2.undistort(mask, self.matrix_intrinsics, self.distortion_params)
    #                 masks.append({'segmentation': mask, 'area': np.sum(mask)})
            
    #         if len(masks) == 0:
    #             continue

    #         # Sort masks by area in descending order
    #         sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)

    #         # Initialize an image with -1 (background)
    #         processed_img = -np.ones((self.image_height, self.image_width), dtype=np.int16)
    #         for idx, mask_info in enumerate(sorted_masks):
    #             mask = mask_info['segmentation']
    #             processed_img[mask > 0] = idx  # Assign unique ID to each mask

    #         # Save the processed image as a NumPy array
    #         save_path = os.path.join(output_folder, json_file.replace('.json', '.npy'))
    #         if add_background:
    #             processed_img += 1  # Add 1 to all IDs to make background 0
    #         else:
    #             np.save(save_path, processed_img)

    #         # Save the overlap image
    #         if save_overlap:
    #             reshape_width = 400
    #             reshape_height = int(reshape_width * self.image_height / self.image_width)
    #             # get the name before the extension
    #             keyframe = json_file.split('.')[0]
    #             # use keyframe to find the image name in the image folder
    #             image_path = [f for f in self.image_files if keyframe in f][0]
    #             image = cv2.imread(os.path.join(self.image_folder, image_path))
    #             # undistort the image
    #             image = cv2.undistort(image, self.matrix_intrinsics, self.distortion_params)
    #             image = cv2.resize(image, (reshape_width, reshape_height))

    #             img = np.ones((reshape_height, reshape_width, 3))
    #             for mask in sorted_masks:
    #                 m = mask['segmentation']
    #                 m = cv2.resize(m, (reshape_width, reshape_height), interpolation=cv2.INTER_NEAREST) > 0
    #                 color_mask = np.concatenate([np.random.random(3)])
    #                 img[m] = color_mask

    #             img = (img * 255).astype(np.uint8)
    #             img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #             img = cv2.addWeighted(img, 0.35, image, 0.65, 0)
    #             save_overlap_path = os.path.join(output_folder, json_file.replace('.json', '_overlap.png'))
    #             cv2.imwrite(save_overlap_path, img)

    def process_single_file(self, json_file, save_overlap, add_background):
        """
        Process a single JSON file to extract polygon points, create binary masks,
        (optionally) save an overlap image, and write out the processed data.
        """
        if self.distortion_params is None or self.matrix_intrinsics is None:
            raise ValueError("Camera parameters not loaded. Call load_camera_parameters() first.")

        json_file_path = os.path.join(self.segmentation_folder, json_file)

        # Load JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        # Extract polygons
        shapes = data.get("shapes", [])
        masks = []

        for shape in shapes:
            points = shape.get("points", [])
            if points:
                # Create a blank mask
                mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
                polygon = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [polygon], 1)  # Use 1 for binary mask
                # Undistort the mask
                mask = cv2.undistort(mask, self.matrix_intrinsics, self.distortion_params)
                masks.append({'segmentation': mask, 'area': np.sum(mask)})

        if len(masks) == 0:
            return  # No polygons to process

        # Sort masks by area (descending order)
        sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)

        # Prepare array for saving
        processed_img = -np.ones((self.image_height, self.image_width), dtype=np.int16)
        for idx, mask_info in enumerate(sorted_masks):
            mask = mask_info['segmentation']
            processed_img[mask > 0] = idx  # Assign a unique ID to each mask

        # Optionally add background as class 0
        if add_background:
            processed_img += 1

        # Save the processed image as a NumPy array
        save_path = os.path.join(self.output_folder, json_file.replace('.json', '.npy'))
        np.save(save_path, processed_img)

        # Optionally save overlap visualization
        if save_overlap:
            reshape_width = 400
            reshape_height = int(reshape_width * self.image_height / self.image_width)

            # Determine image file name from keyframe
            keyframe = json_file.split('.')[0]
            image_path = [f for f in self.image_files if keyframe in f][0]

            image = cv2.imread(os.path.join(self.image_folder, image_path))
            # Undistort the image
            image = cv2.undistort(image, self.matrix_intrinsics, self.distortion_params)
            image = cv2.resize(image, (reshape_width, reshape_height))

            # Initialize a blank overlay image
            img = np.ones((reshape_height, reshape_width, 3))

            for mask_dict in sorted_masks:
                m = mask_dict['segmentation']
                # Resize the mask and convert to boolean
                m = cv2.resize(m, (reshape_width, reshape_height), interpolation=cv2.INTER_NEAREST) > 0
                color_mask = np.random.rand(3)  # Random color in [0,1]
                img[m] = color_mask

            # Convert to uint8
            img = (img * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # Blend the mask overlay with the original image
            img = cv2.addWeighted(img, 0.35, image, 0.65, 0)

            # Save the overlap visualization
            save_overlap_path = os.path.join(self.output_folder, json_file.replace('.json', '_overlap.png'))
            cv2.imwrite(save_overlap_path, img)

    def extract_segmentation(self, save_overlap=False, add_background=False, n_jobs=8):
        """
        Uses parallel processing with a tqdm progress bar to handle all
        segmentation files in self.segmentation_files.
        """
        self._lock = Lock()

        # Initialize a progress bar
        pbar = tqdm(total=len(self.segmentation_files), desc="Processing Segmentation Files")

        def process_and_update(json_file):
            # Actual processing
            self.process_single_file(json_file, save_overlap, add_background)
            # Update progress bar inside a lock to avoid overlapping writes
            with self._lock:
                pbar.update(1)

        # Use 'threading' backend so we don't directly invoke multiprocessing
        Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(process_and_update)(json_file) for json_file in self.segmentation_files
        )

        # Close the progress bar
        pbar.close()

            


if __name__ == "__main__":
    image_folder = "../../data/granite_dells/DJI_photos"
    segmentation_folder = "../../data/granite_dells/DJI_photos"
    output_folder = "../../data/granite_dells/segmentations_classes"
    camera_file_path = "../../data/granite_dells/SfM_products/cameras.xml"

    segmentation_extraction = SegmentationExtraction(image_folder, segmentation_folder, output_folder)
    segmentation_extraction.load_camera_parameters(camera_file_path)
    segmentation_extraction.extract_segmentation(save_overlap=True, add_background=True)