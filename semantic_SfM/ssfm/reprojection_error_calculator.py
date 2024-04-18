import os
import numpy as np
import xml.etree.ElementTree as ET
from numba import jit
from joblib import Parallel, delayed
from tqdm import tqdm

from ssfm.files import read_camera_parameters_agisoft

class ReprojectionErrorCalculator(object):
    def __init__(self, camera_file_path):
        self.cameras = read_camera_parameters_agisoft(camera_file_path)

        tree = ET.parse(camera_file_path)
        root = tree.getroot()

        # read and create a dictionary of photo_id and its name
        self.photo_id_name = {}
        for photo in root.findall('.//Photo'):
            photo_id = int(photo.find('Id').text)
            photo_path = photo.find('ImagePath').text
            photo_name = os.path.basename(photo_path)
            self.photo_id_name[photo_id] = photo_name

        tie_points = []
        # read tie points 
        for tie_point in root.findall('.//TiePoint'):
            # Extract the 3D position
            pos_elem = tie_point.find('Position')
            position = [
                float(pos_elem.find('x').text),
                float(pos_elem.find('y').text),
                float(pos_elem.find('z').text)]

            # Extract measurements
            measurements = []
            for measurement in tie_point.findall('Measurement'):
                photo_id = int(measurement.find('PhotoId').text)
                x = float(measurement.find('x').text)
                y = float(measurement.find('y').text)
                measurements.append({photo_id: (x, y)})
                

            # Append the extracted data to the tie_points list
            tie_points.append({
                'Position': position,
                'Projection': measurements})

        # group tie points by photo id
        self.tie_points = {}
        for tie_point in tie_points:
            for measurement in tie_point['Projection']:
                photo_id = list(measurement.keys())[0]
                if photo_id not in self.tie_points:
                    self.tie_points[photo_id] = [[],[]]
                self.tie_points[photo_id][0].append(tie_point['Position'])
                self.tie_points[photo_id][1].append(measurement[photo_id])
    
    def calculate_reprojection_error(self, photo_id):
        frame_key = self.photo_id_name[photo_id]
        extrinsic_matrix = self.cameras[frame_key]
        camera_intrinsics = self.cameras['K'] 
        image_height = self.cameras['height']
        image_width = self.cameras['width']

        points = np.array(self.tie_points[photo_id][0])

        # Transform the point cloud using the extrinsic matrix
        points_homogeneous = np.hstack((points, np.ones((len(points), 1))))

        extrinsic_matrix_inv = np.linalg.inv(extrinsic_matrix)

        points_transformed = np.matmul(points_homogeneous, extrinsic_matrix_inv.T)

        # Project the points using the intrinsic matrix
        # Drop the homogeneous component (w)
        points_camera_space = points_transformed[:, :3]

        points_projected_d = np.matmul(points_camera_space, camera_intrinsics.T)
        points_projected = points_projected_d / points_projected_d[:, -1].reshape(-1, 1)
        # replace depth 
        points_projected[:, 2] = points_projected_d[:, 2]

        # Calculate the reprojection error
        original_pixel_coordinates = np.array(self.tie_points[photo_id][1])
        reprojected_pixel_coordinates = points_projected[:, :2]
        error = np.linalg.norm(original_pixel_coordinates - reprojected_pixel_coordinates, axis=1)

        max_error = np.max(error)
        mean_error = np.mean(error)

        return mean_error, max_error
    
    def calculate_reprojection_error_all(self, num_cores=16):
        results = Parallel(n_jobs=num_cores)(delayed(self.calculate_reprojection_error)(photo_id) for photo_id in tqdm(self.tie_points.keys()))
        photo_ids = list(self.tie_points.keys())
        file_names = [self.photo_id_name[photo_id] for photo_id in photo_ids]
        mean_errors = [result[0] for result in results]
        max_errors = [result[1] for result in results]

        # sort by mean errors
        sorted_indices = np.argsort(mean_errors)
        sorted_mean_errors = [mean_errors[i] for i in sorted_indices]
        sorted_mean_error_file_names = [file_names[i] for i in sorted_indices]

        # sort by max errors
        sorted_indices = np.argsort(max_errors)
        sorted_max_errors = [max_errors[i] for i in sorted_indices]
        sorted_max_error_file_names = [file_names[i] for i in sorted_indices]
        return sorted_mean_errors, sorted_mean_error_file_names, sorted_max_errors, sorted_max_error_file_names



if __name__ == "__main__":
    camera_file_path = "../../data/box_canyon_park/SfM_products/agisoft_cameras.xml"
    reprojection_error_calculator = ReprojectionErrorCalculator(camera_file_path)
    sorted_mean_errors, sorted_mean_error_file_names, sorted_max_errors, sorted_max_error_file_names = reprojection_error_calculator.calculate_reprojection_error_all()