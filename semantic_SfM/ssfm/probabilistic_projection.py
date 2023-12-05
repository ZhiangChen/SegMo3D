import sys
sys.path.append(".")
from files import *

import numpy as np
import cv2
import torch
import laspy
import open3d as o3d

def extract_extrinsic_matrix(camera):
    # Get the extrinsics
    translation_vector = np.array(camera["translation"]).reshape(-1, 1)
    rotation_vector = camera["rotation"]

    print(translation_vector)
    # Create rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(np.array(rotation_vector))

    # Combine into a 4x4 extrinsic matrix
    extrinsic_matrix = np.hstack((rotation_matrix, translation_vector))
    extrinsic_matrix = np.vstack((extrinsic_matrix, [0, 0, 0, 1]))
    return extrinsic_matrix

def extract_intrinsic_matrix(camera_intrinsics):
    focal_x = camera_intrinsics["focal_x"]
    focal_y = camera_intrinsics["focal_y"]
    c_x = camera_intrinsics["c_x"]
    c_y = camera_intrinsics["c_y"]
    width = camera_intrinsics["width"]
    height = camera_intrinsics["height"]

    # Get the camera intrinsics
    f_x = focal_x * width
    f_y = focal_y * height
    c_x = c_x * width 
    c_y = c_y * height 

    intrinsic_matrix = np.array([
        [f_x, 0, c_x],
        [0, f_y, c_y],
        [0, 0, 1]
    ])

    return intrinsic_matrix

def project_point_cloud_with_occlusion(points, colors, intrinsic_matrix, image_width, image_height, extrinsic_matrix):
    """
    Projects a point cloud onto an image plane, taking into account occlusion. Also, generates pixel-to-point associations.

    :param points: (N, 3) array of 3D points
    :param colors: (N, 3) array of RGB colors
    :param intrinsic_matrix: (3, 3) camera intrinsic matrix
    :param image_width: width of the image
    :param image_height: height of the image
    :param extrinsic_matrix: (4, 4) camera extrinsic matrix
    :return: (associations, image)
        associations: dictionary of pixel coordinates to 3D points
        image: (image_height, image_width, 3) array of RGB values
    """

    # Transform the point cloud using the extrinsic matrix
    points_homogeneous = np.hstack((points, np.ones((len(points), 1))))

    extrinsic_matrix_inv = np.linalg.inv(extrinsic_matrix)

    print(extrinsic_matrix)
    points_transformed = np.matmul(points_homogeneous, extrinsic_matrix_inv.T)



    # Project the points using the intrinsic matrix
    # Drop the homogeneous component (w)
    points_camera_space = points_transformed[:, :3]

    # Project the points using the intrinsic matrix
    points_projected = np.matmul(points_camera_space, intrinsic_matrix.T)

    print(points_projected)
    points_projected /= points_projected[:, -1].reshape(-1, 1)
    print(points_projected)
    print(colors)

    #write_las(points_camera_space, colors, "test.las")

    # Create a blank image - Black image
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    

    # Iterate through the points and draw them on the image
    for point, color in zip(points_projected, colors):
        x, y = int(point[0]), int(point[1])
        
        # Check if the point is within the image bounds
        if 0 <= x < image_width and 0 <= y < image_height:
            # Draw the point on the image
            image[y, x] = color

    return image


  


if __name__ == "__main__":
    
    las_file = "../../data/model_0.las"
    points, colors = read_las_file(las_file)

    
    """
    camera_intrinsics_file = "../../data/camera_0.json"
    camera_intrinsics = read_camera_intrinsics_webodm(camera_intrinsics_file)
    intrinsic_matrix = extract_intrinsic_matrix(camera_intrinsics)

    camera_list_file = "../../data/shots_0.geojson"
    camera_list = read_camera_extrinsics_webodm(camera_list_file)

    extrinsic_matrix = extract_extrinsic_matrix(camera_list[5])
    image_height = camera_intrinsics["height"]
    image_width = camera_intrinsics["width"]

    print(camera_list[5])

    print(extrinsic_matrix)

    image = project_point_cloud_with_occlusion(points, colors, intrinsic_matrix, image_width, image_height, extrinsic_matrix)

    cv2.imwrite("test.png", image)
    """
    



