import laspy
import os
import numpy as np
import json
import xml.etree.ElementTree as ET
import cv2



def read_mesh_file(file_path):
    assert os.path.exists(file_path)

    mesh_vertices_color = np.load(file_path)
    vertices = mesh_vertices_color[0]
    colors = mesh_vertices_color[1]

    return vertices, colors

def read_camera_scannet(ssfm_scene_folder_path):
    assert os.path.exists(ssfm_scene_folder_path), 'Scene folder does not exist!'
    camera = dict()
    intrinsic_color_depth_file_path = os.path.join(ssfm_scene_folder_path, 'reconstructions/intrinsic_color_depth.npy')
    intrinsic_color_depth = np.load(intrinsic_color_depth_file_path)
    intrinsic_color = intrinsic_color_depth[0]
    intrinsic_depth = intrinsic_color_depth[1]

    camera['intrinsic_color'] = intrinsic_color
    camera['intrinsic_depth'] = intrinsic_depth

    camera_poses_file_path = os.path.join(ssfm_scene_folder_path, 'reconstructions/camera_poses.npy')
    camera_poses = np.load(camera_poses_file_path, allow_pickle=True).item()
    
    for key, value in camera_poses.items():
        camera[key] = value

    # get camera height and width
    photo_folder_path = os.path.join(ssfm_scene_folder_path, 'photos')
    photo_file_path = os.path.join(photo_folder_path, os.listdir(photo_folder_path)[0])
    photo = cv2.imread(photo_file_path)
    camera['width'] = photo.shape[1]
    camera['height'] = photo.shape[0]

    return camera


def read_las_file(file_path):
    # check if the file exists
    assert os.path.exists(file_path)

    # Open the file in read mode
    pc = laspy.read(file_path)

    # Get the points
    x = pc.x.scaled_array()
    y = pc.y.scaled_array()
    z = pc.z.scaled_array()
    r = pc.red
    g = pc.green
    b = pc.blue

    # Stack points: (N, 3), where N is the number of points 
    points = np.vstack((x, y, z)).T

    # Stack colors: (N, 3), where N is the number of points
    colors = np.vstack((r, g, b)).T

    return points, colors


def read_camera_intrinsics_webodm(file_path):
    # check if the file exists
    assert os.path.exists(file_path)

    # Open the json file
    with open(file_path, "r") as f:
        data = json.load(f)
    
    # Get the camera intrinsics
    camera_name = list(data.keys())[0]
    camera_intrinsics = data[camera_name]

    focal_x = camera_intrinsics["focal_x"]
    focal_y = camera_intrinsics["focal_y"]
    c_x = camera_intrinsics["c_x"]
    c_y = camera_intrinsics["c_y"]
    width = camera_intrinsics["width"]
    height = camera_intrinsics["height"]

    # Get the camera intrinsics
    f_x = f_y = focal_x * width 
    c_x = c_x * width + width/2
    c_y = c_y * height + height/2

    intrinsic_matrix = np.array([
        [f_x, 0, c_x],
        [0, f_y, c_y],
        [0, 0, 1]
    ])

    k1 = camera_intrinsics["k1"]
    k2 = camera_intrinsics["k2"]
    p1 = camera_intrinsics["p1"]
    p2 = camera_intrinsics["p2"]
    k3 = camera_intrinsics["k3"]

    distortion_params = np.array([k1, k2, p1, p2, k3])

    return intrinsic_matrix, distortion_params

def read_camera_extrinsics_webodm(file_path):
    # check if the file exists
    assert os.path.exists(file_path)

    # Open the geojson file
    with open(file_path, "r") as f:
        data = json.load(f)

    # Get the camera extrinsics
    features = data["features"]
    
    cameras = dict()

    width = features[0]["properties"]["width"]
    height = features[0]["properties"]["height"]

    cameras["width"] = width
    cameras["height"] = height

    for feature in features:
        camera = feature["properties"]
        # Get the extrinsics
        translation_vector = np.array(camera["translation"]).reshape(-1, 1)
        rotation_vector = camera["rotation"]

        # Create rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(np.array(rotation_vector))

        # Combine into a 4x4 extrinsic matrix
        extrinsic_matrix = np.hstack((rotation_matrix.T, translation_vector))
        extrinsic_matrix = np.vstack((extrinsic_matrix, [0, 0, 0, 1]))
        cameras[camera["filename"]] = extrinsic_matrix

    return cameras


def read_camera_parameters_webodm(intrinsic_file_path, extrinsic_file_path):
    intrinsic_matrix, distortion_params = read_camera_intrinsics_webodm(intrinsic_file_path)
    cameras = read_camera_extrinsics_webodm(extrinsic_file_path)
    cameras["K"] = intrinsic_matrix
    cameras["distortion_params"] = distortion_params
    return cameras


def read_camera_parameters_agisoft(file_path):
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
    sensor_size_mm = root.find('.//SensorSize').text
    focal_length_mm = root.find('.//FocalLength').text

    # Convert Focal Length from mm to pixels if needed
    f_x = f_y = width * float(focal_length_mm) / float(sensor_size_mm)

    cx = float(root.find('.//PrincipalPoint/x').text)
    cy = float(root.find('.//PrincipalPoint/y').text)

    # Intrinsic Matrix
    K = [[f_x, 0, cx],
        [0, f_y, cy],
        [0, 0, 1]]
    
    cameras['K'] = np.asarray(K)

    # Extract Distortion Parameters
    k1 = float(root.find('.//Distortion/K1').text)
    k2 = float(root.find('.//Distortion/K2').text)
    k3 = float(root.find('.//Distortion/K3').text)
    p1 = float(root.find('.//Distortion/P1').text)
    p2 = float(root.find('.//Distortion/P2').text)

    distortion_params = np.array([k1, k2, p1, p2, k3])
    
    cameras['distortion_params'] = distortion_params

    # Iterate over each photo and extract extrinsic matrices
    for photo in tree.findall('.//Photo'):
        image_path = photo.find('ImagePath').text
        image_path = os.path.basename(image_path)
        pose = photo.find('Pose')

        # Extract rotation matrix and translation vector
        rotation_matrix = []
        translation_vector = []

        # Extract rotation matrix
        rotation = []
        for i in range(3):
            row = []
            for j in range(3):
                element = pose.find(f'Rotation/M_{i}{j}')
                if element is not None:
                    row.append(float(element.text))
            rotation.append(row)
        
        rotation_matrix.append(rotation)
        
        # Extract translation vector
        translation = []
        for axis in ['x', 'y', 'z']:
            element = pose.find(f'Center/{axis}')
            if element is not None:
                translation.append(float(element.text))

        translation.append(1)

        translation_vector.append(translation)

        # Convert to numpy arrays
        rotation_matrix = np.array(rotation_matrix)
        translation_vector = np.array(translation_vector)

        # compose a transformation matrix
        transformation_matrix = np.zeros((4, 4))    
        transformation_matrix[:3, :3] = rotation_matrix.T.reshape(3, 3)
        transformation_matrix[:4, 3] = translation_vector

        cameras[image_path] = transformation_matrix

    return cameras



def write_las(points, colors, filename):
    """
    Write points and colors to a LAS file.

    Parameters:
    points (np.array): Nx3 numpy array with point coordinates.
    colors (np.array): Nx3 numpy array with RGB color values (0-255).
    filename (str): Path of the file to be written.
    """
    # Create a new LAS file
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

    # Write the LAS file
    las.write(filename)

    return
    


if __name__ == "__main__":
    """
    las_file = "../../data/courtright/SfM_products/agisoft_model.las"
    points, colors = read_las_file(las_file)
    print(f"Read {len(points)} points and {len(colors)} colors from {las_file}")

    mesh_file = "../../data/scene0000_00/reconstructions/mesh_vertices_color.npy"
    vertices, colors = read_mesh_file(mesh_file)
    print(f"Read {len(vertices)} vertices and {len(colors)} colors from {mesh_file}")
    """
    camera = read_camera_scannet("../../data/scene0000_00")
    print(camera['height'])