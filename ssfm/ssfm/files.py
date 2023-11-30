import laspy
import os
import numpy as np
import json
import xml.etree.ElementTree as ET


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

    return camera_intrinsics 

def read_camera_extrinsics_webodm(file_path):
    # check if the file exists
    assert os.path.exists(file_path)

    # Open the geojson file
    with open(file_path, "r") as f:
        data = json.load(f)

    # Get the camera extrinsics
    features = data["features"]
    
    camera_list = []
    for camera in features:
        camera_list.append(camera["properties"])

    return camera_list

def read_camera_intrinsics_agisoft(file_path):
    # read camera intrinsics from agisoft xml file
    # check if the file exists
    assert os.path.exists(file_path)

    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Find the calibration element and extract the parameters
    calibration = root.find('.//calibration')

    f = float(calibration.find('f').text)
    cx = float(calibration.find('cx').text)
    cy = float(calibration.find('cy').text)
    k1 = float(calibration.find('k1').text)
    k2 = float(calibration.find('k2').text)
    k3 = float(calibration.find('k3').text)
    p1 = float(calibration.find('p1').text)
    p2 = float(calibration.find('p2').text)

    # Create the camera intrinsics matrix
    camera_intrinsics = np.array([[f, 0, cx],
                                  [0, f, cy],
                                  [0, 0, 1]])
    
    # Create the distortion coefficients
    distortion_coefficients = np.array([k1, k2, p1, p2, k3])

    return camera_intrinsics, distortion_coefficients

def read_camera_extrinsics_agisoft(file_path):
    # read camera extrinsics from agisoft xml file
    # check if the file exists
    assert os.path.exists(file_path)

    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    cameras = dict()

    image_dimensions = root.find('.//ImageDimensions')
    width = int(image_dimensions.find('Width').text)
    height = int(image_dimensions.find('Height').text)

    cameras['width'] = width
    cameras['height'] = height

    # Iterate over each photo and extract extrinsic matrices
    for photo in tree.findall('.//Photo'):
        image_path = photo.find('ImagePath').text
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
    hdr.scale = [0.01, 0.01, 0.01]  # Example scale factor, adjust as needed
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
    las_file = "../../data/model.las"
    #points, colors = read_las_file(las_file)

    #camera_intrinsics_file = "../../data/camera.json"
    #camera_intrinsics = read_camera_intrinsics_webodm(camera_intrinsics_file)

    #camera_list_file = "../../data/shots.geojson"
    #camera_list = read_camera_extrinsics_webodm(camera_list_file)

    #write_las(points, colors, "test.las")

    camera_intrinsics_file = "../../data/box_canyon_export/camera_intrinsics.xml"
    camera_intrinsics, distortion_coefficients = read_camera_intrinsics_agisoft(camera_intrinsics_file)
    print(camera_intrinsics, distortion_coefficients)

    cameras = read_camera_extrinsics_agisoft("../../data/box_canyon_export/camera_extrinsics.xml")

    print(cameras)