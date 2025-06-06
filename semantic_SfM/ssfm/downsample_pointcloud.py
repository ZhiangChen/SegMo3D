import laspy
import numpy as np
import open3d as o3d
import os
from ssfm.keyimage_associations_builder import *

def downsample_pointcloud(input_file, output_file, method="uniform", every_k_points=2):
    """
    Downsample a point cloud

    Parameters
    ----------
    input_file : str
        Path to the input point cloud file
    output_file : str
        Path to the output point cloud file
    method : str
        Method to downsample the point cloud. Options: "uniform"
    every_k_points : int
        The number of points to keep after downsampling. Default is 2, which means that every 2nd point will be kept. 
    """
    # assert if input file exists
    assert os.path.exists(input_file), "Input file does not exist"
    # load the point cloud .las file
    try:
        pc = laspy.read(input_file)
    except:
        raise Exception("Error reading the input file")
        return

    # assert if output file is a .las file
    assert output_file.endswith(".las"), "Output file must be a .las file"

    # get the point cloud data
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
    

    if method == "uniform":
        indices = np.arange(0, points.shape[0], every_k_points)
        downsample_points = points[indices]
        downsample_colors = colors[indices]

    else:
        raise Exception("Invalid method")


    # Create a LasData object
    hdr = laspy.LasHeader(version="1.2", point_format=3)
    hdr.offset = pc.header.offset
    hdr.scale = pc.header.scale
    out_las = laspy.LasData(hdr)
    out_las.x = downsample_points[:, 0]
    out_las.y = downsample_points[:, 1]
    out_las.z = downsample_points[:, 2]
    out_las.red = downsample_colors[:, 0]
    out_las.green = downsample_colors[:, 1]
    out_las.blue = downsample_colors[:, 2]
    out_las.write(output_file)

    print(f"Downsampled point cloud saved to {output_file}")


def downsample_pointcloud_using_semantics(input_pc_file, output_pc_file, image_list, associations_folder_path, segmentations_folder_path, keep_ratio=0.2):
    # assert if input file exists
    assert os.path.exists(input_pc_file), "Input file does not exist"
    # load the point cloud .las file
    try:
        pc = laspy.read(input_pc_file)
    except:
        raise Exception("Error reading the input file")
        return

    # assert if output file is a .las file
    assert output_pc_file.endswith(".las"), "Output file must be a .las file"

    # get the point cloud data
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

    # keyimage_associations_builder = KeyimageAssociationsBuilder(image_list, associations_folder_path, segmentations_folder_path)
    # keyimage_associations = keyimage_associations_builder.build_associations_downsampling_background()
    # np.save("tmp.npy", keyimage_associations)

    keyimage_associations = np.load("tmp.npy")

    # apply or operation to the keyimage associations along the second axis
    keyimage_associations = np.sum(keyimage_associations, axis=1) > 0
    
    keep_indices = np.where(keyimage_associations)[0]
    remove_indices = np.where(~keyimage_associations)[0]
    
    downsampled_indices = np.random.choice(remove_indices, size=int(len(remove_indices) * keep_ratio), replace=False)
    downsampled_indices = np.concatenate((downsampled_indices, keep_indices))
    downsampled_indices = np.sort(downsampled_indices)
    downsampled_points = points[downsampled_indices]
    downsampled_colors = colors[downsampled_indices]
    # Create a LasData object
    hdr = laspy.LasHeader(version="1.2", point_format=3)
    hdr.offset = pc.header.offset
    hdr.scale = pc.header.scale
    out_las = laspy.LasData(hdr)
    out_las.x = downsampled_points[:, 0]
    out_las.y = downsampled_points[:, 1]
    out_las.z = downsampled_points[:, 2]
    out_las.red = downsampled_colors[:, 0]
    out_las.green = downsampled_colors[:, 1]
    out_las.blue = downsampled_colors[:, 2]
    out_las.write(output_pc_file)
    print(f"Downsampled point cloud saved to {output_pc_file}")





if __name__ == "__main__":
    # input_file = "../../data/granite_dells/SfM_products/granite_dells_wgs_utm.las"
    # output_file = "../../data/granite_dells/granite_dells_wgs_utm_downsampled_0.las"
    # downsample_pointcloud(input_file, output_file, method="uniform", every_k_points=3)

    input_pc_file = "../../data/centennial_bluff/mission_a/SfM_products/a_downsampled_1.las"
    output_pc_file = "../../data/centennial_bluff/mission_a/SfM_products/a_downsampled_1_semantics.las"
    associations_folder_path = "../../data/centennial_bluff/mission_a/associations_1"
    segmentations_folder_path = "../../data/centennial_bluff/mission_a/segmentations_class_filter"
    photos_folder_path = "../../data/centennial_bluff/mission_a/DJI_photos"
    image_list = [image for image in os.listdir(photos_folder_path)]
    # sort images based on the values of keyimages in file names
    image_list = sorted(image_list, key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
    downsample_pointcloud_using_semantics(input_pc_file, output_pc_file, image_list, associations_folder_path, segmentations_folder_path, keep_ratio=0.2)