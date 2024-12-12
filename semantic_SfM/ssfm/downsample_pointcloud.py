import laspy
import numpy as np
import open3d as o3d
import os

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


if __name__ == "__main__":
    input_file = "../../data/granite_dells/SfM_products/granite_dells_wgs_utm.las"
    output_file = "../../data/granite_dells/granite_dells_wgs_utm_downsampled_0.las"
    downsample_pointcloud(input_file, output_file, method="uniform", every_k_points=3)