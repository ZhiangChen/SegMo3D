import numpy as np
import os
from ssfm.files import *
from prettytable import PrettyTable

def memory_calculator(pointcloud_file, image_file, num_images, num_segmentation_ids):
    # assert if the pointcloud file exists
    assert os.path.exists(pointcloud_file), "Pointcloud file does not exist"
    # assert if the image file exists
    assert os.path.exists(image_file), "Image file does not exist"

    # read the pointcloud file
    points, colors = read_las_file(pointcloud_file)
    # get the number of points
    num_points = points.shape[0]

    # read the image file
    image = cv2.imread(image_file)
    # get the image shape
    H, W = image.shape[:2]

    # calculate the memory required for the segmentation for each image; the array has the dtype np.int16
    image_segmentation_memory = H * W * 2 / 1024 / 1024 / 1024 # in GB
    #print("Memory required for segmentation for each image: ", image_segmentation_memory, "GB")
    # calculate the memory required for the segmentation for all images
    total_segmentation_memory = num_images * image_segmentation_memory 
    #print("Memory required for segmentation for all images: ", total_segmentation_memory, "GB")

    # calculate the memory required for pixel2point association; each pixel has the dtype np.int32
    pixel2point_memory = H * W * 4 / 1024 / 1024 / 1024 # in GB
    #print("Memory required for pixel2point association: ", pixel2point_memory, "GB")
    # calculate the memory required for pixel2point association for all images
    total_pixel2point_memory = num_images * pixel2point_memory 
    #print("Memory required for pixel2point association for all images: ", total_pixel2point_memory, "GB")

    # calculate the memory required for point2pixel association; each point has the dtype np.int16
    point2pixel_memory = num_points * 2* 2 / 1024 / 1024 / 1024 # in GB
    #print("Memory required for point2pixel association: ", point2pixel_memory, "GB")
    # calculate the memory required for point2pixel association for all images
    total_point2pixel_memory = num_images * point2pixel_memory 
    #print("Memory required for point2pixel association for all images: ", total_point2pixel_memory, "GB")

    # calculate the memory required for pc_segmentation_ids; each point has the dtype np.int32
    pc_segmentation_ids_memory = num_points * num_segmentation_ids * 4 / 1024 / 1024 / 1024 # in GB
    #print("Memory required for pc_segmentation_ids: ", pc_segmentation_ids_memory, "GB")

    # calculate the memory required for pc_segmentation_probs; each point has the dtype np.float32
    pc_segmentation_probs_memory = num_points * num_segmentation_ids * 4 / 1024 / 1024 / 1024 # in GB
    #print("Memory required for pc_segmentation_probs: ", pc_segmentation_probs_memory, "GB")

    # calculate the memory required for keyimage_association which is a boolean 2D array with the shape (num_points, num_images); numpy boolean arrays have size 1 byte per element
    keyimage_association_memory = num_points * num_images / 1024 / 1024 / 1024 # in GB
    #print("Memory required for keyimage_association: ", keyimage_association_memory, "GB")

    # calculate the total memory required
    total_memory = total_segmentation_memory + total_pixel2point_memory + total_point2pixel_memory + pc_segmentation_ids_memory + pc_segmentation_probs_memory + keyimage_association_memory
    #print("Total memory required: ", total_memory, "GB")

    # pretty print the memory required as pretty table
    x = PrettyTable()
    x.field_names = ["Memory Type", "Memory Required (GB)"]
    x.add_row(["Segmentation for each image", image_segmentation_memory])
    x.add_row(["Pixel2point association for each image", pixel2point_memory])
    x.add_row(["Point2pixel association for each image", point2pixel_memory])
    # add a horizontal line
    x.add_row(["", ""])

    x.add_row(["Segmentation for all images", total_segmentation_memory])
    x.add_row(["Pixel2point association for all images", total_pixel2point_memory])
    x.add_row(["Point2pixel association for all images", total_point2pixel_memory])
    x.add_row(["pc_segmentation_ids", pc_segmentation_ids_memory])
    x.add_row(["pc_segmentation_probs", pc_segmentation_probs_memory])
    x.add_row(["keyimage_association", keyimage_association_memory])
    x.add_row(["Total", total_memory])
    print(x)


if __name__ == "__main__":
    # pointcloud file
    las_file = "../../data/model.las"
    # image file
    image_file = "../../data/mission_2/DJI_0183.JPG"
    # number of images
    num_images = 192
    # number of segmentation ids
    num_segmentation_ids = 5

    memory_calculator(las_file, image_file, num_images, num_segmentation_ids)