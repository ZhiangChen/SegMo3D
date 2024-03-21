import open3d as o3d
import numpy as np
import os
from ssfm.files import *

class PC_translation(object):
    def __init__(self, pc_file_path, utm_file_path=None):
        # assert if pc file exists
        assert os.path.exists(pc_file_path), "PC file does not exist"
        self.pc_file_path = pc_file_path


        self.utm = np.array([0, 0, 0])
        if utm_file_path is not None:
            # assert if the utm file exists
            assert os.path.exists(utm_file_path), "UTM file does not exist"
            self.utm_file_path = utm_file_path

            with open(utm_file_path, 'r') as file:
                # Skip the first line
                next(file)
                # Read the second line and convert to float
                second_line = next(file).strip().split()
                self.utm = np.array([float(second_line[0]), float(second_line[1]), 0])


    def positive_translation(self, save_pc_file_path):
        points, colors = read_las_file(self.pc_file_path)
        points = points + self.utm
        write_las(save_pc_file_path, points, colors)

    def negative_translation(self, save_pc_file_path):
        points, colors = read_las_file(self.pc_file_path)
        points = points - self.utm
        write_las(save_pc_file_path, points, colors)



if __name__ == "__main__":
    original_pc_file_path = "../../../Documents/KOA-Holiday-10-24-2023-georeferenced_model.las"
    utm_file_path = "../../data/mission_b/odm_georeferencing/coords.txt"
    save_pc_file_path = "translated_pc.las"
    

