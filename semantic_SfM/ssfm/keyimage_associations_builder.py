import numpy as np
import os
import yaml
from tqdm import tqdm
from joblib import Parallel, delayed
import networkx as nx
from scipy.sparse import csr_matrix
import time
from numba import jit
import torch
import json
from ssfm.files import *

@jit(nopython=True)
def calculate_edges(points, associations_keyimage_slice, i):
    edges = []
    links = np.zeros(i)  # Preallocate links array
    
    # Calculate links without broadcasting
    for j in range(i):
        link_sum = 0
        for k in range(points.shape[0]):  # Iterate over rows
            link_sum += points[k] * associations_keyimage_slice[k, j]
        links[j] = link_sum

    # Add edges based on link weights
    for j in range(i):
        if links[j] > 0:
            edges.append((i, j, links[j]))
    
    return edges

class KeyimageAssociationsBuilder(object):
    def __init__(self, image_list, read_folder_path, segmentation_folder_path):
        """
        Arguments:
            read_folder_path (str): The path to the folder containing the point2pixel files and pixel2point files.
            segmentation_folder_path (str): The path to the folder containing the segmentation files.
        """
        self.read_folder_path = read_folder_path
        self.segmentation_folder_path = segmentation_folder_path

        # get keyimage_list where each element comes from image_list with '.jpg' or '.png' replaced by '.npy'
        keyimage_list = [f.replace('.jpg', '.npy').replace('.png', '.npy').replace('.JPG', '.npy') for f in image_list]
        self.keyimage_list = keyimage_list


        pixel2point_folder_path = os.path.join(read_folder_path, 'pixel2point')
        assert os.path.exists(pixel2point_folder_path), 'Pixel2point folder path does not exist.'
        self.pixel2point_file_paths = [os.path.join(pixel2point_folder_path, f) for f in keyimage_list if f in os.listdir(pixel2point_folder_path)]
        self.N_images = len(self.pixel2point_file_paths)
        
        # get the number of points
        point2pixel_folder_path = os.path.join(read_folder_path, 'point2pixel')  
        assert os.path.exists(point2pixel_folder_path), 'Point2pixel folder path does not exist.'  
        self.point2pixel_file_paths = [os.path.join(point2pixel_folder_path, f) for f in keyimage_list if f in os.listdir(point2pixel_folder_path)]
        point2pixel = np.load(self.point2pixel_file_paths[0])
        self.N_points = point2pixel.shape[0]

        assert os.path.exists(segmentation_folder_path), 'Segmentation folder path does not exist.'
        self.segmentation_file_paths = [os.path.join(segmentation_folder_path, f) for f in keyimage_list if f in os.listdir(segmentation_folder_path)]


        # check if the number of images is the same
        assert len(self.segmentation_file_paths) == self.N_images, 'The number of images is not the same as the number of segmentation files.'
        self.segmentation_pixel2point_pairs = []
        for i in range(len(self.segmentation_file_paths)):
            segmentation_file_path = self.segmentation_file_paths[i]
            pixel2point_file_path = self.pixel2point_file_paths[i]
            # check if the base of the file name is the same
            assert os.path.basename(segmentation_file_path)[:-4] == os.path.basename(pixel2point_file_path)[:-4], 'The base of the file name is not the same.'
            self.segmentation_pixel2point_pairs.append((segmentation_file_path, pixel2point_file_path))

    def build_associations(self):
        # build associations
        self.associations_keyimage = np.full((self.N_points, self.N_images), False, dtype=bool)

        for k in tqdm(range(len(self.point2pixel_file_paths))):
            segmentation = np.load(self.segmentation_pixel2point_pairs[k][0])
            pixel2point = np.load(self.segmentation_pixel2point_pairs[k][1])
            pixel2point[segmentation == -1] = -1
            valid_point_ids = pixel2point[pixel2point != -1]
            self.associations_keyimage[valid_point_ids, k] = True

        # save associations
        save_file_path = os.path.join(self.read_folder_path, 'associations_keyimage.npy')
        np.save(save_file_path, self.associations_keyimage)

        # save image file names 
        image_file_names = [os.path.basename(f) for f in self.segmentation_file_paths]
        save_file_path = os.path.join(self.read_folder_path, 'keyimages.yaml')
        with open(save_file_path, 'w') as f:
            yaml.dump(image_file_names, f)


    def build_graph(self, num_chunks=0):
        """
        Build a graph from the associations_keyimage matrix

        Arguments:
            num_chunks (int): The number of chunks to split the associations_keyimage matrix into for parallel processing. 
            If 0, the CPU-based method will be used. Default is 0. Otherwise, the GPU-based method will be used. 
        """
        # Initialize the graph
        self.graph = nx.Graph()
        
        t2 = time.time()
        edges = []

        if num_chunks == 0:
            # CPU-based method
            for i in range(self.N_images):
                # Extract relevant points for image i
                points = self.associations_keyimage[:, i]
                extracted_points = self.associations_keyimage[points, :i]
                # Compute link weights
                links = np.sum(extracted_points, axis=0)
                
                # Add edges if the link weight is greater than 0
                edges.extend([(i, j, links[j]) for j in range(i) if links[j] > 0])

            t3 = time.time()
            print(f"Building edges on CPU took {t3 - t2} seconds.")

        else:
            # GPU-based method with chunking
            chunk_size = int(self.N_points / num_chunks)

            for chunk in range(num_chunks):
                # Calculate the start and end indices for this chunk
                start_idx = chunk * chunk_size
                end_idx = (chunk + 1) * chunk_size if chunk < num_chunks - 1 else self.N_points

                # Move current chunk of associations_keyimage to GPU
                associations_keyimage_chunk = torch.tensor(self.associations_keyimage[start_idx:end_idx, :]).cuda()

                # Process each image in the current chunk
                for i in range(self.N_images):
                    points = associations_keyimage_chunk[:, i]
                    extracted_points = associations_keyimage_chunk[points.bool(), :i]
                    links = torch.sum(extracted_points, dim=0).cpu().numpy()  # Move result back to CPU as NumPy array

                    # Add edges if the link weight is greater than 0
                    edges.extend([(i, j, links[j]) for j in range(i) if links[j] > 0])

            t3 = time.time()
            print(f"Building edges on GPU with {num_chunks} chunks took {t3 - t2} seconds.")

        # Combine weights for duplicate edges
        edges_dict = {}
        for edge in edges:
            key = (edge[0], edge[1])
            if key in edges_dict:
                edges_dict[key] += edge[2]
            else:
                edges_dict[key] = edge[2]

        # Add all edges to the graph at once
        self.graph.add_weighted_edges_from([(key[0], key[1], edges_dict[key]) for key in edges_dict])

        # Save the graph as a GraphML file
        save_file_path = os.path.join(self.read_folder_path, 'graph.graphml')
        nx.write_graphml(self.graph, save_file_path)

    def add_camera_to_graph(self, camera_file_path_list, camera_type="Agisoft"):
        """
        Add camera nodes to the graph

        Arguments:
            camera_file_path_list (list): A list of camera file paths.
            camera_type (str): The type of camera. Default is "Agisoft".
        """
        graph_file_path = os.path.join(self.read_folder_path, 'graph.graphml')
        assert os.path.exists(graph_file_path), 'Graph file does not exist.'
        self.graph = nx.read_graphml(graph_file_path)

        if camera_type == "Agisoft":
            assert len(camera_file_path_list) == 1, 'Only one camera file is required for Agisoft.'
            camera_file_path = camera_file_path_list[0]
            cameras =  read_camera_parameters_agisoft(camera_file_path)

        elif camera_type == "WebODM":
            assert len(camera_file_path_list) == 2, 'Two camera files are required for WebODM.'
            cameras = read_camera_parameters_webodm(camera_file_path_list[0], camera_file_path_list[1])

        elif camera_type == "Kubric":
            assert len(camera_file_path_list) == 1, 'Only one camera file is required for Kubric.'
            camera_file_path = camera_file_path_list[0]
            cameras = read_camera_parameters_kubric(camera_file_path)

        elif camera_type == "Scannet":
            assert len(camera_file_path_list) == 1, 'Only one camera file is required for Scannet.'
            camera_file_path = camera_file_path_list[0]
            cameras = read_camera_scannet(camera_file_path)

        else:
            raise ValueError("Camera type is not supported.")

        
        # find extension of the camera file
        for camera in cameras:
            if '.' in camera:
                extension = camera.split('.')[-1]
                break
            
        # Add camera positions to the graph nodes
        for i, keyframe in enumerate(self.keyimage_list):
            keyimage = keyframe.split('.')[0] + '.' + extension
            camera_transform = cameras[keyimage]
            camera_position = camera_transform[:3, 3].tolist()
            camera_position_str = json.dumps(camera_position)
            # set node position to the camera position
            if str(i) in self.graph.nodes:
                self.graph.nodes[str(i)]['pos'] = camera_position_str
            else:
                self.graph.add_node(str(i), pos=camera_position_str)
        
        # Save the graph as a GraphML file
        save_file_path = os.path.join(self.read_folder_path, 'graph_with_cameras.graphml')
        print(save_file_path)
        nx.write_graphml(self.graph, save_file_path)


        
        
    def read_associations(self, associations_keyimage_file_path=None):
        """
        Read the associations_keyimage file
        """
        if associations_keyimage_file_path is not None:
            self.associations_keyimage = np.load(associations_keyimage_file_path)
        else:
            associations_keyimage_file_path = os.path.join(self.read_folder_path, 'associations_keyimage.npy')
            assert os.path.exists(associations_keyimage_file_path), 'Associations keyimage file does not exist.'
            self.associations_keyimage = np.load(associations_keyimage_file_path)
        
        # print the associations are loaded
        print(f"Loaded associations keyimage file from {associations_keyimage_file_path}")

        # assert the number of images is the same
        assert self.associations_keyimage.shape[1] == len(self.segmentation_file_paths), 'The number of images is not the same as the number of segmentation files.'
    
    def find_min_cover(self):
        """
        Find the minimum cover of the associations_keyimage
        """
        # find the number of True values in each column of the associations_keyimage
        num_true_values = np.sum(self.associations_keyimage, axis=1)
        N_points = num_true_values.shape[0]
        # find the minimum number in the num_true_values
        min_num_true_values = np.min(num_true_values)
        # get the number of elements equal to 0 in the num_true_values
        num_zeros = np.sum(num_true_values == 0)
        # get the number of points covered by less than or equal to 1 image
        num_one = np.sum(num_true_values <= 1)
        # get the number of points covered by less than or equal to 3 images
        num_three = np.sum(num_true_values <= 3)
        # get the number of points covered by less than or equal to 5 images
        num_five = np.sum(num_true_values <= 5)

        # Prepare header and row format
        header = "| {0: <60} | {1: <10} | {2: <20} |"
        row = "| {0: <60} | {1: <10d} | {2: <20.2f} |"

        # Print table header
        print(header.format("Metric", "Count", "Percentage"))
        print("-" * 100)

        # Print table rows
        print(row.format("Number of points not covered by any image                 ", num_zeros, num_zeros / N_points * 100))
        print(row.format("Number of points covered by less than or equal to 1 image ", num_one, num_one / N_points * 100))
        print(row.format("Number of points covered by less than or equal to 3 images", num_three, num_three / N_points * 100))
        print(row.format("Number of points covered by less than or equal to 5 images", num_five, num_five / N_points * 100))


    def refine(self, remove_ratio=0.3):
        """
        Refine the associations_keyimage by removing the data with the smallest importance
        """
        N_image_per_point = np.sum(self.associations_keyimage, axis=1)
        _, N = self.associations_keyimage.shape

        importance_scarce_points_list = []
        for i in tqdm(range(N)):
            association = self.associations_keyimage[:, i]
            importance_scarce_points = 5-N_image_per_point[association & (N_image_per_point !=0)]
            importance_scarce_points = np.sum(importance_scarce_points[importance_scarce_points > 0])
            importance_scarce_points_list.append(importance_scarce_points)

        importance_scarce_points_list = np.array(importance_scarce_points_list)
        sorted_indices = np.argsort(importance_scarce_points_list)
        # remove the data with the smallest importance: self.segmentation_file_paths, self.associations_keyimage
        remove_indices = sorted_indices[:int(remove_ratio*N)]
        self.segmentation_file_paths = [self.segmentation_file_paths[i] for i in range(N) if i not in remove_indices]
        self.associations_keyimage = np.delete(self.associations_keyimage, remove_indices, axis=1)

        # save self.segmentation_file_paths and self.associations_keyimage
        image_file_names = [os.path.basename(f) for f in self.segmentation_file_paths]
        save_file_path = os.path.join(self.read_folder_path, 'refined_keyimages.yaml')
        with open(save_file_path, 'w') as f:
            yaml.dump(image_file_names, f)

        save_file_path = os.path.join(self.read_folder_path, 'refined_associations_keyimage.npy')
        np.save(save_file_path, self.associations_keyimage)

        


if __name__ == '__main__':
    scene_dir = '../../data/courtright'
    photo_folder_path = os.path.join(scene_dir, 'DJI_photos')
    image_list = [f for f in os.listdir(photo_folder_path) if f.endswith('.JPG')]
    # sort image list based on the values in the image file names
    image_list = sorted(image_list, key=lambda x: int(x.split('_')[1].split('.')[0]))
    smc_solver = KeyimageAssociationsBuilder(image_list, '../../data/courtright/associations', '../../data/courtright/segmentations')
    smc_solver.build_associations()
    #smc_solver.read_associations('../../data/courtright/associations/associations_keyimage.npy')
    #smc_solver.find_min_cover()
    #smc_solver.refine(0.5)
    smc_solver.build_graph(num_chunks=10)
    smc_solver.add_camera_to_graph([os.path.join(scene_dir, 'SfM_products', 'agisoft_cameras.xml')], camera_type="Agisoft")
    