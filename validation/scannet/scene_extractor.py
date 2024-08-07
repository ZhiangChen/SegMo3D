import os
import numpy as np
import json
import shutil 
from PIL import Image
from joblib import Parallel, delayed
from tqdm import tqdm
import open3d as o3d
import json

class SceneExtractor(object):
    def __init__(self, scene_dir, save_dir, extract_scene_folder='output'):
        self.scene_dir = scene_dir
        self.save_dir = save_dir
        self.extract_scene_folder = extract_scene_folder
        assert os.path.exists(scene_dir), 'Scene directory does not exist!'
        self.extract_scene_folder_path = os.path.join(self.scene_dir, extract_scene_folder)
        assert os.path.exists(self.extract_scene_folder_path), 'Extract scene folder does not exist!'

        # create photos folder
        self.photo_folder_path = os.path.join(save_dir, 'photos')
        if not os.path.exists(self.photo_folder_path):
            os.makedirs(self.photo_folder_path)

        # create segmentations folder
        self.segmentation_folder_path = os.path.join(save_dir, 'segmentations')
        if not os.path.exists(self.segmentation_folder_path):
            os.makedirs(self.segmentation_folder_path)
        
        # create associations folder
        self.associations_folder_path = os.path.join(save_dir, 'associations')
        if not os.path.exists(self.associations_folder_path):
            os.makedirs(self.associations_folder_path)

        # create reconstruction folder
        self.reconstruction_folder_path = os.path.join(save_dir, 'reconstructions')
        if not os.path.exists(self.reconstruction_folder_path):
            os.makedirs(self.reconstruction_folder_path)

        

    def extract_scene(self):
        pass

    def extract_photos(self):
        scannet_photo_folder_path = os.path.join(self.extract_scene_folder_path, 'color')
        # move all photos in scannet_photo_folder_path to self.photo_folder_path
        for file in os.listdir(scannet_photo_folder_path):
            if file.endswith('.jpg'):
                os.rename(os.path.join(scannet_photo_folder_path, file), os.path.join(self.photo_folder_path, file))
        
        # print the number of photos extracted
        print(f'Extracted {len(os.listdir(self.photo_folder_path))} photos')

    def extract_depths(self):
        scannet_depth_folder_path = os.path.join(self.extract_scene_folder_path, 'depth')
        depth_folder_path = os.path.join(self.associations_folder_path, 'depth')
        if not os.path.exists(depth_folder_path):
            os.makedirs(depth_folder_path)
        else:
            # remove all files in depth_folder_path
            for file in os.listdir(depth_folder_path):
                os.remove(os.path.join(depth_folder_path, file))

        # read each depth file in scannet_depth_folder_path, convert from mm to m, and save in depth_folder_path, in parallel processing
        Parallel(n_jobs=8)(delayed(self.convert_depth)(file, scannet_depth_folder_path, depth_folder_path) for file in os.listdir(scannet_depth_folder_path))
        # print the number of depths extracted
        print(f'Extracted {len(os.listdir(depth_folder_path))} depths')

    def convert_depth(self, file, scannet_depth_folder_path, depth_folder_path):
        # read depth file (.png) in scannet_depth_folder_path. Note depth is a single channel image
        depth = np.array(Image.open(os.path.join(scannet_depth_folder_path, file)))
        # convert depth from mm to m
        depth = depth / 1000
        # save_depth_file should be npy instead of png
        np.save(os.path.join(depth_folder_path, file.replace('.png', '.npy')), depth )

    def extract_segmentations(self):
        # get scene name from scene_dir
        scene_name = os.path.basename(self.scene_dir)
        unzip_folder = "instance-filt"
        unzip_folder_path = os.path.join(self.scene_dir, unzip_folder)
        if not os.path.exists(unzip_folder_path):
            # filtered instance segmentation zip file path
            filtered_instance_segmentation_zip_path = os.path.join(self.scene_dir, f'{scene_name}_2d-instance-filt.zip')
            # extract raw instance segmentation zip file to current folder
            # note that instance segmentation files: 
            os.system(f'unzip {filtered_instance_segmentation_zip_path} -d {self.scene_dir}')
        else:
            None

        # read each instance segmentation file in unzip_folder_path, convert to npy files and save in unzip_folder_path, in parallel processing
        Parallel(n_jobs=8)(delayed(self.convert_segmentation)(file, unzip_folder_path) for file in os.listdir(unzip_folder_path))
        # print the number of segmentations extracted
        print(f'Extracted {len(os.listdir(unzip_folder_path))} segmentations')

    def convert_segmentation(self, file, unzip_folder_path):
        # read segmentation file (.png) in unzip_folder_path. Note segmentation is a single channel image
        img = Image.open(os.path.join(unzip_folder_path, file))
        img = np.array(img)
        # get unique values in segmentation
        seg_ids = np.unique(img)
        # sort seg_ids
        seg_ids.sort()
        # create an empty array with the same shape as img with intial value of -1
        segmentation = np.zeros_like(np.array(img), dtype=np.int16) - 1
        # iterate through seg_ids and assign the index to segmentation
        new_id = 0
        for i in seg_ids[1:]:
            segmentation[img == i] = new_id
            new_id += 1
        # save_segmentation_file should be npy instead of png
        np.save(os.path.join(self.segmentation_folder_path, file.replace('.png', '.npy')), segmentation )

    def extract_reconstruction(self):
        # get scene name from scene_dir
        scene_name = os.path.basename(self.scene_dir)
        # mesh file path: <scene_name>_vh_clean.ply
        mesh_file_path = os.path.join(self.scene_dir, f'{scene_name}_vh_clean_2.ply')
        # read mesh file
        mesh = o3d.io.read_triangle_mesh(mesh_file_path)

        # check if mesh has vertices
        if not mesh.has_vertices():
            raise ValueError('Mesh does not have vertices!')
        
        # check if vertices have color
        if not mesh.has_vertex_colors():
            raise ValueError('Mesh does not have vertex colors!')
        
        # extract vertices and colors
        vertices = np.asarray(mesh.vertices)
        colors = np.asarray(mesh.vertex_colors)

        mesh_ndarray = np.asarray([vertices, colors])
        # save mesh as npy file under reconstruction folder
        np.save(os.path.join(self.reconstruction_folder_path, 'mesh_vertices_color.npy'), mesh_ndarray)

        # print the number of vertices and colors extracted
        print(f'Extracted {len(vertices)} vertices and {len(colors)} colors')

        # read intrinsic parameters
        intrinsic_color_file_path = os.path.join(self.scene_dir, f'{self.extract_scene_folder}/intrinsic/intrinsic_color.txt')
        intrinsic_depth_file_path = os.path.join(self.scene_dir, f'{self.extract_scene_folder}/intrinsic/intrinsic_depth.txt')
        assert os.path.exists(intrinsic_color_file_path), 'Intrinsic color file does not exist!'
        assert os.path.exists(intrinsic_depth_file_path), 'Intrinsic depth file does not exist!'

        # read intrinsic color parameters
        intrinsic_color = np.loadtxt(intrinsic_color_file_path)
        # read intrinsic depth parameters
        intrinsic_depth = np.loadtxt(intrinsic_depth_file_path)
        
        intrinsic_color_depth = np.asarray([intrinsic_color, intrinsic_depth])
        # save intrinsic parameters as npy file under reconstruction folder
        np.save(os.path.join(self.reconstruction_folder_path, 'intrinsic_color_depth.npy'), intrinsic_color_depth)

        camera_poses = dict()
        # read camera poses 
        camera_poses_folder_path = os.path.join(self.scene_dir, f'{self.extract_scene_folder}/pose')
        assert os.path.exists(camera_poses_folder_path), 'Camera poses folder does not exist!'
        for pose_file in os.listdir(camera_poses_folder_path):
            pose_file_path = os.path.join(camera_poses_folder_path, pose_file)
            pose = np.loadtxt(pose_file_path)
            frame_name = pose_file.split('.')[0] + '.jpg'
            camera_poses[frame_name] = pose

        # save camera poses as npy files under reconstruction folder
        np.save(os.path.join(self.reconstruction_folder_path, 'camera_poses.npy'), camera_poses)
        

    def extract_ground_truth(self, ):
        # get scene name from scene_dir
        scene_name = os.path.basename(self.scene_dir)
        # over-segmentation annotation file path: <scanId>_vh_clean.aggregation.json
        over_annotation_file_path = os.path.join(self.scene_dir, f'{scene_name}_vh_clean_2.0.010000.segs.json')
        # read over-segmentation annotation file
        with open(over_annotation_file_path, 'r') as f:
            over_annotation = json.load(f)
        
        segIndices = over_annotation['segIndices']

        # instance-segmentation annotation file path: <scanId>.aggregation.json
        instance_annotation_file_path = os.path.join(self.scene_dir, f'{scene_name}.aggregation.json')
        # read instance-segmentation annotation file
        with open(instance_annotation_file_path, 'r') as f:
            instance_annotation = json.load(f)
            
        segGroups = instance_annotation['segGroups']
        # build a dictionary of segGroups: key is segIndex, value is segGroup id
        segGroupDict = dict()
        for segGroup in segGroups:
            for segIndex in segGroup['segments']:
                # if segIndex is not in segGroupDict, add it
                if segIndex not in segGroupDict:
                    segGroupDict[segIndex] = segGroup['id']
                else:
                    raise ValueError('segIndex already exists in segGroupDict!')
                
        # get the maximum segGroup id
        max_segGroup_id = max([segGroup['id'] for segGroup in segGroups])

        semantic_points = []
        for segIndex in segIndices:
            # if segIndex is in segGroupDict, add it to semantic_points
            if segIndex in segGroupDict:
                # get segGroup id from segGroupDict
                segGroup = segGroupDict[segIndex]
                semantic_points.append(segGroup)
            else:
                semantic_points.append(max_segGroup_id + 1)

        # save semantic points as npy file under associations folder
        np.save(os.path.join(self.associations_folder_path, 'semantic_points.npy'), semantic_points)


def batch_extract_scenes(scene_dir, save_dir):
    # iterate over folders in scene_dir
    for scene_folder in os.listdir(scene_dir):
        scene_folder_path = os.path.join(scene_dir, scene_folder)
        save_folder_path = os.path.join(save_dir, scene_folder)
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        scene_extractor = SceneExtractor(scene_folder_path, save_folder_path)
        scene_extractor.extract_photos()
        scene_extractor.extract_depths()
        #scene_extractor.extract_segmentations()
        scene_extractor.extract_reconstruction()
        #scene_extractor.extract_ground_truth()




if __name__ == '__main__':
    # scene_dir = '../../data/scannet/scans/scene0000_00'
    # save_dir = '../../data/scene0000_00'
    # scene_extractor = SceneExtractor(scene_dir, save_dir)
    # #scene_extractor.extract_photos()
    # #scene_extractor.extract_depths()
    # #scene_extractor.extract_segmentations()
    # #scene_extractor.extract_reconstruction()
    # scene_extractor.extract_ground_truth()

    scene_dir = '../../data/scannet/scans_test'
    save_dir = '../../data/scannet/ssfm'
    batch_extract_scenes(scene_dir, save_dir)