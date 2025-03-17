from ssfm.files import *
from ssfm.image_segmentation import *
from ssfm.probabilistic_projection import *

import os
import time
import numpy as np
from collections import defaultdict
import logging
import yaml
from collections import Counter
from tqdm import tqdm
from scipy.spatial import cKDTree

import torch

from utils.main_utils import *
from utils.sam_utils import *
from tqdm import trange
import open3d as o3d
from segment_anything import sam_model_registry, SamPredictor
import gc

def process_batch(
    predictor,
    points: torch.Tensor,
    ins_idxs: torch.Tensor,
    im_size: Tuple[int, ...],
) -> MaskData:
    points = points[:, [1, 0]]
    transformed_points = predictor.transform.apply_coords_torch(points, im_size)
    in_points = torch.as_tensor(transformed_points, device=predictor.device)
    in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)

    masks, iou_preds, _ = predictor.predict_torch(
        in_points[:, None, :],
        in_labels[:, None],
        multimask_output=False,
        return_logits=True,
    )

    # Serialize predictions and store in MaskData  
    data_original = MaskData(
        masks=masks.flatten(0, 1),
        iou_preds=iou_preds.flatten(0, 1),
        points=points, 
        corre_3d_ins=ins_idxs 
    )

    # save masks, ins_idxs
    frame_name = 0
    save_file_name = str(frame_name) + ".npy"
    np.save("mask_" + save_file_name, masks.cpu().numpy())
    np.save("idx_" + save_file_name, ins_idxs.cpu().numpy())
    

    return data_original

def furthest_point_sampling(xyz, num_samples):
    N, C = xyz.shape  # Ensure it's (N, 3)

    fps_idx = torch.zeros(num_samples, dtype=torch.long, device=xyz.device)
    farthest = torch.randint(0, N, (1,), device=xyz.device)
    distances = torch.full((N,), float('inf'), device=xyz.device)

    for i in tqdm(range(num_samples)):
        fps_idx[i] = farthest
        farthest_point = xyz[farthest, :].squeeze(0)  
        dist = torch.sum((xyz - farthest_point) ** 2, dim=1)  
        distances = torch.minimum(distances, dist)
        farthest = torch.argmax(distances).unsqueeze(0)

    return fps_idx


def prompt_init(xyz, rgb, voxel_size, device):
    """
    Initializes 3D prompts using voxelization and Furthest Point Sampling (FPS).

    :param xyz: (N, 3) Numpy array of 3D point coordinates
    :param rgb: (N, 3) Numpy array of RGB colors
    :param voxel_size: Float, size of the voxel grid for downsampling
    :param device: PyTorch device (CPU or CUDA)
    :return: fps_points (M, 3) Tensor of sampled 3D points
             fps_colors (M, 3) Tensor of sampled colors
    """
    # Apply voxelization
    idx_sort, num_pt = voxelize(xyz, voxel_size, mode=1)
    print("The number of initial 3D prompts:", len(num_pt))
    
    # Convert to PyTorch tensors
    xyz = torch.from_numpy(xyz).to(device=device).contiguous()
    rgb = torch.from_numpy(rgb / 256.).to(device=device).contiguous()

    # Apply FPS using PyTorch implementation
    num_samples = len(num_pt)  # Number of points to sample
    fps_idx = furthest_point_sampling(xyz, num_samples)
    
    # Get sampled points and colors
    fps_points = xyz[fps_idx, :]
    fps_colors = rgb[fps_idx, :]

    return fps_points, fps_colors, fps_idx


def create_output_folders(data_folder_path):
    sampro3d_folder_path = os.path.join(data_folder_path, 'sampro3d')
    if not os.path.exists(sampro3d_folder_path):
        os.makedirs(sampro3d_folder_path)
    
    points_folder_path = os.path.join(sampro3d_folder_path, 'points_npy')
    if not os.path.exists(points_folder_path):
        os.makedirs(points_folder_path)

    iou_preds_folder_path = os.path.join(sampro3d_folder_path, 'iou_preds_npy')
    if not os.path.exists(iou_preds_folder_path):
        os.makedirs(iou_preds_folder_path)

    masks_folder_path = os.path.join(sampro3d_folder_path, 'masks_npy')
    if not os.path.exists(masks_folder_path):
        os.makedirs(masks_folder_path)

    corre_3d_ins_folder_path = os.path.join(sampro3d_folder_path, 'corre_3d_ins_npy')
    if not os.path.exists(corre_3d_ins_folder_path):
        os.makedirs(corre_3d_ins_folder_path)

    return points_folder_path, iou_preds_folder_path, masks_folder_path, corre_3d_ins_folder_path


class InitialProcess(object):
    def __init__(self, pointcloud_path, segmentation_folder_path, association_folder_path, image_folder_path, keyimage_associations_file_name=None, image_list=None, device=torch.device('cuda:1')):
        self.pointcloud_path = pointcloud_path
        self.segmentation_folder_path = segmentation_folder_path
        self.association_folder_path = association_folder_path
        self.scene_folder_path = os.path.dirname(self.segmentation_folder_path)
        self.image_folder_path = image_folder_path

        # check if the pointcloud file exists
        assert os.path.exists(self.pointcloud_path), 'Pointcloud path does not exist.'

        if pointcloud_path.endswith('.las'):
            with laspy.open(self.pointcloud_path) as las_file:
                header = las_file.header
                self.N_points = header.point_count
            
            pc = laspy.read(self.pointcloud_path)
            self.points = np.vstack((pc.x, pc.y, pc.z)).T
            self.colors = np.vstack((pc.red, pc.green, pc.blue)).T
            self.semantics_gt = pc.intensity

        elif pointcloud_path.endswith('.npy'):
            mesh_vertices_color = np.load(pointcloud_path)
            self.points = mesh_vertices_color[0]
            self.colors = mesh_vertices_color[1]
            self.N_points = points.shape[0]

        # load keyimage association files (.npy)
        if keyimage_associations_file_name is None:
            keyimage_associations_file_path = os.path.join(self.association_folder_path, 'associations_keyimage.npy')
            assert os.path.exists(keyimage_associations_file_path), 'Keyimage association file path does not exist.'
            self.keyimage_associations = np.load(keyimage_associations_file_path, allow_pickle=True)
        else:
            keyimage_associations_file_path = os.path.join(self.association_folder_path, keyimage_associations_file_name)
            assert os.path.exists(keyimage_associations_file_path), 'Keyimage association file path does not exist.'
            self.keyimage_associations = np.load(keyimage_associations_file_path, allow_pickle=True)


        # Load pixel2point association files (.npy) and sort them
        if image_list is None:
            # check if the folder exists
            self.associations_pixel2point_path = os.path.join(self.association_folder_path, 'pixel2point')
            assert os.path.exists(self.associations_pixel2point_path), 'Association pixel2point folder path does not exist.'
            self.associations_pixel2point_file_paths = [os.path.join(self.associations_pixel2point_path, f) for f in os.listdir(self.associations_pixel2point_path) if f.endswith('.npy')]
            # sort the association files based on the number in the file name
            self.associations_pixel2point_file_paths = sorted(self.associations_pixel2point_file_paths, key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))

            # Load point2pixel association files (.npy) and sort them
            # check if the folder exists
            self.associations_point2pixel_path = os.path.join(self.association_folder_path, 'point2pixel')
            assert os.path.exists(self.associations_point2pixel_path), 'Association point2pixel folder path does not exist.'
            self.associations_point2pixel_file_paths = [os.path.join(self.associations_point2pixel_path, f) for f in os.listdir(self.associations_point2pixel_path) if f.endswith('.npy')]
            # sort the association files based on the number in the file name
            self.associations_point2pixel_file_paths = sorted(self.associations_point2pixel_file_paths, key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))

            # Load segmentation files (.npy) and sort them
            # check if the folder exists
            assert os.path.exists(self.segmentation_folder_path), 'Segmentation folder path does not exist.'
            self.segmentation_file_paths = [os.path.join(self.segmentation_folder_path, f) for f in os.listdir(self.segmentation_folder_path) if f.endswith('.npy')]
            # sort the segmentation files based on the number in the file name
            self.segmentation_file_paths = sorted(self.segmentation_file_paths, key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))

            # Load image files and sort them
            assert os.path.exists(self.image_folder_path), 'Image folder path does not exist.'
            image_list = [f for f in os.listdir(self.image_folder_path) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.JPG')]
            image_list = sorted(image_list, key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))
            basename_valid = [os.path.basename(f).split('.')[0] for f in self.segmentation_file_paths]
            image_list_valid = []
            for image in image_list:
                if os.path.basename(image).split('.')[0] in basename_valid:
                    image_list_valid.append(image)
            image_list = image_list_valid

        else: 
            keyimage_list = [f.replace('.jpg', '.npy').replace('.png', '.npy').replace('.JPG', '.npy') for f in image_list]
                       
            self.associations_pixel2point_path = os.path.join(self.association_folder_path, 'pixel2point')
            assert os.path.exists(self.associations_pixel2point_path), 'Association pixel2point folder path does not exist.'
            associations_pixel2point_files = [f for f in os.listdir(self.associations_pixel2point_path) if f.endswith('.npy')]
            self.associations_pixel2point_file_paths = [os.path.join(self.associations_pixel2point_path, f) for f in keyimage_list if f in associations_pixel2point_files]

            self.associations_point2pixel_path = os.path.join(self.association_folder_path, 'point2pixel')
            assert os.path.exists(self.associations_point2pixel_path), 'Association point2pixel folder path does not exist.'
            associations_point2pixel_files = [f for f in os.listdir(self.associations_point2pixel_path) if f.endswith('.npy')]
            self.associations_point2pixel_file_paths = [os.path.join(self.associations_point2pixel_path, f) for f in keyimage_list if f in associations_point2pixel_files]

            assert os.path.exists(self.segmentation_folder_path), 'Segmentation folder path does not exist.'
            segmentation_files = [f for f in os.listdir(self.segmentation_folder_path) if f.endswith('.npy')]
            self.segmentation_file_paths = [os.path.join(self.segmentation_folder_path, f) for f in keyimage_list if f in segmentation_files]
        

        # Check if the number of segmentation files and association files are the same
        assert len(self.segmentation_file_paths) == len(self.associations_pixel2point_file_paths), 'The number of segmentation files and pixel2point association files are not the same.'
        assert len(self.segmentation_file_paths) == len(self.associations_point2pixel_file_paths), 'The number of segmentation files and point2pixel association files are not the same.'

        # create segmentation-association pairs
        self.segmentation_association_pairs = []
        for i in range(len(self.segmentation_file_paths)):
            # check if the segmentation file name and association file names are the same
            segmentation_file_name = os.path.basename(self.segmentation_file_paths[i]).split('.')[0]
            associations_pixel2point_file_name = os.path.basename(self.associations_pixel2point_file_paths[i]).split('.')[0]
            associations_point2pixel_file_name = os.path.basename(self.associations_point2pixel_file_paths[i]).split('.')[0]
            assert segmentation_file_name == associations_pixel2point_file_name, 'The segmentation file name and association pixel2point file name are not the same.'
            assert segmentation_file_name == associations_point2pixel_file_name, 'The segmentation file name and association point2pixel file name are not the same.'
            self.segmentation_association_pairs.append((self.segmentation_file_paths[i], self.associations_pixel2point_file_paths[i], 
                                                        self.associations_point2pixel_file_paths[i], image_list[i]))


        self.voxel_size = 3
        self.device = device

        self.prompt_folder_path = os.path.join(self.scene_folder_path, 'sampro3d')
        if not os.path.exists(self.prompt_folder_path):
            os.makedirs(self.prompt_folder_path)
        self.prompt_ply_path = os.path.join(self.prompt_folder_path, 'init_prompt.ply')
        self.prompt_init_idx_path = os.path.join(self.prompt_folder_path, 'init_prompt_idx.npy')
        self.points_folder_path, self.iou_preds_folder_path, self.masks_folder_path, self.corre_3d_ins_folder_path = create_output_folders(self.scene_folder_path)

        # unique semantics and their counts
        self.unique_semantics, self.unique_semantics_counts = np.unique(self.semantics_gt, return_counts=True)
        # find the most frequent semantics
        self.most_frequent_semantics = self.unique_semantics[np.argmax(self.unique_semantics_counts)]
        

    def init_process(self, excluding_floor=True):
        if excluding_floor:
            # get non-floor idx
            non_floor_idx = np.where(self.semantics_gt != self.most_frequent_semantics)[0]
            xyz = self.points[non_floor_idx]
            rgb = self.colors[non_floor_idx]
            init_prompt, init_color, init_idx = prompt_init(xyz, rgb, self.voxel_size, self.device)
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(init_prompt.cpu().numpy())
            point_cloud.colors = o3d.utility.Vector3dVector(init_color.cpu().numpy())
            o3d.io.write_point_cloud(self.prompt_ply_path, point_cloud)
            # update init_idx, which is the indices of the initial prompts in the original point cloud
            init_idx = non_floor_idx[init_idx.cpu().numpy()]
            np.save(self.prompt_init_idx_path, init_idx)

        else:
            xyz = self.points
            rgb = self.colors
            init_prompt, init_color, init_idx = prompt_init(xyz, rgb, self.voxel_size, self.device)
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(init_prompt.cpu().numpy())
            point_cloud.colors = o3d.utility.Vector3dVector(init_color.cpu().numpy())
            o3d.io.write_point_cloud(self.prompt_ply_path, point_cloud)
            np.save(self.prompt_init_idx_path, init_idx.cpu().numpy())

    def sam_seg(self):
        sam = sam_model_registry['vit_h'](checkpoint="../../semantic_SfM/sam/sam_vit_h_4b8939.pth").to(device=self.device)
        predictor = SamPredictor(sam)
        
        for i in tqdm(range(len(self.segmentation_association_pairs))):
            self.frame_id = i
            self.frame_name = os.path.basename(self.segmentation_association_pairs[i][3]).split('.')[0]
            self.associations_pixel2point = np.load(self.segmentation_association_pairs[i][1], allow_pickle=True)
            self.associations_point2pixel = np.load(self.segmentation_association_pairs[i][2], allow_pickle=True)
            self.image_path = os.path.join(self.image_folder_path, self.segmentation_association_pairs[i][3])

            save_file_name = str(self.frame_name) + ".npy"
            save_file_path = os.path.join(self.points_folder_path, save_file_name)
            # if os.path.exists(save_file_path):
            #     continue

            image = cv2.imread(self.image_path)
            # convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_size = image.shape[:2]

            # load initial prompts
            init_xyz, init_rgb = load_ply(self.prompt_ply_path)
            init_idx = np.load(self.prompt_init_idx_path)

            # Use boolean masking to remove (-1, -1) pixel indices
            valid_pixels_mask = (self.associations_point2pixel[init_idx][:, 0] != -1)
            input_pixel_indices_ = self.associations_point2pixel[init_idx][valid_pixels_mask]

            # Vectorized indexing for point indices
            input_point_indices_ = self.associations_pixel2point[input_pixel_indices_[:, 0], input_pixel_indices_[:, 1]]

            # Remove -1 point indices using boolean masking
            valid_points_mask = (input_point_indices_ != -1)
            input_point_indices_ = input_point_indices_[valid_points_mask]

            # save input_point_indices_
            # np.save(os.path.join(self.prompt_folder_path, str(self.frame_name) + '_input_point_indices_.npy'), input_point_indices_)

            # find indices where `input_point_indices_` is in `init_idx`
            projected_idx = np.where(np.isin(init_idx, input_point_indices_))[0]

            # Convert to NumPy array
            projected_init_idx = np.array(projected_idx)
            
            # Convert input_pixel_indices_, projected_init_idx to torch tensors
            input_point_pos = torch.from_numpy(input_pixel_indices_).to(device=self.device)
            corre_ins_idx = torch.from_numpy(projected_init_idx).to(device=self.device)

            # if input_point_pos is empty, skip the frame
            if input_point_pos.size(0) == 0:
                continue

            # save input_point_pos and corre_ins_idx
            # np.save(os.path.join(self.prompt_folder_path, str(self.frame_name) + '_input_point_pos.npy'), input_point_pos.cpu().numpy())
            # np.save(os.path.join(self.prompt_folder_path, str(self.frame_name) + '_corre_ins_idx_.npy'), corre_ins_idx.cpu().numpy())

            # break

            predictor.set_image(image)
            # SAM segmetaion on image
            data_original = MaskData()


            try:
                for (points, ins_idxs) in batch_iterator(16, input_point_pos, corre_ins_idx):
                    batch_data_original = process_batch(predictor, points, ins_idxs, image_size)
                    data_original.cat(batch_data_original)
                    del batch_data_original
                predictor.reset_image()
                data_original.to_numpy()

                save_file_name = str(self.frame_name) + ".npy"
                np.save(os.path.join(self.points_folder_path, save_file_name), data_original["points"])
                np.save(os.path.join(self.masks_folder_path, save_file_name), data_original["masks"])  
                np.save(os.path.join(self.iou_preds_folder_path, save_file_name), data_original["iou_preds"])  
                np.save(os.path.join(self.corre_3d_ins_folder_path, save_file_name), data_original["corre_3d_ins"])
            except Exception as e:
                print(e)
                print("Error occurred in frame: ", self.frame_name)
                continue

            # clear GPU memory
            del data_original
            torch.cuda.empty_cache()
            gc.collect()
            del input_point_pos
            del corre_ins_idx

def batch_initialization(sampling=False):
    scene_paths = [os.path.join('../../data', f) for f in os.listdir('../../data') if "kubric" in f]
    # ramdomly sample 10 scenes
    # set seed
    if sampling:
        np.random.seed(0)
        scene_paths = np.random.choice(scene_paths, 10, replace=False)
    for scene_path in scene_paths:
        device = torch.device('cuda:1')
        pointcloud_path = os.path.join(scene_path, 'reconstructions', 'combined_point_cloud.las')
        segmentation_folder_path = os.path.join(scene_path, 'segmentations_filtered')
        association_folder_path = os.path.join(scene_path, 'associations')
        image_folder_path = os.path.join(scene_path, 'DJI_photos')

        init_process = InitialProcess(pointcloud_path, segmentation_folder_path, association_folder_path, image_folder_path, device=device)
        init_process.init_process()
        init_process.sam_seg()
    
def clear_sampro3d():
    scene_paths = [os.path.join('../../data', f) for f in os.listdir('../../data') if "kubric" in f]
    for scene_path in scene_paths:
        sampro3d_folder_path = os.path.join(scene_path, 'sampro3d')
        if os.path.exists(sampro3d_folder_path):
            os.system("rm -rf " + sampro3d_folder_path)

def gd_initilization():
    scene_path = '../../data/granite_dells'
    device = torch.device('cuda:0')
    pointcloud_path = os.path.join(scene_path, 'SfM_products', 'downsampled_sampro3d.las')
    segmentation_folder_path = os.path.join(scene_path, 'segmentations_classes')
    association_folder_path = os.path.join(scene_path, 'associations')
    image_folder_path = os.path.join(scene_path, 'DJI_photos')

    init_process = InitialProcess(pointcloud_path, segmentation_folder_path, association_folder_path, image_folder_path, device=device)
    init_process.init_process(excluding_floor=False)
    init_process.sam_seg()

if __name__ == '__main__':
    """
    scene_path = '../../data/kubric_0'
    device = torch.device('cuda:1')
    pointcloud_path = os.path.join(scene_path, 'reconstructions', 'combined_point_cloud.las')
    segmentation_folder_path = os.path.join(scene_path, 'segmentations_filtered')
    association_folder_path = os.path.join(scene_path, 'associations')
    image_folder_path = os.path.join(scene_path, 'photos')

    init_process = InitialProcess(pointcloud_path, segmentation_folder_path, association_folder_path, image_folder_path, device=device)
    init_process.init_process()
    init_process.sam_seg()
    """
    #clear_sampro3d()
    #batch_initialization(sampling=True)
    gd_initilization()