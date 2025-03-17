"""
Main Script (including 2D-Guided Prompt Filter, Prompt Consolidation, 3D Segmentaiton in the paper)

Author: Mutian Xu (mutianxu@link.cuhk.edu.cn)

Modified by Zhiang Chen
"""

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("default")
import os
import sys
import numpy as np
import torch
import argparse
from natsort import natsorted 
import open3d as o3d
from tqdm import tqdm

import laspy
from utils.sam_utils import *
from utils.main_utils import *
from utils.vis_utils import *
from segment_anything import sam_model_registry, SamPredictor

from ssfm.files import *

from scipy.spatial import cKDTree
from torchvision.ops import batched_nms


def add_semantics_to_pointcloud(pointcloud_path, semantics_path, save_las_path, remove_small_N=0, nearest_interpolation=0):
    """
    Add semantics to the point cloud and save it as a .las file.

    Parameters
    ----------
    pointcloud_path : str, the path to the .las file
    semantics_path : str, the path to the semantics file
    save_las_path : str, the path to save the .las file
    remove_small_N : int, remove the semantics with numbers smaller than N
    nearest_interpolation : 0, not to use nearest interpolation to assign semantics to the unlabeled points; positive integer, the number of nearest neighbors to use for nearest interpolation

    Returns
    -------
    None
    """
    if pointcloud_path.endswith('.las'):
        las = laspy.read(pointcloud_path)
        points = np.vstack((las.x, las.y, las.z)).T
        colors = np.vstack((las.red, las.green, las.blue)).T
        semantics_gt = las.intensity

    # unqiue semantics and their counts
    unique_semantics, counts = np.unique(semantics_gt, return_counts=True)
    # the most frequent semantics is the floor
    floor_id = unique_semantics[np.argmax(counts)]
    # get non-floor indices
    non_floor_indices = np.where(semantics_gt != floor_id)[0]
    # get floor indices
    floor_indices = np.where(semantics_gt == floor_id)[0]  



    semantics = np.load(semantics_path)
    floor_semantic = np.max(semantics) + 1
    semantics[floor_indices] = floor_semantic

    semantics_ids = np.unique(semantics)
    N_semantics = len(semantics_ids)

    print("Before removing small semantics: ")
    print('maximum of semantics: ', semantics.max())
    print('number of unique semantics: ', N_semantics)

    # remove the semantics with numbers smaller than a threshold
    for semantics_id in semantics_ids:
        if np.sum(semantics == semantics_id) < remove_small_N:
            semantics[semantics == semantics_id] = -1

    print("After removing small semantics: ")
    print('number of unique semantics: ', len(np.unique(semantics)))
            

    # construct a .las file
    hdr = laspy.LasHeader(version="1.2", point_format=3)
    hdr.scale = [0.0001, 0.0001, 0.0001]  # Example scale factor, adjust as needed
    hdr.offset = np.mean(points, axis=0)

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

    # Add semantics
    if nearest_interpolation == 0:
        # make the semantics start from 0. For all semantics < 0, set them to the maximum semantics + 1
        semantics[semantics < 0] = semantics.max() + 1
        las.intensity = semantics
        # add floor semantics
        floor_semantic = np.max(semantics) + 1
        # add floor semantics
        # las.intensity[floor_indices] = floor_semantics
        # las.intensity[non_floor_indices] = semantics

    elif nearest_interpolation > 0:
        # labeled points are the points with semantics >=0; unlabeled points are the points with semantics < 0
        # points = points[non_floor_indices]
        labeled_points = points[semantics >= 0]
        labeled_semantics = semantics[semantics >= 0]
        unlabeled_points = points[semantics < 0]

        # construct a KDTree
        tree = cKDTree(labeled_points)

        # find the N nearest neighbors for the unlabeled points
        N = nearest_interpolation
        distances, indices = tree.query(unlabeled_points, k=N)
        nearest_semantics = labeled_semantics[indices]
        # find the most frequent semantics
        unlabeled_semantics = np.array([np.argmax(np.bincount(nearest_semantics[i])) for i in range(unlabeled_points.shape[0])])

        # combine the labeled and unlabeled semantics
        combined_semantics = np.zeros(semantics.shape)
        combined_semantics[semantics >= 0] = labeled_semantics
        combined_semantics[semantics < 0] = unlabeled_semantics

        # add floor semantics
        # las.intensity[floor_indices] = floor_semantics
        # las.intensity[non_floor_indices] = combined_semantics
        las.intensity = combined_semantics
        floor_semantic = np.max(combined_semantics) + 1
    
    else:
        raise ValueError("nearest_interpolation should be a non-negative integer")

    # Write the LAS file
    las.write(save_las_path)


# 2D-Guided Prompt Filter:
def prompt_filter(init_prompt, scene_output_path, npy_path, predictor, device, pred_iou_thres, stability_score_thres, box_nms_thres, keep_thres):
    # gap = 1  # number of skipped frames
    stop_limit = 10  # we find that not considering all frames for filter is better

    keep_score = torch.zeros(len(init_prompt), device=device)
    counter = torch.zeros(len(init_prompt), device=device)
    del_score = torch.zeros(len(init_prompt), device=device)

    for i, (npy_file) in enumerate(tqdm(npy_path)):
        # if i != 0 and i % gap != 0:
        #     continue

        # load the corresponding SAM segmentations data of the corresponding frame:
        points_data = torch.from_numpy(np.load(os.path.join(scene_output_path, 'points_npy', npy_file))).to(device)
        iou_preds_data = torch.from_numpy(np.load(os.path.join(scene_output_path, 'iou_preds_npy', npy_file))).to(device)
        masks_data = torch.from_numpy(np.load(os.path.join(scene_output_path, 'masks_npy', npy_file))).to(device)
        corre_3d_ins_data = torch.from_numpy(np.load(os.path.join(scene_output_path, 'corre_3d_ins_npy', npy_file))).to(device)  # the valid (i.e., has mapped pixels at the current frame) prompt ID  in the original 3D point cloud of initial prompts
        data = MaskData(
                masks=masks_data,
                iou_preds=iou_preds_data,
                points=points_data, 
                corre_3d_ins=corre_3d_ins_data 
            )

        corr_ins_idx = data['corre_3d_ins']
        # ins_flag[corr_ins_idx] = 1 # set the valid ins value to 1
        valid_corr_ins_idx = (corr_ins_idx >= 0) & (corr_ins_idx < counter.shape[0])
        if not valid_corr_ins_idx.all():
            print(f"Invalid corr_ins_idx detected: {corr_ins_idx[~valid_corr_ins_idx]}")
            
        corr_ins_idx = corr_ins_idx[valid_corr_ins_idx]  # Keep only valid indices
        counter[corr_ins_idx] += 1
        # counter[corr_ins_idx] += 1  # only count if it is not the stopped instances  


        # print(f"Counter min: {counter.min()}, max: {counter.max()}, shape: {counter.shape}")
        # print(f"Stop limit: {stop_limit}")

        # try:
        #     stop_id = torch.where(counter >= stop_limit)[0]
        #     print(f"stop_id shape: {stop_id.shape}, min: {stop_id.min()}, max: {stop_id.max()}")
        # except Exception as e:
        #     print(f"Error in stop_id calculation: {e}")
        #     print(f"Counter tensor: {counter}")
        #     exit(1)  # Stop execution to debug the issue

        # stop_id = torch.where(counter >= stop_limit)[0] # bug

        stop_id = torch.where(counter >= stop_limit)[0]

        if stop_id.numel() == 0:  # Handle empty tensor
            print("No values in counter exceed stop_limit.")
            stop_id = torch.tensor([], device=counter.device, dtype=torch.long)  # Empty tensor of correct type
        else:
            print(f"stop_id shape: {stop_id.shape}, min: {stop_id.min()}, max: {stop_id.max()}")

        ############ start filter:
        # Filter by predicted IoU
        if pred_iou_thres > 0.0:
            keep_mask = data["iou_preds"] > pred_iou_thres
            data.filter(keep_mask)
        #     print(data['points'].shape)
        
        # Calculate stability score
        data["stability_score"] = calculate_stability_score(
            masks=data["masks"], mask_threshold=predictor.model.mask_threshold, threshold_offset=1.0
        )

        if stability_score_thres > 0.0:
            keep_mask = data["stability_score"] >= stability_score_thres
            data.filter(keep_mask)
    #     print(data['points'].shape)
        
        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > predictor.model.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        
        # Remove duplicates within this crop.
        from torchvision.ops.boxes import batched_nms, box_area 
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=box_nms_thres,
        )
        data.filter(keep_by_nms)

        keep_ins_idx = data["corre_3d_ins"]
        del_ins_idx = corr_ins_idx[torch.isin(corr_ins_idx, keep_ins_idx, invert=True)]

        #if stop_id.shape[0] > 0:
        keep_ins_idx = keep_ins_idx[torch.isin(keep_ins_idx, stop_id, invert=True)]
        del_ins_idx = del_ins_idx[torch.isin(del_ins_idx, stop_id, invert=True)]
        keep_score[keep_ins_idx] += 1
        del_score[del_ins_idx] += 1 

    # make all selected frames happy:
    counter[torch.where(counter >= stop_limit)] = stop_limit
    #counter[torch.where(counter == 0)] = -1  #  avoid that the the score is divided by counter of 0
    counter[counter == 0] = -1
    # keep prompts whose score is larger than a threshold:
    keep_score_mean = keep_score / counter
    keep_idx = torch.where(keep_score_mean >= keep_thres)[0]

    print("the number of prompts after filter", keep_idx.shape[0])

    return keep_idx

def perform_3dsegmentation0(xyz, keep_idx, scene_output_path, npy_path, device, scene_path):
    batch_size = 50000  # Process in batches to prevent OOM
    n_points = xyz.shape[0]
    num_ins = keep_idx.shape[0]

    pt_score = torch.zeros([n_points, num_ins], device=device)
    counter_final = torch.zeros([n_points, num_ins], device=device)

    for i, npy_file in enumerate(tqdm(npy_path)):
        # Load 2D segmentation data
        points_data = torch.from_numpy(np.load(os.path.join(scene_output_path, 'points_npy', npy_file))).to(device)
        iou_preds_data = torch.from_numpy(np.load(os.path.join(scene_output_path, 'iou_preds_npy', npy_file))).to(device)
        masks_data = torch.from_numpy(np.load(os.path.join(scene_output_path, 'masks_npy', npy_file))).to(device)
        corre_3d_ins_data = torch.from_numpy(np.load(os.path.join(scene_output_path, 'corre_3d_ins_npy', npy_file))).to(device)

        data = MaskData(
            masks=masks_data,
            iou_preds=iou_preds_data,
            points=points_data, 
            corre_3d_ins=corre_3d_ins_data
        )

        frame_id = npy_file[:-4]

        # Load 3D-2D association mappings
        association_pixel2point_path = os.path.join(scene_path, 'associations', 'pixel2point', frame_id + '.npy')
        association_point2pixel_path = os.path.join(scene_path, 'associations', 'point2pixel', frame_id + '.npy')

        association_pixel2point = np.load(association_pixel2point_path)
        association_point2pixel = np.load(association_point2pixel_path)

        n_points = association_point2pixel.shape[0]
        mapping = np.zeros((n_points, 3), dtype=int)

        valid_mask = association_point2pixel[:, 0] != -1
        mapping[valid_mask, 0] = association_point2pixel[valid_mask, 0]  # Y-coordinate
        mapping[valid_mask, 1] = association_point2pixel[valid_mask, 1]  # X-coordinate
        mapping[valid_mask, 2] = 1  # Mark as valid

        if mapping[:, 2].sum() == 0:  # No points correspond to this image
            continue

        mapping = torch.from_numpy(mapping).to(device)

        # Apply filtering based on `keep_idx`
        keep_mask = torch.isin(data["corre_3d_ins"], keep_idx)
        data.filter(keep_mask)

        masks_logits = data["masks"]
        masks = masks_logits > 0.

        # Keep only valid instance indices
        ins_idx_all = []
        for actual_idx in data["corre_3d_ins"]:
            ins_idx = torch.where(keep_idx == actual_idx)[0]
            ins_idx_all.append(ins_idx.item())

        # Process in Batches
        for start_idx in range(0, n_points, batch_size):
            end_idx = min(start_idx + batch_size, n_points)

            counter_point_batch = mapping[start_idx:end_idx, 2]
            counter_point_batch = counter_point_batch.reshape(-1, 1).repeat(1, num_ins)

            counter_ins = torch.zeros(num_ins, device=device)
            counter_ins[ins_idx_all] += 1
            counter_ins = counter_ins.reshape(1, -1).repeat(end_idx - start_idx, 1)

            counter_final[start_idx:end_idx] += (counter_point_batch * counter_ins)

            del counter_point_batch, counter_ins
            torch.cuda.empty_cache()  # Free GPU memory

        # Compute mask scores per batch
        for index, mask in enumerate(masks):
            ins_id = ins_idx_all[index]
            mask = mask.int()

            mask_2d_3d = mask[mapping[:, 0], mapping[:, 1]]
            mask_2d_3d = mask_2d_3d * mapping[:, 2]

            pt_score[:, ins_id] += mask_2d_3d

    # Move results to CPU for further processing
    pt_score_cpu = pt_score.cpu().numpy()
    counter_final_cpu = counter_final.cpu().numpy()

    counter_final_cpu[np.where(counter_final_cpu == 0)] = -1  # Avoid divide by zero
    pt_score_mean = pt_score_cpu / counter_final_cpu
    pt_score_abs = pt_score_cpu

    max_score = np.max(pt_score_mean, axis=-1)
    max_score_abs = np.max(pt_score_abs, axis=-1)

    # Resolve conflicts: If multiple instances have the same max score, prefer the one with higher raw score
    max_indices_mean = np.where(pt_score_mean == max_score[:, np.newaxis])
    pt_score_mean_new = pt_score_mean.copy()
    pt_score_mean_new[max_indices_mean] += pt_score_cpu[max_indices_mean]
    pt_pred_mean = np.argmax(pt_score_mean_new, axis=-1)

    pt_pred_abs = np.argmax(pt_score_abs, axis=-1)

    # Handle unassigned points
    low_pt_idx_mean = np.where(max_score <= 0.)[0]
    pt_score_mean[low_pt_idx_mean] = 0.
    pt_pred_mean[low_pt_idx_mean] = -1

    low_pt_idx_abs = np.where(max_score_abs <= 0.)[0]
    pt_score_abs[low_pt_idx_abs] = 0.
    pt_pred_abs[low_pt_idx_abs] = -1

    return pt_score_abs, pt_pred_abs, pt_score_mean

def perform_3dsegmentation(xyz, keep_idx, scene_output_path, npy_path, device, scene_path):
    # gap = 1  # number of skipped frames
    n_points = xyz.shape[0]
    num_ins = keep_idx.shape[0]
    pt_score = torch.zeros([n_points, num_ins], device=device)  # All input points have a score
    counter_final = torch.zeros([n_points, num_ins], device=device)

    for i, (npy_file) in enumerate(tqdm(npy_path)):
        # if i != 0 and i % gap != 0:
        #     continue

        # load the corresponding SAM segmentations data of the corresponding frame:
        points_data = torch.from_numpy(np.load(os.path.join(scene_output_path, 'points_npy', npy_file))).to(device)
        iou_preds_data = torch.from_numpy(np.load(os.path.join(scene_output_path, 'iou_preds_npy', npy_file))).to(device)
        masks_data = torch.from_numpy(np.load(os.path.join(scene_output_path, 'masks_npy', npy_file))).to(device)
        corre_3d_ins_data = torch.from_numpy(np.load(os.path.join(scene_output_path, 'corre_3d_ins_npy', npy_file))).to(device)  # the valid (i.e., has mapped pixels at the current frame) prompt ID  in the original 3D point cloud of initial prompts
        data = MaskData(
                masks=masks_data,
                iou_preds=iou_preds_data,
                points=points_data, 
                corre_3d_ins=corre_3d_ins_data)

        frame_id = npy_file[:-4]

        # calculate the 3d-2d mapping on ALL input points (not just prompt)
        association_pixel2point_path = os.path.join(scene_path, 'associations', 'pixel2point', frame_id + '.npy')
        association_point2pixel_path = os.path.join(scene_path, 'associations', 'point2pixel', frame_id + '.npy')
        association_pixel2point = np.load(association_pixel2point_path)
        association_point2pixel = np.load(association_point2pixel_path)

        # Number of 3D points
        n_points = association_point2pixel.shape[0]

        # Create an empty mapping array (N, 3) initialized with zeros
        mapping = np.zeros((n_points, 3), dtype=int)

        # Mask where pixel coordinates are valid (not -1)
        valid_mask = association_point2pixel[:, 0] != -1

        # Assign valid pixel coordinates
        mapping[valid_mask, 0] = association_point2pixel[valid_mask, 0]  # Y-coordinate
        mapping[valid_mask, 1] = association_point2pixel[valid_mask, 1]  # X-coordinate
        mapping[valid_mask, 2] = 1  # Mark as valid

        if mapping[:, 2].sum() == 0: # no points corresponds to this image, skip
            continue
        mapping = torch.from_numpy(mapping).to(device)

        keep_mask = torch.isin(data["corre_3d_ins"], keep_idx)  # only keep the mask that has been kept during previous prompt filter process
        data.filter(keep_mask)

        masks_logits = data["masks"]
        masks = masks_logits > 0.

        ins_idx_all = []
        for actual_idx in data["corre_3d_ins"]:  # the actual prompt ID in the original 3D point cloud of initial prompts, \
            # for calculating pt_score later (since pt_score is considered on all initial prompts)
            ins_idx = torch.where(keep_idx == actual_idx)[0]
            ins_idx_all.append(ins_idx.item())
        
        # when both a point i and a prompt j is found in this frame, counter[i, j] + 1
        counter_point = mapping[:, 2]   # the found points
        counter_point = counter_point.reshape(-1, 1).repeat(1, num_ins)
        counter_ins = torch.zeros(num_ins, device=device)
        counter_ins[ins_idx_all] += 1   # the found prompts
        counter_ins = counter_ins.reshape(1, -1).repeat(n_points, 1)
        counter_final += (counter_point * counter_ins)

        # caculate the score on mask area:
        for index, (mask) in enumerate(masks):  # iterate over each mask area segmented by different prompts
            ins_id = ins_idx_all[index]  # get the actual instance id  # ins_idx_al
            mask = mask.int()
        
            mask_2d_3d = mask[mapping[:, 0], mapping[:, 1]]
            mask_2d_3d = mask_2d_3d * mapping[:, 2]  # set the score to 0 if no mapping is found
            
            pt_score[:, ins_id] += mask_2d_3d  # For each individual input point in the scene, \
            # if it is projected within the mask area segmented by a prompt k at current frame, we assign its prediction as the prompt ID k

    pt_score_cpu = pt_score.cpu().numpy()
    counter_final_cpu = counter_final.cpu().numpy()
    counter_final_cpu[np.where(counter_final_cpu==0)] = -1  # avoid divided by zero

    pt_score_mean = pt_score_cpu / counter_final_cpu  # mean score denotes the probability of a point assigned to a specified prompt ID, and is only used for later thresholding
    pt_score_abs = pt_score_cpu
    max_score = np.max(pt_score_mean, axis=-1)  # the actual scores that has been segmented into one instance
    max_score_abs = np.max(pt_score_abs, axis=-1)

    # if pt_score_mean has the max value on more than one instance，we use the instance with higher pt_score as the pred
    max_indices_mean = np.where(pt_score_mean == max_score[:, np.newaxis])
    pt_score_mean_new = pt_score_mean.copy()   # only for calculate label, merge will still use pt_score_mean
    pt_score_mean_new[max_indices_mean] += pt_score_cpu[max_indices_mean]
    pt_pred_mean = np.argmax(pt_score_mean_new, axis=-1) # the ins index

    pt_pred_abs = np.argmax(pt_score_abs, axis=-1)

    low_pt_idx_mean = np.where(max_score <= 0.)[0]  # assign ins_label=-1 (unlabelled) if its score=0 (i.e., no 2D mask assigned)
    pt_score_mean[low_pt_idx_mean] = 0.
    pt_pred_mean[low_pt_idx_mean] = -1

    low_pt_idx_abs = np.where(max_score_abs <= 0.)[0]
    pt_score_abs[low_pt_idx_abs] = 0.
    pt_pred_abs[low_pt_idx_abs] = -1

    return pt_score_abs, pt_pred_abs, pt_score_mean


def prompt_consolidation(xyz, pt_score_abs, pt_pred_abs, pt_score_mean):
    pt_pred_final = pt_pred_abs.copy()

    # for each segmentated space, we first use DBSCAN to separate noisy predictions that are isolated in 3D space. (This aims to refine the SAM results)
    pt_score_merge = isolate_on_pred(xyz, pt_pred_abs.copy(), pt_score_abs.copy())
    pt_score_mean_ori = pt_score_mean.copy()
    pt_score_merge_ori = pt_score_merge.copy()

    # for each segmentated space, we again use DBSCAN to separate noisy score-level predictions (indicating a point has been segmented to a label at one frame) \
    # that are isolated in 3D space. (This aims to refine the SAM results)
    pt_score_merge = isolate_on_score(xyz, pt_score_mean_ori, pt_score_merge_ori)

    # only regard "confident" (label probability > 0.5) points as valid points belonging to an instance (or prompt) for consolidation:
    valid_thres = 0.5
    ins_areas = []
    ins_ids = []
    ins_score_mean = pt_score_mean.T
    ins_score = pt_score_merge.T
    for ins_id in range(ins_score.shape[0]):
        ins_area_mean = np.where(ins_score_mean[ins_id] >= valid_thres)[0]  # mean_score (probability) is only for thresholding more easily
        ins_area_abs = np.where(ins_score[ins_id] > 0)[0]
        ins_area = ins_area_abs[np.isin(ins_area_abs, ins_area_mean)]
        if ins_area.shape[0] > 0:
            ins_areas.append(ins_area)  # append the valid point idx of each instance/prompt
            ins_ids.append(ins_id)

    inter_all = []  # the intersection list to denote which prompts are segmenting the same 3D object
    for i in range(len(ins_areas)):
        inter_ins = [ins_ids[i]]
        for j in range(i+1, len(ins_areas)):
            inter = np.intersect1d(ins_areas[i], ins_areas[j])
            inter_ratio = inter.shape[0] / ins_areas[i].shape[0]
            if inter_ratio > 0.1:  # consider i and j are segmenting the same 3D object if have a certain overlap \
                # and append together in a sublist which are started from i:
                inter_ins.append(ins_ids[j])
            inter_all.append(inter_ins)

    consolidated_list = merge_common_values(inter_all)  # consolidate all prompts (i, j, k, ...) that are segmenting the same 3D object
    print("number of instances after Prompt Consolidation", len(consolidated_list))
        
    # Consolidate the result:
    for sublist in consolidated_list:
        for consolidate_id in sublist:
            mask = np.isin(pt_pred_final, sublist)
            pt_pred_final[mask] = sublist[0]  # regard the first prompt id as the pseudo prompt id

    return pt_pred_final


def merge_floor(pred_ins, floor_propose_ids, floor_id, scene_inter_thres):
    unique_pre_ins_ids = np.unique(pred_ins)
    for i in range(len(unique_pre_ins_ids)):
        if unique_pre_ins_ids[i] == -1:
            pre_instance_points_idx = np.where(pred_ins == unique_pre_ins_ids[i])[0]
            insection = np.isin(pre_instance_points_idx, floor_propose_ids) # the intersection between the floor and the predicted instance
            if sum(insection) > 0: 
                pred_ins[pre_instance_points_idx[insection]] = floor_id
            continue
        
        pre_instance_points_idx = np.where(pred_ins == unique_pre_ins_ids[i])[0]
        insection = sum(np.isin(pre_instance_points_idx, floor_propose_ids))  # the intersection between the floor and the predicted instance
     
        ratio = insection / len(pre_instance_points_idx)
        if ratio > scene_inter_thres:
            pred_ins[pre_instance_points_idx] = floor_id
            print(unique_pre_ins_ids[i])

    return pred_ins


def ransac_plane_seg(scene_plypath, pred_ins, floor_id, scene_dist_thres):
    point_cloud = o3d.io.read_point_cloud(scene_plypath)
    plane, inliers = point_cloud.segment_plane(distance_threshold=scene_dist_thres, ransac_n=3, num_iterations=1000)
    pred_ins[inliers] = floor_id

    return pred_ins

def sampro3d(scene_path = '../../data/kubric_0'):
    pred_iou_thres, stability_score_thres, box_nms_thres, keep_thres = 0.7, 0.6, 0.8, 0.4
    device = 'cuda:2'
    device = torch.device(device)
    sam = sam_model_registry['vit_h'](checkpoint="../../semantic_SfM/sam/sam_vit_h_4b8939.pth").to(device=device)
    predictor = SamPredictor(sam)

    
    print("Start loading SAM segmentations and other meta data ...")
    # Load the initial 3D input prompts (i.e., fps-sampled input points)
    data_folder = 'sampro3d'
    prompt_ply_file = os.path.join(scene_path, data_folder,  'init_prompt.ply')
    init_prompt, _ = load_ply(prompt_ply_file)
    print("the number of initial prompts", init_prompt.shape[0])
    
    # Load all 3D points of the input scene:
    pointcloud_path = os.path.join(scene_path, 'reconstructions', 'combined_point_cloud.las')

    with laspy.open(pointcloud_path) as las_file:
        header = las_file.header
        N_points = header.point_count
    
    pc = laspy.read(pointcloud_path)
    xyz = np.vstack((pc.x, pc.y, pc.z)).T
    rgb = np.vstack((pc.red, pc.green, pc.blue)).T

    
    # Load SAM segmentations generated by previous 3D Prompt Proposal: 
    scene_output_path = os.path.join(scene_path, data_folder)
    points_npy_path = natsorted(os.listdir(os.path.join(scene_output_path, 'points_npy')))
    iou_preds_npy_path = natsorted(os.listdir(os.path.join(scene_output_path, 'iou_preds_npy')))
    masks_npy_path = natsorted(os.listdir(os.path.join(scene_output_path, 'masks_npy')))
    corre_3d_ins_npy_path = natsorted(os.listdir(os.path.join(scene_output_path, 'corre_3d_ins_npy')))
    pred_path = os.path.join(scene_output_path, 'prediction')
    assert(points_npy_path == iou_preds_npy_path == masks_npy_path == corre_3d_ins_npy_path)
    print("Finished loading SAM segmentations and other meta data!")
    print("********************************************************")
    
    #"""
    # 2D-Guided Prompt Filter:
    print("Start 2D-Guided Prompt Filter ...")
    # points_npy_path = points_npy_path[:10]
    keep_idx = prompt_filter(init_prompt, scene_output_path, points_npy_path, predictor, device, pred_iou_thres, stability_score_thres, box_nms_thres, keep_thres)
    del predictor
    # save keep_idx
    np.save(os.path.join(scene_output_path, 'keep_idx.npy'), keep_idx.cpu().numpy())
    # pt_filtered = pt_init[keep_idx.clone().cpu().numpy()]
    print("Finished 2D-Guided Prompt Filter!")
    print("********************************************************")
    
    # load keep_idx
    keep_idx = np.load(os.path.join(scene_output_path, 'keep_idx.npy'))
    keep_idx = torch.from_numpy(keep_idx).to(device)
    # Now we need to perform 3D segmentation to get the initial segmentation label and per-point segmentation score, aimming to check if they are segmenting the same 3D object:
    print("Start initial 3D segmentation ...")
    pt_score_abs, pt_pred_abs, pt_score_mean = perform_3dsegmentation0(xyz, keep_idx, scene_output_path, points_npy_path, device, scene_path)
    print("Finished initial 3D segmentation!")
    print("********************************************************")
    
    
    # Prompt Consolidation:
    print("Start Prompt Consolidation and finalizing 3D Segmentation ...")
    pt_pred = prompt_consolidation(xyz, pt_score_abs, pt_pred_abs, pt_score_mean)
    print("Finished running the whole SAMPro3D!")
    print("********************************************************")
    
    pt_pred = num_to_natural(pt_pred)
    create_folder(pred_path)
    # save the prediction result:
    pred_file = os.path.join(pred_path, 'seg.npy')
    np.save(pred_file, pt_pred)
    #"""
    
    # load the prediction result:
    pred_file = os.path.join(pred_path, 'seg.npy')
    pt_pred = np.load(pred_file)


    # add semantics to the point cloud
    # add_semantics_to_pointcloud(pointcloud_path, pred_file, os.path.join(scene_output_path, 'seg.las'))
    add_semantics_to_pointcloud(pointcloud_path, pred_file, os.path.join(scene_output_path, 'seg.las'), remove_small_N=500, nearest_interpolation=500)
    

def sampro3d_gd():
    scene_path = '../../data/granite_dells'
    pred_iou_thres, stability_score_thres, box_nms_thres, keep_thres = 0.7, 0.6, 0.8, 0.4
    device = 'cuda:2'
    device = torch.device(device)
    sam = sam_model_registry['vit_h'](checkpoint="../../semantic_SfM/sam/sam_vit_h_4b8939.pth").to(device=device)
    predictor = SamPredictor(sam)

    
    print("Start loading SAM segmentations and other meta data ...")
    # Load the initial 3D input prompts (i.e., fps-sampled input points)
    data_folder = 'sampro3d'
    prompt_ply_file = os.path.join(scene_path, data_folder,  'init_prompt.ply')
    print(prompt_ply_file)
    init_prompt, _ = load_ply(prompt_ply_file)
    print("the number of initial prompts", init_prompt.shape[0])
    
    # Load all 3D points of the input scene:
    pointcloud_path = os.path.join(scene_path, 'SfM_products', 'downsampled_sampro3d.las')

    with laspy.open(pointcloud_path) as las_file:
        header = las_file.header
        N_points = header.point_count
    
    pc = laspy.read(pointcloud_path)
    xyz = np.vstack((pc.x, pc.y, pc.z)).T
    rgb = np.vstack((pc.red, pc.green, pc.blue)).T

    
    # Load SAM segmentations generated by previous 3D Prompt Proposal: 
    scene_output_path = os.path.join(scene_path, data_folder)
    points_npy_path = natsorted(os.listdir(os.path.join(scene_output_path, 'points_npy')))
    iou_preds_npy_path = natsorted(os.listdir(os.path.join(scene_output_path, 'iou_preds_npy')))
    masks_npy_path = natsorted(os.listdir(os.path.join(scene_output_path, 'masks_npy')))
    corre_3d_ins_npy_path = natsorted(os.listdir(os.path.join(scene_output_path, 'corre_3d_ins_npy')))
    pred_path = os.path.join(scene_output_path, 'prediction')
    assert(points_npy_path == iou_preds_npy_path == masks_npy_path == corre_3d_ins_npy_path)
    print("Finished loading SAM segmentations and other meta data!")
    print("********************************************************")
    
    #"""
    # 2D-Guided Prompt Filter:
    print("Start 2D-Guided Prompt Filter ...")
    # points_npy_path = points_npy_path[:10]
    keep_idx = prompt_filter(init_prompt, scene_output_path, points_npy_path, predictor, device, pred_iou_thres, stability_score_thres, box_nms_thres, keep_thres)
    del predictor
    # save keep_idx
    np.save(os.path.join(scene_output_path, 'keep_idx.npy'), keep_idx.cpu().numpy())
    # pt_filtered = pt_init[keep_idx.clone().cpu().numpy()]
    print("Finished 2D-Guided Prompt Filter!")
    print("********************************************************")
    
    # load keep_idx
    keep_idx = np.load(os.path.join(scene_output_path, 'keep_idx.npy'))
    keep_idx = torch.from_numpy(keep_idx).to(device)
    # Now we need to perform 3D segmentation to get the initial segmentation label and per-point segmentation score, aimming to check if they are segmenting the same 3D object:
    print("Start initial 3D segmentation ...")
    pt_score_abs, pt_pred_abs, pt_score_mean = perform_3dsegmentation0(xyz, keep_idx, scene_output_path, points_npy_path, device, scene_path)
    print("Finished initial 3D segmentation!")
    print("********************************************************")
    
    
    # Prompt Consolidation:
    print("Start Prompt Consolidation and finalizing 3D Segmentation ...")
    pt_pred = prompt_consolidation(xyz, pt_score_abs, pt_pred_abs, pt_score_mean)
    print("Finished running the whole SAMPro3D!")
    print("********************************************************")
    
    pt_pred = num_to_natural(pt_pred)
    create_folder(pred_path)
    # save the prediction result:
    pred_file = os.path.join(pred_path, 'seg.npy')
    np.save(pred_file, pt_pred)
    #"""
    
    # load the prediction result:
    pred_file = os.path.join(pred_path, 'seg.npy')
    pt_pred = np.load(pred_file)


    # add semantics to the point cloud
    # add_semantics_to_pointcloud(pointcloud_path, pred_file, os.path.join(scene_output_path, 'seg.las'))
    add_semantics_to_pointcloud(pointcloud_path, pred_file, os.path.join(scene_output_path, 'seg.las'), remove_small_N=500, nearest_interpolation=500)

if __name__ == "__main__":
    None
    """
    scene_path = '../../data/kubric_0'
    sampro3d(scene_path)
    """

    """
    scene_paths = [os.path.join('../../data', f) for f in os.listdir('../../data') if "kubric" in f]
    scene_paths_valid = []
    for scene_path in scene_paths:
        if os.path.exists(os.path.join(scene_path, 'sampro3d')):
            scene_paths_valid.append(scene_path)
    for scene_path in scene_paths_valid:
        print(scene_path)
        sampro3d(scene_path)
    """
    
    sampro3d_gd()
        