import laspy
import numpy as np
import os
import time
from joblib import Parallel, delayed
import csv
from scipy.spatial import cKDTree

class Validator(object):
    def __init__(self, prediction_path, ground_truth_path):
        # assert files exist
        assert os.path.exists(prediction_path), "Prediction file does not exist"
        assert os.path.exists(ground_truth_path), "Ground truth file does not exist"

        # read files
        pc_prediction = laspy.read(prediction_path)
        pc_ground_truth = laspy.read(ground_truth_path)

        # get semantics from intensity
        self.prediction_semantics = pc_prediction.intensity
        self.ground_truth_semantics = pc_ground_truth.intensity

        # print length of unique semantics
        print("Unique semantics in prediction: ", len(np.unique(self.prediction_semantics)))
        print("Unique semantics in ground truth: ", len(np.unique(self.ground_truth_semantics)))

    def _associate(self, semantics1, semantics2, lower_bound_iou=0.8):
        # get all unique semantics in semantics1
        unique_semantics1 = np.unique(semantics1)

        # Use joblib to parallelize the loop over unique semantics
        associations = Parallel(n_jobs=4)(delayed(self._find_best_semantic2)(
            semantic1, semantics1, semantics2, lower_bound_iou) for semantic1 in unique_semantics1)

        # Create a dictionary of associations
        return {semantic1: best_semantic2 for semantic1, best_semantic2 in associations}

    def _find_best_semantic2(self, semantic1, semantics1, semantics2, lower_bound_iou):
        """
        Helper function to find the best matching semantic2 for a given semantic1.
        This function will be parallelized.
        """
        # get indices of all points with semantic1 in semantics1
        indices1 = np.where(semantics1 == semantic1)[0]
        # get all semantics in semantics2 at these indices
        semantics2_at_semantic1 = semantics2[indices1]
        # get unique semantics in semantics2_at_semantic1
        unique_semantics2_at_semantic1 = np.unique(semantics2_at_semantic1)

        best_semantic2 = None
        for semantic2 in unique_semantics2_at_semantic1:
            # get indices of all points with semantic2 in semantics2
            indices2 = np.where(semantics2 == semantic2)[0]
            iou = self._calculate_iou(indices1, indices2)
            if iou > lower_bound_iou:
                lower_bound_iou = iou
                best_semantic2 = semantic2

        return semantic1, best_semantic2

    def _calculate_iou(self, indices1, indices2):
        """
        Calculate Intersection over Union (IoU) between two sets of points

        Args:
            indices1: indices of points in 1d numpy array
            indices2: indices of points in 1d numpy array
        """
        # get intersection
        intersection = np.intersect1d(indices1, indices2)
        # get union
        union = np.union1d(indices1, indices2)
        # calculate IoU
        iou = len(intersection) / len(union)
        return iou

    def validate(self, iou_list):
        results = {}
        AP_list = []
        AR_list = []
        TP_list = []
        FP_list = []
        FN_list = []

        for lower_bound_iou in iou_list:
            association1 = self._associate(self.prediction_semantics, self.ground_truth_semantics, lower_bound_iou)

            associated_gt = np.array([association1[semantic] for semantic in self.prediction_semantics if association1[semantic] is not None])
            associated_gt = np.unique(associated_gt)
            unassociated_gt = np.setdiff1d(np.unique(self.ground_truth_semantics), associated_gt)
            
            TP = len(associated_gt)
            FP = len([x for x in association1.values() if x is None])
            FN = len(unassociated_gt)
            
            AP = TP / (TP + FP)
            AR = TP / (TP + FN)
            
            AP_list.append(AP)
            AR_list.append(AR)
            TP_list.append(TP)
            FP_list.append(FP)
            FN_list.append(FN)

        results['AP'] = AP_list
        results['AR'] = AR_list
        results['TP'] = TP_list
        results['FP'] = FP_list
        results['FN'] = FN_list

        return results

            
def batch_validation():
    with open('results.csv', mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['Scene', 'AP_50', 'AP_55', 'AP_60', 'AP_65', 'AP_70', 'AP_75', 'AP_80', 'AP_85', 'AP_90', 'AP_95', 'mAP', 'AR_50', 'AR_55', 'AR_60', 'AR_65', 'AR_70', 'AR_75', 'AR_80', 'AR_85', 'AR_90', 'AR_95', 'mAR'])
    scene_paths = [os.path.join('../../data', f) for f in os.listdir('../../data') if "kubric" in f]
    scene_paths_valid = []
    for scene_path in scene_paths:
        if os.path.exists(os.path.join(scene_path, 'associations', 'merged_pcd')):
            scene_paths_valid.append(scene_path)
    scene_paths = scene_paths_valid
    print("Total scenes: ", len(scene_paths))
    
    for scene_path in scene_paths:
        prediction_path = os.path.join(scene_path, "associations/sam3d_no_floor.las")
        ground_truth_path = os.path.join(scene_path, 'reconstructions/combined_point_cloud.las')
        print("Validation for scene: ", scene_path)
        validator = Validator(prediction_path, ground_truth_path)
        results = validator.validate(np.arange(0.5, 1.0, 0.05))
        mAP = results['AP']
        mAR = results['AR']
        print('mAP list: ', mAP, ' mAR list: ', mAR)
        mAP = np.sum(mAP) / len(mAP)
        mAR = np.sum(mAR) / len(mAR)
        print('mAP: ', mAP, ' mAR: ', mAR)
        with open('results.csv', mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([scene_path, *results['AP'], mAP, *results['AR'], mAR])

    # read the csv file 
    mAP_list = []
    mAR_list = []
    with open('results.csv', mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            # ['Scene', 'AP_50', 'AP_55', 'AP_60', 'AP_65', 'AP_70', 'AP_75', 'AP_80', 'AP_85', 'AP_90', 'AP_95', 'mAP', 'AR_50', 'AR_55', 'AR_60', 'AR_65', 'AR_70', 'AR_75', 'AR_80', 'AR_85', 'AR_90', 'AR_95', 'mAR']
            if row[0] == 'Scene':
                continue
            mAP_list.append(float(row[11]))
            mAR_list.append(float(row[22]))

    # calculate the average mAP and mAR
    mAP = np.sum(mAP_list) / len(mAP_list)
    mAR = np.sum(mAR_list) / len(mAR_list)
    print('Average mAP: ', mAP)
    print('Average mAR: ', mAR)


def calculate_table():
    # open the csv file
    with open('results.csv', mode='r') as file:
        reader = csv.reader(file)
        # read the header
        header = next(reader)
        # get the number of columns
        num_columns = len(header)
        # ['Scene', 'AP_50', 'AP_55', 'AP_60', 'AP_65', 'AP_70', 'AP_75', 'AP_80', 'AP_85', 'AP_90', 'AP_95', 'mAP', 'AR_50', 'AR_55', 'AR_60', 'AR_65', 'AR_70', 'AR_75', 'AR_80', 'AR_85', 'AR_90', 'AR_95', 'mAR']
        # create a list 
        values_list = []
        # read the rows
        for row in reader:
            values = []
            # skip the first row
            if row[0] == 'Scene':
                continue
            # append the row to the values
            # iterate over each column
            for i in range(1, num_columns):
                values.append(float(row[i]))
            values_list.append(values)
        # convert the list to numpy array
        values_list = np.array(values_list)
        # calculate the average
        average = np.mean(values_list, axis=0)
        keys = ['AP_50', 'AP_55', 'AP_60', 'AP_65', 'AP_70', 'AP_75', 'AP_80', 'AP_85', 'AP_90', 'AP_95', 'mAP', 'AR_50', 'AR_55', 'AR_60', 'AR_65', 'AR_70', 'AR_75', 'AR_80', 'AR_85', 'AR_90', 'AR_95', 'mAR']
        for i in range(len(keys)):
            print(keys[i], " : ", average[i])


def sample_ground_truth(prediction_las, gt_las):
    prediction_las = laspy.read(prediction_las)
    prediction_x = prediction_las.x
    prediction_y = prediction_las.y
    prediction_z = prediction_las.z
    
    gt_las = laspy.read(gt_las)
    gt_semantics = gt_las.user_data
    gt_x = gt_las.x
    gt_y = gt_las.y
    gt_z = gt_las.z
    gt_red = gt_las.red
    gt_green = gt_las.green
    gt_blue = gt_las.blue

    # for each point in prediction, find the nearest point in ground truth using kd-tree, and record the point indices
    kd_tree = cKDTree(np.column_stack((gt_x, gt_y, gt_z)))
    indices = kd_tree.query(np.column_stack((prediction_x, prediction_y, prediction_z)))[1]

    points = np.column_stack((gt_x[indices], gt_y[indices], gt_z[indices]))
    colors = np.column_stack((gt_red[indices], gt_green[indices], gt_blue[indices]))

    # write the sampled ground truth to a new las file
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


    las.intensity = gt_semantics[indices]


    las.write('sampled_ground_truth.las')


def validate_gd(prediction_path, ground_truth_path):
    validator = Validator(prediction_path, ground_truth_path)
    results = validator.validate(np.arange(0.5, 1.0, 0.05))
    mAP = results['AP']
    mAR = results['AR']
    print('mAP list: ', mAP, ' mAR list: ', mAR)
    mAP = np.sum(mAP) / len(mAP)
    mAR = np.sum(mAR) / len(mAR)
    print('mAP: ', mAP, ' mAR: ', mAR)



if __name__ == "__main__":
    #calculate_table()
    prediction_las = '../../data/granite_dells/associations/sam3d_refine.las'
    gt_las = '../../data/granite_dells/associations/semantics.las'
    assert os.path.exists(prediction_las), "Prediction file does not exist"
    assert os.path.exists(gt_las), "Ground truth file does not exist"
    #sample_ground_truth(prediction_las, gt_las)
    gt_las = 'sampled_ground_truth.las'
    validate_gd(prediction_las, gt_las)
    