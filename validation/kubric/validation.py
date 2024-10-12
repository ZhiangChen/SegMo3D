import laspy
import numpy as np
import os
import time
from joblib import Parallel, delayed

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

            
        
        

if __name__ == "__main__":
    # validate
    validator = Validator(
        "../../data/kubric_0/associations/semantics/semantics_284_shuffled.las",
        "../../data/kubric_0/reconstructions/combined_point_cloud.las"
    )

    results = validator.validate(np.arange(0.5, 1.0, 0.05))
    mAP = results['AP']
    mAR = results['AR']
    print('mAP list: ', mAP, ' mAR list: ', mAR)
    mAP = np.sum(mAP) / len(mAP)
    mAR = np.sum(mAR) / len(mAR)
    print('mAP: ', mAP, ' mAR: ', mAR)