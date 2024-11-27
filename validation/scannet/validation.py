import numpy as np
import os
import laspy
from joblib import Parallel, delayed

class ScannetValidation(object):
    def __init__(self, scene_folder_path, prediction_path=None) -> None:
        assert os.path.exists(scene_folder_path), 'Scene folder does not exist!'

        # read the ground truth
        ground_truth_file_path = os.path.join(scene_folder_path, 'associations', 'semantic_points.npy')
        ground_truth = np.load(ground_truth_file_path)

        print('Unique semantics in the ground truth: ', np.unique(ground_truth).shape[0])


        # get the prediction files
        prediction_folder_path = os.path.join(scene_folder_path, 'associations', 'semantics')
        prediction_files = [os.path.join(prediction_folder_path, file) for file in os.listdir(prediction_folder_path) if file.endswith('.npy')]

        # sort prediction files based on the number in the file name: semantics_0.npy, semantics_1.npy, ...
        prediction_files = sorted(prediction_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        if prediction_path is not None:
            prediction_file_path = prediction_path
            if not os.path.exists(prediction_file_path):
                print('Invalid prediction file path')
                prediction_file_path = prediction_files[-1]
                print('Using the last prediction file: ', prediction_file_path)
                raw_prediction = np.load(prediction_file_path)
            else:
                print('Using the provided prediction file: ', prediction_file_path)
                if prediction_file_path.endswith('.npy'):
                    raw_prediction = np.load(prediction_file_path)
                elif prediction_file_path.endswith('.las'):
                    raw_prediction = laspy.read(prediction_file_path).intensity


        else:
            prediction_file_path = prediction_files[-1][:-4] + '_shuffled.las'
            print('Using the last prediction file: ', prediction_file_path)
            raw_prediction = laspy.read(prediction_file_path).intensity

        # get the unique semantics
        unique_semantics_prediction = np.unique(raw_prediction)
        unique_semantics_ground_truth = np.unique(ground_truth)
        print('Number of unique semantics in the prediction: ', unique_semantics_prediction.shape[0])
        print('Number of unique semantics in the ground truth: ', unique_semantics_ground_truth.shape[0])
        self.prediction_semantics = raw_prediction
        self.ground_truth_semantics = ground_truth

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

            print('TP: ', TP, ' FP: ', FP, ' FN: ', FN)
            
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

    
if __name__ == '__main__':
    scene_folder_path = '../../data/scene0000_00'
    validator = ScannetValidation(scene_folder_path)
    results = validator.validate(np.arange(0.5, 1.0, 0.05))
    #results = validator.validate([0.5, 0.6])
    mAP = results['AP']
    mAR = results['AR']
    print('mAP list: ', mAP, ' mAR list: ', mAR)
    mAP = np.sum(mAP) / len(mAP)
    mAR = np.sum(mAR) / len(mAR)
    print('mAP: ', mAP, ' mAR: ', mAR)
    