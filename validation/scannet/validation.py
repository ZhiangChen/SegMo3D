import numpy as np
import os
import laspy

class ScannetValidation(object):
    def __init__(self) -> None:
        pass

    def read_scene(self, scene_folder_path, pred_file_path=None, remove_background_in_prediction=True):
        assert os.path.exists(scene_folder_path), 'Scene folder does not exist!'

        # read the ground truth
        ground_truth_file_path = os.path.join(scene_folder_path, 'associations', 'semantic_points.npy')
        ground_truth = np.load(ground_truth_file_path)

        # get the prediction files
        prediction_folder_path = os.path.join(scene_folder_path, 'associations', 'semantics')
        prediction_files = [os.path.join(prediction_folder_path, file) for file in os.listdir(prediction_folder_path) if file.endswith('.npy')]

        # sort prediction files based on the number in the file name: semantics_0.npy, semantics_1.npy, ...
        prediction_files = sorted(prediction_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        if pred_file_path is not None:
            prediction_file_path = pred_file_path
            if not os.path.exists(prediction_file_path):
                print('Invalid prediction file path')
                prediction_file_path = prediction_files[-1]
            else:
                if prediction_file_path.endswith('.npy'):
                    raw_prediction = np.load(prediction_file_path)
                elif prediction_file_path.endswith('.las'):
                    raw_prediction = laspy.read(prediction_file_path).intensity

        else:
            prediction_file_path = prediction_files[-1]
            raw_prediction = np.load(prediction_file_path)
        
        
        # get the unique semantics 
        unique_semantics = np.unique(raw_prediction)
        # sort the unique semantics
        unique_semantics = np.sort(unique_semantics)
        N_unique_semantics = unique_semantics.shape[0]

        prediction = np.zeros_like(raw_prediction)
        for i, semantic in enumerate(unique_semantics):
            if semantic == -1:
                prediction[raw_prediction == semantic] = N_unique_semantics - 1
            else:
                prediction[raw_prediction == semantic] = i-1

        if remove_background_in_prediction:
            # get the indices of the background points in the prediction 
            background_indices = np.where(prediction == N_unique_semantics - 1)
            # remove the background points from the prediction
            prediction = np.delete(prediction, background_indices, axis=0)
            # remove the background points from the ground truth
            ground_truth = np.delete(ground_truth, background_indices, axis=0)

        # get the indices of the points that are not in the ground truth
        indices = np.where(ground_truth == np.max(ground_truth))
        # remove the points that are not in the ground truth from the prediction
        prediction = np.delete(prediction, indices, axis=0)
        # remove the points that are not in the ground truth from the ground truth
        ground_truth = np.delete(ground_truth, indices, axis=0)

        self.prediction = prediction
        self.ground_truth = ground_truth

    def __association(self):
        N_labels = np.max(self.ground_truth) + 1
        ground_truth_associated = np.zeros_like(self.ground_truth)

        for i in range(N_labels):
            indices = np.where(self.ground_truth == i)
            if indices[0].shape[0] > 0:
                prediction_labels = self.prediction[indices]
                unique, counts = np.unique(prediction_labels, return_counts=True)
                associated_label = unique[np.argmax(counts)]
                ground_truth_associated[indices] = associated_label

        return ground_truth_associated

    def evaluate(self):
        ground_truth_associated = self.__association()

        unique, counts = np.unique(ground_truth_associated, return_counts=True)
        print('Unique labels in the ground truth: ', unique)
        print('Counts of unique labels in the ground truth: ', counts)
        print('Number of unique labels in the ground truth: ', unique.shape[0])
        # we compare ground_truth_associated with self.prediction to calculate the accuracy, precision, and f1-score
        TP = np.sum(ground_truth_associated == self.prediction)
        FP = np.sum(ground_truth_associated != self.prediction)

        accuracy = TP / (TP + FP)
        precision = TP / (TP + FP)   
        f1_score = 2 * precision * accuracy / (precision + accuracy)     
        print('Accuracy: ', accuracy)
        print('Precision: ', precision)
        print('F1-score: ', f1_score)


    def evaluate2(self, iou_threshold=0.5):
        """
        This function calculates the intersection over union (IoU) between the ground truth and the prediction to calculate average precision (AP)
        """ 
        ground_truth_associated = self.__association()
        # get unique labels in the ground truth
        unique_labels = np.unique(ground_truth_associated)

        # calculate the intersection over union (IoU) between the ground truth and the prediction
        TP = 0
        FP = 0
        for label in unique_labels:
            # get the indices of the label in the ground truth
            indices = np.where(ground_truth_associated == label)
            # get the indices of the label in the prediction
            prediction_indices = np.where(self.prediction == label)
            # calculate the intersection
            intersection = np.intersect1d(indices, prediction_indices)
            # calculate the union
            union = np.union1d(indices, prediction_indices)
            # calculate the IoU
            IoU = intersection.shape[0] / union.shape[0]
            #print('IoU for label {}: '.format(label), IoU)
            if IoU >= iou_threshold:
                TP += 1
            else:
                FP += 1
        
        # calculate average precision (AP)
        AP = TP / (TP + FP)
        print('Average precision (AP): ', AP)

    
if __name__ == '__main__':
    validation = ScannetValidation()
    """
    # without nearest interpolation
    validation.read_scene('../../data/scene0000_00', remove_background_in_prediction=False)
    validation.evaluate()
    # with nearest interpolation
    validation.read_scene('../../data/scene0000_00', '../../data/scene0000_00/associations/semantics/semantics_613_shuffled.las', remove_background_in_prediction=False)
    validation.evaluate()
    """
    # with nearest interpolation and filter
    validation.read_scene('../../data/scene0000_00', '../../data/scene0000_00/associations/semantics/semantic_613_interpolated_shuffled_filtered.las', remove_background_in_prediction=False)
    validation.evaluate()
    validation.evaluate2()