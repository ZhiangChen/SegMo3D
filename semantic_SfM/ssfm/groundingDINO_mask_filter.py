import os
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from torchvision.ops import box_convert
import torch
from ssfm.files import read_camera_parameters_webodm, read_camera_parameters_agisoft
from joblib import Parallel, delayed

import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, predict, annotate





class GroundingDINOMaskFilter(object):
    def __init__(self, mask_folder_path, save_folder_path):
        self.mask_folder_path = mask_folder_path
        self.save_folder_path = save_folder_path

        assert os.path.exists(self.mask_folder_path), "The mask folder does not exist: {}".format(self.mask_folder_path)
        if not os.path.exists(self.save_folder_path):
            os.makedirs(self.save_folder_path)
        
        self.mask_lists = [os.path.join(self.mask_folder_path, f) for f in os.listdir(self.mask_folder_path) if f.endswith('.npy')]
        self.mask_lists.sort()

        self.distortion_params = None
        self.matrix_intrinsics = None
        self.keep_background = True

    def set_distortion_correction(self, camera_parameters_path, additional_camera_parameters_path=None):
        """
        Arguments:
            camera_parameters_path (str): Path to the camera parameters file.
            additional_camera_parameters_path (str): Path to the additional camera parameters file.
        """
        if additional_camera_parameters_path is not None:
            # WebODM
            cameras = read_camera_parameters_webodm(camera_parameters_path, additional_camera_parameters_path)
        else:
            # Agisoft
            cameras = read_camera_parameters_agisoft(camera_parameters_path)

        self.distortion_params = cameras['distortion_params']
        self.matrix_intrinsics = cameras['K']
        self.width = cameras['width']
        self.height = cameras['height']
        
    def load_image(self, image_path):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_source = Image.open(image_path).convert("RGB")
        image = np.asarray(image_source)
        if self.distortion_params is not None:
            image = cv2.undistort(image, self.matrix_intrinsics, self.distortion_params)
        image_source = Image.fromarray(image)
        image_transformed, _ = transform(image_source, None)
        return image, image_transformed

    def predict_bounding_boxes(self, image_folder_path, grounding_dino_config):
        # get image paths
        assert os.path.exists(image_folder_path), "The image folder does not exist: {}".format(image_folder_path)
        image_lists = [os.path.join(image_folder_path, f) for f in os.listdir(image_folder_path) if f.endswith('.JPG')]
        image_lists.sort()
        # print the number of images
        print("The number of images: {}".format(len(image_lists)))
        if len(image_lists) == 0:
            # print the error message
            print("There is no image in the folder: {}".format(image_folder_path))
            return
        
        else:
            weights_path = grounding_dino_config['weights_path']
            config_path = grounding_dino_config['config_path']
            text_prompt = grounding_dino_config['text_prompt']
            box_treshold = grounding_dino_config['box_treshold']
            text_treshold = grounding_dino_config['text_treshold']
            device = grounding_dino_config['device']
            prediction_save_folder_path = grounding_dino_config['prediction_save_folder_path']
            remove_background = grounding_dino_config['remove_background']

            if not os.path.exists(prediction_save_folder_path):
                os.makedirs(prediction_save_folder_path)

            # load the model
            assert os.path.exists(weights_path), "The weights file does not exist: {}".format(weights_path)
            assert os.path.exists(config_path), "The config file does not exist: {}".format(config_path)
            model = load_model(config_path, weights_path, device)

            for i in tqdm(range(len(image_lists))):
                image_path = image_lists[i]
                image_source, image = self.load_image(image_path)

                # predict the bounding boxes
                # boxes: [N, 4] where N is the number of boxes
                # logits: [N] scores of the boxes
                # phrases: [N] the predicted phrases of the boxes
                boxes, logits, phrases = predict(model=model, 
                                                 image=image, 
                                                 caption=text_prompt, 
                                                 box_threshold=box_treshold, 
                                                 text_threshold=text_treshold)
                
                if remove_background:
                    boxes_width = boxes[:, 2]
                    boxes_height = boxes[:, 3]
                    boxes_area = boxes_width * boxes_height
                    boxes = boxes[boxes_area < 0.9]
                    logits = logits[boxes_area < 0.9]

                annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

                # save the annotated frame
                image_name = os.path.basename(image_path)
                save_path = os.path.join(prediction_save_folder_path, image_name)
                cv2.imwrite(save_path, annotated_frame)

                # boxes have the format of [cx, cy, w, h] and need to be converted to [x1, y1, x2, y2]
                boxes = boxes * torch.Tensor([self.width, self.height, self.width, self.height])
                boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
                # concatenate the boxes and logits
                logits = logits.reshape(-1, 1)
                prediction = np.concatenate((boxes, logits), axis=1)
                # save the prediction
                prediction_name = image_name.split('.')[0] + '.npy'
                prediction_path = os.path.join(prediction_save_folder_path, prediction_name)
                np.save(prediction_path, prediction)


    def filter(self, mask_bbox_file):
        mask_file, bbox_file = mask_bbox_file
        # load the mask and bounding box
        masks = np.load(mask_file)
        boxes = np.load(bbox_file)
        scores = boxes[:, 4]
        boxes = boxes[:, :4]
        #print(f"Number of masks: {np.max(masks)+1}")
        #print(f"Number of bounding boxes: {len(boxes)}")
        valid_mask_ids = []
        # create an empty image to draw bounding boxes
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        if self.keep_background:
            N = np.max(masks)
            valid_mask_ids.append(np.max(masks))
        else:
            N = np.max(masks) + 1
        # iterate over the masks
        for i in range(N):
            # get bounding box of the mask
            mask = (masks == i).astype(np.uint8)
            x, y, w, h = cv2.boundingRect(mask)
            # Extract coordinates
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]
            # Calculate the coordinates of the intersection rectangle
            xA = np.maximum(x, x1)
            yA = np.maximum(y, y1)
            xB = np.minimum(x + w, x2)
            yB = np.minimum(y + h, y2)
            # Compute the area of intersection rectangle
            interArea = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
            # Compute the area of mask rectangles
            maskArea = w * h
            # Compute the IoU
            coverage = interArea / maskArea
            if len(coverage) == 0:
                continue
            # get the index of the maximum coverage
            max_coverage_idx = np.argmax(coverage)
            # get the maximum coverage
            max_coverage = coverage[max_coverage_idx]
            if max_coverage > 0.2:
                valid_mask_ids.append(i)
                # draw the mask
                image[mask == 1] = (0, 255, 0)
                # draw the bounding box of the box with the maximum coverage
                x1 = int(x1[max_coverage_idx])
                y1 = int(y1[max_coverage_idx])
                x2 = int(x2[max_coverage_idx])
                y2 = int(y2[max_coverage_idx])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 4)

        #print(f"Number of valid masks: {len(valid_mask_ids)}")
        # filter the masks
        masks_filtered = np.zeros_like(masks)-1
        for i, mask_id in enumerate(valid_mask_ids):
            masks_filtered[masks == mask_id] = i
        # save the filtered masks
        save_path = mask_file.replace(self.mask_folder_path, self.save_folder_path)
        np.save(save_path, masks_filtered)
        # save the image
        image_save_path = save_path.replace('.npy', '.png')
        cv2.imwrite(image_save_path, image)
                    


    def filter_batch(self, grounding_dino_prediction_folder_path, num_processes=8, keep_background=True):
        self.keep_background = keep_background
        assert os.path.exists(grounding_dino_prediction_folder_path), "The folder {} does not exist".format(grounding_dino_prediction_folder_path)
        self.bounding_box_files = [os.path.join(grounding_dino_prediction_folder_path, f) for f in os.listdir(grounding_dino_prediction_folder_path) if f.endswith('.npy')]
        self.bounding_box_files.sort()

        # assert the number of files
        assert len(self.bounding_box_files) == len(self.mask_lists), "The number of files does not match: {} != {}".format(len(self.bounding_box_files), len(self.masks_files))
        # create paris of masks_files and bounding_box_files
        self.mask_bbox_files = list(zip(self.mask_lists, self.bounding_box_files))

        # print the number of files
        print(f"Total number of files: {len(self.bounding_box_files)}")

        Parallel(n_jobs=num_processes)(delayed(self.filter)(f) for f in tqdm(self.mask_bbox_files))



if __name__ == "__main__":
    grounding_dino_config = {}
    grounding_dino_config['weights_path'] = "../grounding_DINO/groundingdino_swint_ogc.pth"
    grounding_dino_config['config_path'] = "../grounding_DINO/GroundingDINO_SwinT_OGC.py"
    grounding_dino_config['prediction_save_folder_path'] = "../../data/courtright/grounding_dino_predictions"
    grounding_dino_config['text_prompt'] = "rock"
    grounding_dino_config['box_treshold'] = 0.15
    grounding_dino_config['text_treshold'] = 0.25
    grounding_dino_config['device'] = 'cuda:5'
    grounding_dino_config['remove_background'] = True

    mask_folder_path = "../../data/courtright/segmentations_filtered"
    save_folder_path = "../../data/courtright/segmentations_filtered_semantics"
    image_folder_path = "../../data/courtright/DJI_photos"

    mask_filter = GroundingDINOMaskFilter(mask_folder_path, save_folder_path)
    mask_filter.set_distortion_correction('../../data/courtright/SfM_products/agisoft_cameras.xml')

    #mask_filter.predict_bounding_boxes(image_folder_path, grounding_dino_config)

    grounding_dino_prediction_folder_path = grounding_dino_config['prediction_save_folder_path'] 
    
    #mask_filter.filter(('../../data/courtright/segmentations_filtered/DJI_0576.npy', '../../data/courtright/grounding_dino_predictions/DJI_0576.npy'))
    mask_filter.filter_batch(grounding_dino_prediction_folder_path)
    