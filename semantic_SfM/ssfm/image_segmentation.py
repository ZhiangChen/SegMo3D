from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os
import cv2
import pickle
import numpy as np

"""
SamAutomaticMaskGenerator
    Arguments:
        model (Sam): The SAM model to use for mask prediction.
        points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.
        points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.
        pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
        stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
        stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
        box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
        crop_n_layers (int): If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run, where each
            layer has 2**i_layer number of image crops.
        crop_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks between different crops.
        crop_overlap_ratio (float): Sets the degree to which crops overlap.
            In the first crop layer, crops will overlap by this fraction of
            the image length. Later layers with more crops scale down this overlap.
        crop_n_points_downscale_factor (int): The number of points-per-side
            sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
        point_grids (list(np.ndarray) or None): A list over explicit grids
            of points used for sampling, normalized to [0,1]. The nth grid in the
            list is used in the nth crop layer. Exclusive with points_per_side.
        min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
        output_mode (str): The form masks are returned in. Can be 'binary_mask',
            'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
            For large resolutions, 'binary_mask' may consume large amounts of
            memory.

https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py
"""

class ImageSegmentation(object):
    def __init__(self, configs):
        self.configs = configs

        model_pool = ['sam']

        model_name = configs['model_name']

        if model_name not in model_pool:
            raise NotImplementedError('Model not implemented.')
        elif model_name == 'sam':
            assert os.path.exists(configs['model_path']), 'Model path does not exist.'

            sam_checkpoint = configs['model_path']
            model_type = configs['model_type']
            device = configs['device']
            points_per_side = configs['points_per_side']
            pred_iou_thresh = configs['pred_iou_thresh']
            stability_score_thresh = configs['stability_score_thresh']

            self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            self.sam.to(device=device)
            
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam, 
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh
            )

    def predict(self, image_path, maximum_size=1000):
        """
        Arguments:
            image_path (str): Path to the image.
            maximum_size (int): The maximum size of the image. If the image is larger than this, it will be resized.

        Returns:
            masks (list): A list of masks.
        """
        assert os.path.exists(image_path), 'Image path does not exist.'
        image = cv2.imread(image_path)

        self.image_size = image.shape

        if image.shape[0] > maximum_size or image.shape[1] > maximum_size:
            if image.shape[0] > image.shape[1]:
                image = cv2.resize(image, (int(image.shape[1] * maximum_size / image.shape[0]), maximum_size))
            else:
                image = cv2.resize(image, (maximum_size, int(image.shape[0] * maximum_size / image.shape[1])))
        else:
            pass
        
        if self.configs['model_name'] == 'sam':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masks = self.mask_generator.generate(image)
        else:
            raise NotImplementedError('Model not implemented.')
        
        self.image = image.copy()
        return masks
    
    def batch_predict(self, image_paths, save_folder_path, maximum_size=1000, save_overlap=False):
        """
        Arguments:
            image_paths (list): A list of image paths.
            save_folder_path (str): The path to save the masks.
            maximum_size (int): The maximum size of the image. If the image is larger than this, it will be resized.
            save_overlap (bool): Whether to save the overlap of the image and the masks.
        """
        # create save folder if not exists
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

        # predict and save npy
        total = len(image_paths)
        for i, image_path in enumerate(image_paths):
            print('Processing image {}/{}.'.format(i+1, total))
            masks = self.predict(image_path, maximum_size)
            save_path = os.path.join(save_folder_path, os.path.basename(image_path).split('.')[0] + '.npy')
            self.save_npy(masks, save_path)
            if save_overlap:
                overlap_save_path = os.path.join(save_folder_path, os.path.basename(image_path).split('.')[0] + '_overlap.png')
                self.save_overlap(self.image, masks, overlap_save_path)

    def save_overlap(self, image, masks, save_path):
        """
        Arguments:
            masks (list): A list of masks.
            save_path (str): The path to save the masks.

        Note that the saved image is a 3-channel image with the maximum size of maxium_size. The images and masks are resized to the maximum size.
        """
        if len(masks) == 0:
            raise ValueError('No masks to save.')
            return
        else:
            sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)

            img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
            for ann in sorted_anns:
                m = ann['segmentation']
                color_mask = np.concatenate([np.random.random(3)])
                img[m] = color_mask
        
        # overlap self.image and img using the last channel of img as alpha channel
        img = img * 255
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.addWeighted(img, 0.35, image, 0.65, 0)
        cv2.imwrite(save_path, img)


    def save_npy(self, masks, save_path):
        """
        Arguments:
            masks (list): A list of masks.
            save_path (str): The path to save the masks.
        
        Returns:
            None

        The saved npy file is a 2D array with the same size as the orignal image. Each pixel in the array 
            is the index of the mask that the pixel belongs to. The valid index starts from 0.
            If the pixel does not belong to any mask, the value is -1.
        """
        if len(masks) == 0:
            raise ValueError('No masks to save.')
            return
        else:
            sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)

            img = -np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]))
            for id, ann in enumerate(sorted_anns):
                m = ann['segmentation']
                img[m] = id

            # resize img to the original size (self.image_size) using nearest neighbor interpolation
            img = cv2.resize(img, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)

            # set the dtype to np.int16
            img = img.astype(np.int16)

            np.save(save_path, img)


sam_params = {}
sam_params['model_name'] = 'sam'
sam_params['model_path'] = '../sam/sam_vit_h_4b8939.pth'
sam_params['model_type'] = 'vit_h'
sam_params['device'] = 'cuda:1'
sam_params['points_per_side'] = 64
sam_params['pred_iou_thresh'] = 0.96
sam_params['stability_score_thresh'] = 0.92

if __name__ == '__main__':
    site = "box_canyon" # "box_canyon" or "courtwright
    single_test = False
    batch_test = True
    write_segmentation_test = False  

    if single_test:
        image_segmentor = ImageSegmentation(sam_params)        
        image_path = '../../data/mission_2/DJI_0247.JPG'
        masks = image_segmentor.predict(image_path)
        image_segmentor.save_overlap(image_segmentor.image, masks, './test.png')
        image_segmentor.save_npy(masks, './test.npy')

    
    if batch_test:
        if site == "box_canyon":
            image_segmentor = ImageSegmentation(sam_params)   
            image_folder_path = '../../data/mission_2'
            segmentation_folder_path = '../../data/mission_2_segmentations'
            image_paths = [os.path.join(image_folder_path, f) for f in os.listdir(image_folder_path) if f.endswith('.JPG')]
            image_segmentor.batch_predict(image_paths, segmentation_folder_path, save_overlap=True)
        elif site == "courtwright":
            image_segmentor = ImageSegmentation(sam_params)   
            image_folder_path = '../../data/courtwright/photos'
            segmentation_folder_path = '../../data/courtwright/segmentations'
            image_paths = [os.path.join(image_folder_path, f) for f in os.listdir(image_folder_path) if f.endswith('.JPG')]
            image_segmentor.batch_predict(image_paths, segmentation_folder_path)

    
    if write_segmentation_test:
        segmentation_path = '../../data/mission_2_segmentations/DJI_0183.npy'
        save_path = '../../data/DJI_0183_overlap.png'
        image_path = '../../data/mission_2/DJI_0183.JPG'
        # read image
        image = cv2.imread(image_path)

        maxium_size = 1000
        if image.shape[0] > maxium_size or image.shape[1] > maxium_size:
            if image.shape[0] > image.shape[1]:
                image = cv2.resize(image, (int(image.shape[1] * maxium_size / image.shape[0]), maxium_size))
            else:
                image = cv2.resize(image, (maxium_size, int(image.shape[0] * maxium_size / image.shape[1])))
        else:
            pass
        
        # read segmentation
        mask = np.load(segmentation_path)
        mask = mask 
        # add a random color to image for each mask
        img = np.ones((mask.shape[0], mask.shape[1], 3))
        for i in range(0, int(mask.max()) + 1):
            color_mask = np.concatenate([np.random.random(3)])
            img[mask == i] = color_mask

        # overlap image and img using the last channel of img as alpha channel
        img = img * 255
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.addWeighted(img, 0.35, image, 0.65, 0)

        cv2.imwrite(save_path, img)
        


            
                                    
            


