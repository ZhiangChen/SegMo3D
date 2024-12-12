import os
import cv2
import pickle
import numpy as np
from ssfm.files import *

from joblib import Parallel, delayed
from tqdm import tqdm

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

        model_pool = ['sam', 'sam_hq', 'sam2']

        model_name = configs['model_name']

        if model_name not in model_pool:
            raise NotImplementedError('Model not implemented.')
        elif model_name == 'sam':
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
            assert os.path.exists(configs['model_path']), 'Model path does not exist.'

            sam_checkpoint = configs['model_path']
            model_type = configs['model_type']
            device = configs['device']
            points_per_side = configs['points_per_side']
            pred_iou_thresh = configs['pred_iou_thresh']
            stability_score_thresh = configs['stability_score_thresh']
            crop_n_layers = configs['crop_n_layers']

            self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            self.sam.to(device=device)
            
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam, 
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                crop_n_layers = crop_n_layers
            )

        elif model_name == 'sam_hq':
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
            assert os.path.exists(configs['model_path']), 'Model path does not exist.'

            sam_checkpoint = configs['model_path']
            model_type = configs['model_type']
            device = configs['device']
            points_per_side = configs['points_per_side']
            pred_iou_thresh = configs['pred_iou_thresh']
            stability_score_thresh = configs['stability_score_thresh']
            crop_n_layers = configs['crop_n_layers']

            self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            self.sam.to(device=device)
            
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam, 
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                crop_n_layers = crop_n_layers
            )

        elif model_name == 'sam2':
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

            sam2_checkpoint = configs['model_path']
            assert os.path.exists(sam2_checkpoint), 'Model path does not exist.'

            device = configs['device']
            points_per_side = configs.get('points_per_side', 32)
            points_per_batch = configs.get('points_per_batch', 128)
            pred_iou_thresh = configs.get('pred_iou_thresh', 0.6)
            stability_score_thresh = configs.get('stability_score_thresh', 0.96)
            stability_score_offset = configs.get('stability_score_offset', 0.5)
            box_nms_thresh = configs.get('box_nms_thresh', 0.6)
            use_m2m = configs.get('use_m2m', True)

            sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=sam2,
                points_per_side=points_per_side,
                points_per_batch=points_per_batch,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                stability_score_offset=stability_score_offset,
                box_nms_thresh=box_nms_thresh,
                use_m2m=use_m2m,
            )


        self.distortion_params = None

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

        if self.distortion_params is not None:
            # undistort the image
            image = cv2.undistort(image, self.matrix_intrinsics, self.distortion_params)

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
        elif self.configs['model_name'] == 'sam_hq':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masks = self.mask_generator.generate(image)
        elif self.configs['model_name'] == 'sam2':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masks = self.mask_generator.generate(image)
        else:
            raise NotImplementedError('Model not implemented.')
        
        self.image = image.copy()
        return masks
    
    def batch_predict(self, image_paths, save_folder_path, maximum_size=1000, save_overlap=False, skip_existing=True):
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
        predicted_images = []
        total = len(image_paths)
        for i, image_path in enumerate(image_paths):
            #print('Processing image {}/{}.'.format(i+1, total))
            save_path = os.path.join(save_folder_path, os.path.basename(image_path).split('.')[0] + '.npy')
            if skip_existing and os.path.exists(save_path):
                continue
            masks = self.predict(image_path, maximum_size)
            result = self.save_npy(masks, save_path)
            if save_overlap:
                overlap_save_path = os.path.join(save_folder_path, os.path.basename(image_path).split('.')[0] + '_overlap.png')
                self.save_overlap(self.image, masks, overlap_save_path)
            if result:
                predicted_images.append(image_path)

        return predicted_images

    def save_overlap(self, image, masks, save_path):
        """
        Arguments:
            masks (list): A list of masks.
            save_path (str): The path to save the masks.

        Note that the saved image is a 3-channel image with the maximum size of maxium_size. The images and masks are resized to the maximum size.
        """
        if len(masks) == 0:
            #raise ValueError('No masks to save.')
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
            #raise ValueError('No masks to save.')
            return False
        else:
            sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)

            img = -np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]))
            for id, ann in enumerate(sorted_anns):
                m = ann['segmentation']
                img[m] = id

            # resize img to the original size (self.image_size) using nearest neighbor interpolation
            img = cv2.resize(img, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)

            """
            if self.distortion_params is not None:
                # undistort the image
                img = img + 1
                img = cv2.undistort(img, self.matrix_intrinsics, self.distortion_params)
                img = img - 1
            else:
                pass
            """
            # set the dtype to np.int16
            img = img.astype(np.int16)

            np.save(save_path, img)

            return True

class ParallelImageSegmentation(object):
    def __init__(self, configs):
        devices = configs['devices']
        assert len(devices) > 0, 'No devices specified.'
        # print the devices
        print('Devices:', devices)
        model_params = {}
        model_params['model_name'] = configs['model_name']
        model_params['model_path'] = configs['model_path']
        model_params['model_type'] = configs['model_type']
        model_params['points_per_side'] = configs['points_per_side']
        model_params['pred_iou_thresh'] = configs['pred_iou_thresh']
        model_params['stability_score_thresh'] = configs['stability_score_thresh']
        model_params['crop_n_layers'] = configs['crop_n_layers']

        self.models_params = []
        for device in devices:
            model_params['device'] = device
            self.models_params.append(model_params.copy())
        
        self.distortion_correction = False
        self.segmentation_folder_path = None
        self.maximum_size = None
        self.save_overlap = None

    def __predict(self, model_params, image_paths):
        """
        Arguments:
            model_params (dict): The parameters of the model.
            image_paths (list): A list of image paths.
        """
        image_segmentor = ImageSegmentation(model_params)
        if self.distortion_correction:
            if self.additional_camera_parameters_path is not None:
                image_segmentor.set_distortion_correction(self.camera_parameters_path, self.additional_camera_parameters_path)
            else:
                image_segmentor.set_distortion_correction(self.camera_parameters_path)

        image_segmentor.batch_predict(image_paths, self.segmentation_folder_path, self.maximum_size, self.save_overlap)


    def parallel_predict(self, image_paths, segmentation_folder_path, maximum_size=10000, save_overlap=False, camera_parameters_path=None, additional_camera_parameters_path=None):
        """
        Arguments:
            image_paths (list): A list of image paths.
            segmentation_folder_path (str): The path to save the masks.
            maximum_size (int): The maximum size of the image. If the image is larger than this, it will be resized.
            save_overlap (bool): Whether to save the overlap of the image and the masks.
        """
        # create segmentation_folder_path if not exists
        if not os.path.exists(segmentation_folder_path):
            os.makedirs(segmentation_folder_path)

        if camera_parameters_path is not None:
            self.distortion_correction = True
            self.camera_parameters_path = camera_parameters_path
            self.additional_camera_parameters_path = additional_camera_parameters_path
        else:
            self.distortion_correction = False
        
        self.segmentation_folder_path = segmentation_folder_path
        self.maximum_size = maximum_size
        self.save_overlap = save_overlap
        
        # based on the devices, divide the image_paths
        num_devices = len(self.models_params)
        num_images = len(image_paths)
        num_images_per_device = num_images // num_devices
        image_paths_list = [image_paths[i*num_images_per_device:(i+1)*num_images_per_device] for i in range(num_devices-1)]
        image_paths_list.append(image_paths[(num_devices-1)*num_images_per_device:])

        # use joblib to parallel predict
        Parallel(n_jobs=num_devices)(delayed(self.__predict)(self.models_params[i], image_paths_list[i]) for i in range(num_devices))



sam_params = {}
sam_params['model_name'] = 'sam'
sam_params['model_path'] = '../sam/sam_vit_h_4b8939.pth'
sam_params['model_type'] = 'vit_h'
sam_params['device'] = 'cuda:2'
sam_params['points_per_side'] = 64
sam_params['pred_iou_thresh'] = 0.96
sam_params['stability_score_thresh'] = 0.92
sam_params['crop_n_layers'] = 2

if __name__ == '__main__':
    site = "courtwright" # "box_canyon" or "courtwright
    single_test = True
    batch_test = False
    write_segmentation_test = False  

    if single_test:
        image_segmentor = ImageSegmentation(sam_params)
        # read an image
        image_path = '../../data/courtright/DJI_photos/DJI_0576.JPG'
        image = cv2.imread(image_path)
        # resize the image
        image = cv2.resize(image, (1000, 666))
        masks = image_segmentor.predict(image_path)
        image_segmentor.save_overlap(image, masks, '../../data/courtright/test.png')
        

    
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
        


            
                                    
            


