from ssfm.image_segmentation import ImageSegmentation
import os


def sam2_segmentation(data_id, device):
    sam_params = {}
    sam_params['model_name'] = 'sam2'
    sam_params['model_path'] = '../../semantic_SfM/sam2/sam2.1_hiera_large.pt'
    sam_params['device'] = f'cuda:{device}'
    sam_params['points_per_side'] = 32
    sam_params['points_per_batch'] = 128
    sam_params['pred_iou_thresh'] = 0.6
    sam_params['stability_score_offset'] = 0.5
    sam_params['box_nms_thresh'] = 0.6
    sam_params['use_m2m'] = True

    scene_dir = f'../../data/kubric_{data_id}'
    photos_folder_path = os.path.join(scene_dir, 'photos')
    segmentations_folder_path = os.path.join(scene_dir, 'segmentations')

    image_paths = [os.path.join(scene_dir, 'photos', image) for image in os.listdir(os.path.join(scene_dir, 'photos'))]
    image_paths = sorted(image_paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    image_segmentor = ImageSegmentation(sam_params)   
    image_segmentor.batch_predict(image_paths, segmentations_folder_path, save_overlap=True)

if __name__ == '__main__':
    sam2_segmentation(data_id=0, device=1)
    sam2_segmentation(data_id=1, device=1)
    sam2_segmentation(data_id=2, device=1)
    sam2_segmentation(data_id=3, device=1)
        