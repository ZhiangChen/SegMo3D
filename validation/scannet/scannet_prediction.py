from select_keyimages import select_scannet_keyimages
from ssfm.image_segmentation import ImageSegmentation
from ssfm.probabilistic_projection import *
from ssfm.object_registration import *
from ssfm.post_processing import *
from ssfm.keyimage_associations_builder import *
import os
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm


class ScanNetPrediction(object):
    def __init__(self, scannet_ssfm_dir):
        assert os.path.exists(scannet_ssfm_dir), 'Invalid ScanNet SSFM directory'
        self.scannet_ssfm_dir = scannet_ssfm_dir
        self.scene_list = os.listdir(scannet_ssfm_dir)
        # sort scene list by the scene number in the scene name
        self.scene_list = sorted(self.scene_list, key=lambda x: int(x[5:9]))
        self.scene_paths = [os.path.join(scannet_ssfm_dir, scene) for scene in self.scene_list]
        

    def batch_keyimage_selection(self):
        for scene_path in self.scene_paths:
            select_scannet_keyimages(scene_path, ratio=0.2, threshold=180, file_cluster_size=20, file_select_window=6, n_jobs=8)

    def batch_image_segmentation(self):
        sam_params = {}
        sam_params['model_name'] = 'sam'
        sam_params['model_path'] = '../../semantic_SfM/sam/sam_vit_h_4b8939.pth'
        sam_params['model_type'] = 'vit_h'
        sam_params['device'] = 'cuda:0'
        sam_params['points_per_side'] = 32
        sam_params['pred_iou_thresh'] = 0.96
        sam_params['stability_score_thresh'] = 0.96
        sam_params['crop_n_layers'] = 1

        image_segmentor = ImageSegmentation(sam_params)   

        for scene_dir in self.scene_paths:
            keyimages_path = os.path.join(scene_dir, 'associations', 'keyimages.yaml')
            # read keyimages
            with open(keyimages_path, 'r') as f:
                keyimages = yaml.load(f, Loader=yaml.FullLoader)

            # replace .npy with .jpg
            images = [keyimage.replace('.npy', '.jpg') for keyimage in keyimages]
            # sort images based on the values of keyimages in file names
            images = sorted(images, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            image_paths = [os.path.join(scene_dir, 'photos', image) for image in images]

            segmentations_folder_path = os.path.join(scene_dir, 'segmentations')
            image_segmentor.batch_predict(image_paths, segmentations_folder_path, maximum_size=10000, save_overlap=True)
            

    def batch_projection_association(self):
        for scene_dir in self.scene_paths:
            pointcloud_projector = PointcloudProjection(depth_filtering_threshold=0.005)
            pointcloud_projector.read_scannet_camera_parameters(scene_dir)
            mesh_file_path = os.path.join(scene_dir, 'reconstructions', 'mesh_vertices_color.npy')
            pointcloud_projector.read_scannet_mesh(mesh_file_path)
            keyimages_path = os.path.join(scene_dir, 'associations', 'keyimages.yaml')
            associations_folder_path = os.path.join(scene_dir, 'associations')
            segmentations_folder_path = os.path.join(scene_dir, 'segmentations')

            with open(keyimages_path, 'r') as f:
                keyimages = yaml.safe_load(f)

            image_list = [os.path.splitext(image)[0] + '.jpg' for image in keyimages]
            pointcloud_projector.parallel_batch_project_joblib(image_list, associations_folder_path, num_workers=16, save_depth=False)

            smc_solver = KeyimageAssociationsBuilder(associations_folder_path, segmentations_folder_path)
            smc_solver.build_associations()
            print('Associations built for scene: {}'.format(scene_dir))
            smc_solver.find_min_cover()

    def object_registration(self, scene_dir):
            pointcloud_path = os.path.join(scene_dir, 'reconstructions', 'mesh_vertices_color.npy')
            segmentations_folder_path = os.path.join(scene_dir, 'segmentations')
            associations_folder_path = os.path.join(scene_dir, 'associations')

            keyimage_associations_file_name = 'associations_keyimage.npy'
            keyimage_yaml_name = 'keyimages.yaml'
            # Create object registration
            obr = ObjectRegistration(pointcloud_path, segmentations_folder_path, associations_folder_path, keyimage_associations_file_name=keyimage_associations_file_name, keyimage_yaml_name=keyimage_yaml_name)

            # Run object registration
            obr.object_registration(iou_threshold=0.5, save_semantics=True)


    def batch_object_registration(self):
        # Parallelize object registration with a progress bar
        Parallel(n_jobs=8)(delayed(self.object_registration)(scene_dir) for scene_dir in tqdm(self.scene_paths))



    def batch_post_processing(self):
        for scene_dir in self.scene_paths:
            pointcloud_path = os.path.join(scene_dir, 'reconstructions', 'mesh_vertices_color.npy')
            associations_folder_path = os.path.join(scene_dir, 'associations')
            # get the semantics id, which is the last number in the file name
            semantics_ids = [int(file.split('_')[-1].split('.')[0]) for file in os.listdir(os.path.join(associations_folder_path, 'semantics')) if file.endswith('.npy')]
            semantics_ids = sorted(semantics_ids)
            semantics_id = semantics_ids[-1]
            semantics_path = os.path.join(associations_folder_path, 'semantics', 'semantics_{}.npy'.format(semantics_id))
            save_las_path = os.path.join(associations_folder_path, 'semantics', 'semantics_{}.las'.format(semantics_id))
            add_semantics_to_pointcloud(pointcloud_path, semantics_path, save_las_path, nearest_interpolation=5)

            post_processing = PostProcessing(save_las_path)
            post_processing.shuffle_semantic_ids()
            save_las_path = os.path.join(associations_folder_path, 'semantics', 'semantics_{}_shuffled.las'.format(semantics_id))
            post_processing.save_semantic_pointcloud(save_las_path)

    def batch_prediction_output(self):
        # format the output of the prediction as scanNet benchmark format
        pass


if __name__ == '__main__':
    scannet_ssfm_dir = '../../data/scannet/ssfm'
    scannet_prediction = ScanNetPrediction(scannet_ssfm_dir)
    #scannet_prediction.batch_keyimage_selection()
    #scannet_prediction.batch_image_segmentation()
    scannet_prediction.batch_projection_association()
    scannet_prediction.batch_object_registration()
    scannet_prediction.batch_post_processing()