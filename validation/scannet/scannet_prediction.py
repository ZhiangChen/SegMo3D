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
import time
from PIL import Image


class ScanNetPrediction(object):
    def __init__(self, scannet_ssfm_dir):
        assert os.path.exists(scannet_ssfm_dir), 'Invalid ScanNet SSFM directory'
        self.scannet_ssfm_dir = scannet_ssfm_dir
        scene_list = os.listdir(scannet_ssfm_dir)
        # sort scene list by the scene number in the scene name
        scene_list = sorted(scene_list, key=lambda x: int(x[5:9]))
        self.scene_paths = [os.path.join(scannet_ssfm_dir, scene) for scene in scene_list]
        # remove the scenes with empty folders
        self.scene_paths = [scene for scene in self.scene_paths if os.path.exists(os.path.join(scene, 'photos'))]
        

    def batch_keyimage_selection(self):
        for scene_path in self.scene_paths:
            select_scannet_keyimages(scene_path, ratio=0.2, threshold=180, file_cluster_size=20, file_select_window=6, n_jobs=8)

    def batch_rename_segmentations(self):
        for scene_dir in self.scene_paths:
            # if segmentation_gt exists, continue
            if os.path.exists(os.path.join(scene_dir, 'segmentations_gt')):
                continue
            segmentations_folder_path = os.path.join(scene_dir, 'segmentations')
            # rename the folder to 'segmentation_gt'
            os.rename(segmentations_folder_path, os.path.join(scene_dir, 'segmentations_gt'))


    def image_segmentation(self, gpu, scene_paths, sam2):
        if not sam2:
            sam_params = {}
            sam_params['model_name'] = 'sam'
            sam_params['model_path'] = '../../semantic_SfM/sam/sam_vit_h_4b8939.pth'
            sam_params['model_type'] = 'vit_h'
            sam_params['device'] = 'cuda:{}'.format(gpu)
            sam_params['points_per_side'] = 16
            sam_params['pred_iou_thresh'] = 0.96
            sam_params['stability_score_thresh'] = 0.96
            sam_params['crop_n_layers'] = 1
        else:
            sam_params = {}
            sam_params['model_name'] = 'sam2'
            sam_params['model_path'] = '../../semantic_SfM/sam2/sam2.1_hiera_large.pt'
            sam_params['device'] = 'cuda:{}'.format(gpu)
            sam_params['points_per_side'] = 32
            sam_params['points_per_batch'] = 128
            sam_params['pred_iou_thresh'] = 0.6
            sam_params['stability_score_offset'] = 0.5
            sam_params['box_nms_thresh'] = 0.6
            sam_params['use_m2m'] = True

        image_segmentor = ImageSegmentation(sam_params)   

        for scene_dir in scene_paths:
            print('Processing scene: {}'.format(scene_dir))
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
            results = image_segmentor.batch_predict(image_paths, segmentations_folder_path, maximum_size=10000, save_overlap=True)

            if len(results) != len(images):
                # update the keyimages.yaml file using the results
                keyimages = [os.path.basename(image).replace('.jpg', '.npy') for image in results]
                with open(keyimages_path, 'w') as f:
                    yaml.dump(keyimages, f)

            
    def batch_image_segmentation(self, gpus, sam2=True):
        # divide the scenes into batches
        batch_number = len(gpus)
        batch_size = len(self.scene_paths) // batch_number
        scene_batches = [self.scene_paths[i:i+batch_size] for i in range(0, len(self.scene_paths), batch_size)]
        # add the remaining scenes to the last batch
        if len(scene_batches) > batch_number:
            scene_batches[-1].extend(scene_batches[-2])
            scene_batches.pop(-2)

        # parallelize the image segmentation
        Parallel(n_jobs=batch_number)(delayed(self.image_segmentation)(gpu, scene_batch, sam2) for gpu, scene_batch in zip(gpus, scene_batches))



    def batch_projection_association(self):
        I = 0
        for scene_dir in self.scene_paths:
            print('Processing scene: {}'.format(scene_dir))
            print('Scene number: {}'.format(I))
            I += 1
            pointcloud_projector = PointcloudProjection(depth_filtering_threshold=0.005)
            pointcloud_projector.read_scannet_camera_parameters(scene_dir)
            mesh_file_path = os.path.join(scene_dir, 'reconstructions', 'mesh_vertices_color.npy')
            pointcloud_projector.read_scannet_mesh(mesh_file_path)
            keyimages_path = os.path.join(scene_dir, 'associations', 'keyimages.yaml')
            associations_folder_path = os.path.join(scene_dir, 'associations')
            segmentations_folder_path = os.path.join(scene_dir, 'segmentations')

            associations_file_path = os.path.join(associations_folder_path, 'associations_keyimage.npy')
            
            with open(keyimages_path, 'r') as f:
                keyimages = yaml.safe_load(f)

            image_list = [os.path.splitext(image)[0] + '.jpg' for image in keyimages]
            image_list = sorted(image_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            pointcloud_projector.parallel_batch_project_joblib(image_list, associations_folder_path, num_workers=16, save_depth=False)

            smc_solver = KeyimageAssociationsBuilder(image_list, associations_folder_path, segmentations_folder_path)
            smc_solver.build_associations()
            print('Associations built for scene: {}'.format(scene_dir))
            smc_solver.find_min_cover()

    def object_registration(self, scene_dir):
        pointcloud_path = os.path.join(scene_dir, 'reconstructions', 'mesh_vertices_color.npy')
        segmentations_folder_path = os.path.join(scene_dir, 'segmentations')
        associations_folder_path = os.path.join(scene_dir, 'associations')

        save_path = os.path.join(scene_dir, 'semantic_model.las')
        # check if save path exists, if yes, skip the object registration
        if os.path.exists(save_path):
            return

        keyimages_path = os.path.join(scene_dir, 'associations', 'keyimages.yaml')
        
        with open(keyimages_path, 'r') as f:
            keyimages = yaml.safe_load(f)

        image_list = [os.path.splitext(image)[0] + '.jpg' for image in keyimages]
        image_list = sorted(image_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))

        keyimage_associations_file_name = 'associations_keyimage.npy'
        keyimage_yaml_name = 'keyimages.yaml'
        # Create object registration
        obr = ObjectRegistration(pointcloud_path, 
                                 segmentations_folder_path,
                                 associations_folder_path, 
                                 keyimage_associations_file_name=keyimage_associations_file_name, 
                                 image_list=image_list,
                                 loginfo=False)

        # Run object registration
        obr.object_registration(iou_threshold=0.5, save_semantics=True)

        # save semantic point cloud
        obr.save_semantic_pointcloud(save_path)


    def batch_object_registration(self, n_jobs=16):  # be careful with the number of jobs because of the memory usage
        # Parallelize object registration with a progress bar
        Parallel(n_jobs=n_jobs)(delayed(self.object_registration)(scene_dir) for scene_dir in tqdm(self.scene_paths))


    def process_scene(self, scene_dir):
        print('Processing scene: {}'.format(scene_dir))
        pointcloud_path = os.path.join(scene_dir, 'reconstructions', 'mesh_vertices_color.npy')
        associations_folder_path = os.path.join(scene_dir, 'associations')

        semantics_ids = [int(file.split('_')[-1].split('.')[0]) 
                        for file in os.listdir(os.path.join(associations_folder_path, 'semantics')) if file.endswith('.npy')]
        semantics_ids = sorted(semantics_ids)
        semantics_id = semantics_ids[-1]
        
        semantics_path = os.path.join(associations_folder_path, 'semantics', f'semantics_{semantics_id}.npy')
        save_las_path = os.path.join(associations_folder_path, 'semantics', f'semantics_{semantics_id}.las')
        
        add_semantics_to_pointcloud(pointcloud_path, semantics_path, save_las_path, remove_small_N=200, nearest_interpolation=200)

        post_processing = PostProcessing(save_las_path)
        post_processing.shuffle_semantic_ids(exclude_largest_semantic=False)
        save_las_path_shuffled = os.path.join(associations_folder_path, 'semantics', f'semantics_{semantics_id}_shuffled.las')
        post_processing.save_semantic_pointcloud(save_las_path_shuffled)

    # Parallelize the process
    def batch_post_processing(self, n_jobs=16):
        Parallel(n_jobs=n_jobs)(delayed(self.process_scene)(scene_dir) for scene_dir in self.scene_paths)

    # def batch_post_processing(self):
    #     for scene_dir in self.scene_paths:
    #         print('Processing scene: {}'.format(scene_dir))
    #         pointcloud_path = os.path.join(scene_dir, 'reconstructions', 'mesh_vertices_color.npy')
    #         associations_folder_path = os.path.join(scene_dir, 'associations')
    #         # get the semantics id, which is the last number in the file name
    #         semantics_ids = [int(file.split('_')[-1].split('.')[0]) for file in os.listdir(os.path.join(associations_folder_path, 'semantics')) if file.endswith('.npy')]
    #         semantics_ids = sorted(semantics_ids)
    #         semantics_id = semantics_ids[-1]
    #         semantics_path = os.path.join(associations_folder_path, 'semantics', 'semantics_{}.npy'.format(semantics_id))
    #         save_las_path = os.path.join(associations_folder_path, 'semantics', 'semantics_{}.las'.format(semantics_id))
    #         add_semantics_to_pointcloud(pointcloud_path, semantics_path, save_las_path, remove_small_N=800, nearest_interpolation=200)

    #         post_processing = PostProcessing(save_las_path)
    #         post_processing.shuffle_semantic_ids(exclude_largest_semantic=False)
    #         save_las_path = os.path.join(associations_folder_path, 'semantics', 'semantics_{}_shuffled.las'.format(semantics_id))
    #         post_processing.save_semantic_pointcloud(save_las_path)

    def batch_prediction_output(self):
        # format the output of the prediction as scanNet benchmark required
        # https://kaldir.vc.in.tum.de/scannet_benchmark/documentation 
        pass

    def examine_keyimages(self):
        for scene_dir in self.scene_paths:
            keyimages_path = os.path.join(scene_dir, 'associations', 'keyimages.yaml')
            with open(keyimages_path, 'r') as f:
                keyimages = yaml.safe_load(f)

            images = [f for f in os.listdir(os.path.join(scene_dir, 'segmentations')) if f.endswith('.npy')]


            # update the keyimages.yaml file using the images
            with open(keyimages_path, 'w') as f:
                yaml.dump(images, f)

    def batch_clean_segmentations(self):
        for scene_dir in tqdm(self.scene_paths):
            segmentations_folder_path = os.path.join(scene_dir, 'segmentations')
            for file in os.listdir(segmentations_folder_path):
                # remove all files in the segmentations folder
                os.remove(os.path.join(segmentations_folder_path, file))



if __name__ == '__main__':
    scannet_ssfm_dir = '../../data/scannet/ssfm_valid'
    scannet_prediction = ScanNetPrediction(scannet_ssfm_dir)
    scannet_prediction.batch_clean_segmentations()
    #scannet_prediction.batch_keyimage_selection()
    #scannet_prediction.batch_rename_segmentations()
    #scannet_prediction.batch_image_segmentation(gpus=[1, 2, 3, 4, 5, 6, 7] , sam2=True)
    #scannet_prediction.examine_keyimages()
    #scannet_prediction.batch_projection_association()
    #scannet_prediction.batch_object_registration()
    #scannet_prediction.batch_post_processing(n_jobs=16)
    