from ssfm.simple_mask_filter import AreaFilter
from ssfm.probabilistic_projection import *
from ssfm.keyimage_associations_builder import *
from ssfm.object_registration import *
from ssfm.post_processing import *
from validation import Validator
import logging


def kubric_validation(data_id, area_lower_threshold, depth_filtering_threshold, nearest_interpolation, only_validation=False, gt_mask=True):
    scene_dir = f'../../data/kubric_{data_id}'
    pointcloud_path = os.path.join(scene_dir, 'reconstructions', 'combined_point_cloud.las')
    associations_folder_path = os.path.join(scene_dir, 'associations')
    photos_folder_path = os.path.join(scene_dir, 'photos')

    print('Filtering areas:')
    if gt_mask:
        segmentations_folder_path = os.path.join(scene_dir, 'segmentations_gt_filtered')
        
        config = {'area_lower_threshold': area_lower_threshold,
        'segmentation_folder_path': f'../../data/kubric_{data_id}/segmentations_gt',
        'output_folder_path': segmentations_folder_path,
        'num_processes':8}
    else:
        segmentations_folder_path = os.path.join(scene_dir, 'segmentations_filtered')
        
        config = {'area_lower_threshold': area_lower_threshold,
        'segmentation_folder_path': f'../../data/kubric_{data_id}/segmentations',
        'output_folder_path': segmentations_folder_path,
        'num_processes':8}

    area_filter = AreaFilter()
    area_filter(config)
    
    if not only_validation:
        print('Projecting point cloud to images:')
        pointcloud_projector = PointcloudProjection(depth_filtering_threshold=depth_filtering_threshold)
        pointcloud_projector.read_kubric_camera_parameters(scene_dir)
        pointcloud_projector.read_pointcloud(pointcloud_path)

        photo_folder_path = os.path.join(scene_dir, 'photos')
        image_list = [f for f in os.listdir(photo_folder_path) if f.endswith('.png')]
        # sort image list based on the number in the file name
        image_list.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

        pointcloud_projector.parallel_batch_project_joblib(image_list, associations_folder_path, num_workers=16, save_depth=False)

        print('Building keyimage associations:')
        smc_solver = KeyimageAssociationsBuilder(associations_folder_path, segmentations_folder_path)
        smc_solver.build_associations()

        keyimage_associations_file_name = 'associations_keyimage.npy'
        keyimage_yaml_name= 'keyimages.yaml'

        print('Running object registration:')
        obr = ObjectRegistration(pointcloud_path, segmentations_folder_path, associations_folder_path, keyimage_associations_file_name=keyimage_associations_file_name, loginfo=False)

        # Run object registration
        obr.object_registration(iou_threshold=0.50, save_semantics=True)

    print('Adding semantics to point cloud:')
    image_id = 284
    semantics_folder_path = os.path.join(associations_folder_path, 'semantics', 'semantics_{}.npy'.format(image_id))
    save_las_path = os.path.join(associations_folder_path, 'semantics', 'semantics_{}.las'.format(image_id))
    if gt_mask:
        add_semantics_to_pointcloud(pointcloud_path, semantics_folder_path, save_las_path, nearest_interpolation=nearest_interpolation)
    else:
        add_semantics_to_pointcloud(pointcloud_path, semantics_folder_path, save_las_path, remove_small_N=500, nearest_interpolation=nearest_interpolation)

    # read las 
    pc = laspy.read(save_las_path)
    # get the semantics from the intensity
    semantics = pc.intensity
    semantics_ids = np.unique(semantics)
    # print the number of points for each semantics
    for i in semantics_ids:
        n = np.sum(semantics == i)
        if n < 100:
            print('semantics id: ', i, ' number of points: ', n)

    print("Shuffling semantics: ")
    semantic_pc_file_path = save_las_path
    post_processing = PostProcessing(semantic_pc_file_path)
    post_processing.shuffle_semantic_ids()
    save_las_path = os.path.join(associations_folder_path, 'semantics', 'semantics_{}_shuffled.las'.format(image_id))
    post_processing.save_semantic_pointcloud(save_las_path)

    print("Validating: ")
    validator = Validator(
            save_las_path,
            pointcloud_path,)

    results = validator.validate(np.arange(0.5, 1.0, 0.05))

    return results

def kubric_parameter_search(data_id):
    # Create a dedicated logger for the main function
    main_logger = logging.getLogger('kubric_batch_validation_main')
    main_logger.setLevel(logging.DEBUG)
    
    # Create a file handler for logging
    fh = logging.FileHandler(f'kubric_batch_validation_main_{data_id}.log')
    fh.setLevel(logging.DEBUG)
    
    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    
    # Add the file handler to the logger
    main_logger.addHandler(fh)
    
    # Disable propagation to prevent logs from other loggers
    main_logger.propagate = False

    main_logger.info('Started')

    results = []

    for area_lower_threshold in range(80, 101, 10):
        depth_filtering_threshold = 1.8
        nearest_interpolation = 350
        results = kubric_validation(data_id, area_lower_threshold, depth_filtering_threshold, nearest_interpolation)
        mAP = results['AP']
        mAR = results['AR']
        print('area_lower_threshold: ', area_lower_threshold, ' depth_filtering_threshold: ', depth_filtering_threshold, ' nearest_interpolation: ', nearest_interpolation)
        print('mAP list: ', mAP, ' mAR list: ', mAR)
        mAP = np.sum(mAP) / len(mAP)
        mAR = np.sum(mAR) / len(mAR)
        print('mAP: ', mAP, ' mAR: ', mAR)
        result = {'area_lower_threshold': area_lower_threshold, 'depth_filtering_threshold': depth_filtering_threshold, 'nearest_interpolation': nearest_interpolation, 'mAP': mAP, 'mAR': mAR}
        print('------------------------------------')
        # log result in main function's logger
        main_logger.info(result)

    for depth_filtering_threshold in np.arange(1.6, 2.1, 0.1):
        area_lower_threshold = 80
        nearest_interpolation = 350
        results = kubric_validation(data_id, area_lower_threshold, depth_filtering_threshold, nearest_interpolation)
        mAP = results['AP']
        mAR = results['AR']
        print('area_lower_threshold: ', area_lower_threshold, ' depth_filtering_threshold: ', depth_filtering_threshold, ' nearest_interpolation: ', nearest_interpolation)
        print('mAP list: ', mAP, ' mAR list: ', mAR)
        mAP = np.sum(mAP) / len(mAP)
        mAR = np.sum(mAR) / len(mAR)
        print('mAP: ', mAP, ' mAR: ', mAR)
        result = {'area_lower_threshold': area_lower_threshold, 'depth_filtering_threshold': depth_filtering_threshold, 'nearest_interpolation': nearest_interpolation, 'mAP': mAP, 'mAR': mAR}
        print('------------------------------------')
        # log result in main function's logger
        main_logger.info(result)
    
    area_lower_threshold = 80
    depth_filtering_threshold = 1.8
    nearest_interpolation = 100
    mAP, mAR = kubric_validation(1, area_lower_threshold, depth_filtering_threshold, nearest_interpolation, only_validation=False)
    for nearest_interpolation in range(250, 550, 50):
        print('area_lower_threshold: ', area_lower_threshold, ' depth_filtering_threshold: ', depth_filtering_threshold, ' nearest_interpolation: ', nearest_interpolation)
        results = kubric_validation(data_id, area_lower_threshold, depth_filtering_threshold, nearest_interpolation, only_validation=True)
        mAP = results['AP']
        mAR = results['AR']
        print('mAP list: ', mAP, ' mAR list: ', mAR)
        mAP = np.sum(mAP) / len(mAP)
        mAR = np.sum(mAR) / len(mAR)
        print('mAP: ', mAP, ' mAR: ', mAR)
        result = {'area_lower_threshold': area_lower_threshold, 'depth_filtering_threshold': depth_filtering_threshold, 'nearest_interpolation': nearest_interpolation, 'mAP': mAP, 'mAR': mAR}
        results.append(result)
        print('------------------------------------')
        # log result in main function's logger
        main_logger.info(result)
                

    print('Results: ')
    for result in results:
        print(result)

def kubric_batch_validation(data_ids, area_lower_threshold, depth_filtering_threshold, nearest_interpolation, only_validation=False, gt_mask=True):
    TP = []
    FP = []
    FN = []
    for data_id in data_ids:
        results = kubric_validation(data_id, area_lower_threshold, depth_filtering_threshold, nearest_interpolation, only_validation, gt_mask)
        TP.append(results['TP'])
        FP.append(results['FP'])
        FN.append(results['FN'])
    
    # reshape the lists; data_id x iou_threshold (0.5, 0.55, ..., 0.95)
    TP = np.array(TP)
    FP = np.array(FP)
    FN = np.array(FN)
    TP = TP.reshape(len(data_ids), -1)
    FP = FP.reshape(len(data_ids), -1)
    FN = FN.reshape(len(data_ids), -1)
    # compute mAP, mAR
    AP = np.sum(TP, axis=0) / (np.sum(TP, axis=0) + np.sum(FP, axis=0))
    AR = np.sum(TP, axis=0) / (np.sum(TP, axis=0) + np.sum(FN, axis=0))
    mAP = np.sum(AP) / len(AP)
    mAR = np.sum(AR) / len(AR)

    print('AP: ', AP, ' AR: ', AR)
    print('mAP: ', mAP, ' mAR: ', mAR)


if __name__ == "__main__":
    
    #data_ids = list(range(15))
    data_ids = [0]
    area_lower_threshold = 80
    depth_filtering_threshold = 1.8
    nearest_interpolation = 500
    kubric_batch_validation(data_ids, area_lower_threshold, depth_filtering_threshold, nearest_interpolation, gt_mask=False)