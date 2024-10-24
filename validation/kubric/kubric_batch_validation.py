from ssfm.simple_mask_filter import AreaFilter
from ssfm.probabilistic_projection import *
from ssfm.keyimage_associations_builder import *
from ssfm.object_registration import *
from ssfm.post_processing import *
from validation import Validator
import csv
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

def kubric_parameter_search(data_id, gt_mask=True):
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
    """
    for area_lower_threshold in range(50, 210, 10):
        depth_filtering_threshold = 1.8
        nearest_interpolation = 350
        results = kubric_validation(data_id, area_lower_threshold, depth_filtering_threshold, nearest_interpolation, gt_mask=gt_mask)
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

    for depth_filtering_threshold in np.arange(1.2, 2.6, 0.1):
        area_lower_threshold = 80
        nearest_interpolation = 350
        results = kubric_validation(data_id, area_lower_threshold, depth_filtering_threshold, nearest_interpolation, gt_mask=gt_mask)
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
    """
    area_lower_threshold = 80
    depth_filtering_threshold = 1.8
    nearest_interpolation = 100
    results = kubric_validation(1, area_lower_threshold, depth_filtering_threshold, nearest_interpolation, only_validation=False, gt_mask=gt_mask)
    for nearest_interpolation in range(350, 750, 50):
        print('area_lower_threshold: ', area_lower_threshold, ' depth_filtering_threshold: ', depth_filtering_threshold, ' nearest_interpolation: ', nearest_interpolation)
        results = kubric_validation(data_id, area_lower_threshold, depth_filtering_threshold, nearest_interpolation, only_validation=True, gt_mask=gt_mask)
        mAP = results['AP']
        mAR = results['AR']
        print('mAP list: ', mAP, ' mAR list: ', mAR)
        mAP = np.sum(mAP) / len(mAP)
        mAR = np.sum(mAR) / len(mAR)
        print('mAP: ', mAP, ' mAR: ', mAR)
        result = {'area_lower_threshold': area_lower_threshold, 'depth_filtering_threshold': depth_filtering_threshold, 'nearest_interpolation': nearest_interpolation, 'mAP': mAP, 'mAR': mAR}
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

    # save results in CSV file
    with open('kubric_batch_validation.csv', mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['data_id', 'TP@50' , 'TP@55', 'TP@60', 'TP@65', 'TP@70', 'TP@75', 'TP@80', 'TP@85', 'TP@90', 'TP@95', 'FP@50', 'FP@55', 'FP@60', 'FP@65', 'FP@70', 'FP@75', 'FP@80', 'FP@85', 'FP@90', 'FP@95', 'FN@50', 'FN@55', 'FN@60', 'FN@65', 'FN@70', 'FN@75', 'FN@80', 'FN@85', 'FN@90', 'FN@95'])

    for data_id in data_ids:
        results = kubric_validation(data_id, area_lower_threshold, depth_filtering_threshold, nearest_interpolation, only_validation, gt_mask)
        TP.append(results['TP'])
        FP.append(results['FP'])
        FN.append(results['FN'])
        # save results in CSV file
        with open('kubric_batch_validation.csv', mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([data_id] + results['TP'] + results['FP'] + results['FN'])

    
    # reshape the lists; data_id x iou_threshold (0.5, 0.55, ..., 0.95)
    print('TP: ', TP)
    print('FP: ', FP)
    print('FN: ', FN)
    
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

def compute_metrics():
    # check if the csv file exists
    if not os.path.exists('kubric_batch_validation.csv'):
        print('The file does not exist')
        return

    # read the csv file
    with open('kubric_batch_validation.csv', mode='r') as file:
        reader = csv.reader(file)
        data = list(reader)

    # remove the header
    data = data[1:]
    data = np.array(data).astype(int)
    data = data[:, 1:]
    # computer metrics for each data_id
    for i in range(len(data)):
        data[i] = data[i].astype(float)
        TP = data[i, :10]
        FP = data[i, 10:20]
        FN = data[i, 20:]
        # compute mAP, mAR
        AP = TP / (TP + FP)
        AR = TP / (TP + FN)
        mAP = np.sum(AP) / len(AP)
        mAR = np.sum(AR) / len(AR)
        print('data_id: ', i, ' mAP: ', mAP, ' mAR: ', mAR)

    # compute mAP50, mAR50
    TP50 = data[:, 0]
    FP50 = data[:, 10]
    FN50 = data[:, 20]
    mAP50 = np.sum(TP50.astype(int)) / (np.sum(TP50.astype(int)) + np.sum(FP50.astype(int)))
    mAR50 = np.sum(TP50.astype(int)) / (np.sum(TP50.astype(int)) + np.sum(FN50.astype(int)))
    # compute mAP55, mAR55
    TP55 = data[:, 1]
    FP55 = data[:, 11]
    FN55 = data[:, 21]
    mAP55 = np.sum(TP55.astype(int)) / (np.sum(TP55.astype(int)) + np.sum(FP55.astype(int)))
    mAR55 = np.sum(TP55.astype(int)) / (np.sum(TP55.astype(int)) + np.sum(FN55.astype(int)))
    # compute mAP60, mAR60
    TP60 = data[:, 2]
    FP60 = data[:, 12]
    FN60 = data[:, 22]
    mAP60 = np.sum(TP60.astype(int)) / (np.sum(TP60.astype(int)) + np.sum(FP60.astype(int)))
    mAR60 = np.sum(TP60.astype(int)) / (np.sum(TP60.astype(int)) + np.sum(FN60.astype(int)))
    # compute mAP65, mAR65
    TP65 = data[:, 3]
    FP65 = data[:, 13]
    FN65 = data[:, 23]
    mAP65 = np.sum(TP65.astype(int)) / (np.sum(TP65.astype(int)) + np.sum(FP65.astype(int)))
    mAR65 = np.sum(TP65.astype(int)) / (np.sum(TP65.astype(int)) + np.sum(FN65.astype(int)))
    # compute mAP70, mAR70
    TP70 = data[:, 4]
    FP70 = data[:, 14]
    FN70 = data[:, 24]
    mAP70 = np.sum(TP70.astype(int)) / (np.sum(TP70.astype(int)) + np.sum(FP70.astype(int)))
    mAR70 = np.sum(TP70.astype(int)) / (np.sum(TP70.astype(int)) + np.sum(FN70.astype(int)))
    # compute mAP75, mAR75
    TP75 = data[:, 5]
    FP75 = data[:, 15]
    FN75 = data[:, 25]
    mAP75 = np.sum(TP75.astype(int)) / (np.sum(TP75.astype(int)) + np.sum(FP75.astype(int)))
    mAR75 = np.sum(TP75.astype(int)) / (np.sum(TP75.astype(int)) + np.sum(FN75.astype(int)))
    # compute mAP80, mAR80
    TP80 = data[:, 6]
    FP80 = data[:, 16]
    FN80 = data[:, 26]
    mAP80 = np.sum(TP80.astype(int)) / (np.sum(TP80.astype(int)) + np.sum(FP80.astype(int)))
    mAR80 = np.sum(TP80.astype(int)) / (np.sum(TP80.astype(int)) + np.sum(FN80.astype(int)))
    # compute mAP85, mAR85
    TP85 = data[:, 7]
    FP85 = data[:, 17]
    FN85 = data[:, 27]
    mAP85 = np.sum(TP85.astype(int)) / (np.sum(TP85.astype(int)) + np.sum(FP85.astype(int)))
    mAR85 = np.sum(TP85.astype(int)) / (np.sum(TP85.astype(int)) + np.sum(FN85.astype(int)))
    # compute mAP90, mAR90
    TP90 = data[:, 8]
    FP90 = data[:, 18]
    FN90 = data[:, 28]
    mAP90 = np.sum(TP90.astype(int)) / (np.sum(TP90.astype(int)) + np.sum(FP90.astype(int)))
    mAR90 = np.sum(TP90.astype(int)) / (np.sum(TP90.astype(int)) + np.sum(FN90.astype(int)))
    # compute mAP95, mAR95
    TP95 = data[:, 9]
    FP95 = data[:, 19]
    FN95 = data[:, 29]
    mAP95 = np.sum(TP95.astype(int)) / (np.sum(TP95.astype(int)) + np.sum(FP95.astype(int)))
    mAR95 = np.sum(TP95.astype(int)) / (np.sum(TP95.astype(int)) + np.sum(FN95.astype(int)))
    
    # computer mAP, mAR, which are the average of mAP50, mAP55, ..., mAP95
    mAP = (mAP50 + mAP55 + mAP60 + mAP65 + mAP70 + mAP75 + mAP80 + mAP85 + mAP90 + mAP95) / 10
    mAR = (mAR50 + mAR55 + mAR60 + mAR65 + mAR70 + mAR75 + mAR80 + mAR85 + mAR90 + mAR95) / 10

    # print the results
    print('mAP50: ', mAP50, ' mAR50: ', mAR50)
    print('mAP55: ', mAP55, ' mAR55: ', mAR55)
    print('mAP60: ', mAP60, ' mAR60: ', mAR60)
    print('mAP65: ', mAP65, ' mAR65: ', mAR65)
    print('mAP70: ', mAP70, ' mAR70: ', mAR70)
    print('mAP75: ', mAP75, ' mAR75: ', mAR75)
    print('mAP80: ', mAP80, ' mAR80: ', mAR80)
    print('mAP85: ', mAP85, ' mAR85: ', mAR85)
    print('mAP90: ', mAP90, ' mAR90: ', mAR90)
    print('mAP95: ', mAP95, ' mAR95: ', mAR95)
    print('mAP: ', mAP, ' mAR: ', mAR)



if __name__ == "__main__":
    
    """data_ids = list(range(16))
    area_lower_threshold = 80
    depth_filtering_threshold = 1.8
    nearest_interpolation = 450
    kubric_batch_validation(data_ids, area_lower_threshold, depth_filtering_threshold, nearest_interpolation, gt_mask=False)"""
    compute_metrics()
    #kubric_parameter_search(1, gt_mask=False)