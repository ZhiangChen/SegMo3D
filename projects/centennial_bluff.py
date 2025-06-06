import os
from ssfm.image_segmentation import ImageSegmentation
from ssfm.probabilistic_projection import *
from ssfm.segmo3d import *
from ssfm.post_processing import *
from ssfm.keyimage_associations_builder import *
import time

mission_name = 'b'
part = '1'
scene_dir = '../data/centennial_bluff_georectified/mission_{}'.format(mission_name)
pointcloud_path = os.path.join(scene_dir, 'SfM_products', '{}_downsampled_{}.las'.format(mission_name, part))
associations_folder_path = os.path.join(scene_dir, 'associations_{}'.format(part))
segmentations_folder_path = os.path.join(scene_dir, 'segmentations')
photos_folder_path = os.path.join(scene_dir, 'DJI_photos')
camera_path = os.path.join(scene_dir, 'SfM_products', '{}.xml'.format(mission_name))
mesh_path = os.path.join(scene_dir, 'SfM_products', '{}.obj'.format(mission_name))
scene_name = 'mission_{}_{}'.format(mission_name, part)

projection_association_option = False
segmentation_association_option = False
segmo3d_option = True
export_las_option = False
export_semantics_id = 800
iou_threshold = 0.5


print("mission_{}_{}".format(mission_name, part) + " started at {}".format(time.strftime("%Y-%m-%d %H:%M:%S")))


image_path_list = [os.path.join(photos_folder_path, image) for image in os.listdir(photos_folder_path)]
# sort images based on the values of keyimages in file names
image_path_list = sorted(image_path_list, key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
image_list = [image for image in os.listdir(photos_folder_path)]
# sort images based on the values of keyimages in file names
image_list = sorted(image_list, key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))

if projection_association_option:
    print("Building associations for projection...")
    pointcloud_projector = PointcloudProjection(depth_filtering_threshold=0.05, effective_depth = np.inf)
    pointcloud_projector.read_camera_parameters(camera_path)
    pointcloud_projector.read_mesh(mesh_path)
    pointcloud_projector.read_pointcloud(pointcloud_path)
    pointcloud_projector.parallel_batch_project_joblib(image_list, associations_folder_path, num_workers=16, save_depth=True)

segmentations_folder_path = "../data/centennial_bluff_georectified/mission_{}/segmentations_class_filter".format(mission_name)

if segmentation_association_option:
    print("Building associations for segmentation...")
    smc_solver = KeyimageAssociationsBuilder(image_list, associations_folder_path, segmentations_folder_path)
    smc_solver.build_associations()
    smc_solver.build_graph(20, device='cuda:4')
    smc_solver.add_camera_to_graph([camera_path], camera_type="Agisoft")

if segmo3d_option:
    print("Starting SegMo3D...")
    obr = SegMo3D(pointcloud_path, segmentations_folder_path, associations_folder_path, image_list=image_list, using_graph=True, radius=2, decaying=1, scene_name=scene_name)
    #obr.segmo3d(iou_threshold=0.5, save_semantics_all=True, explicit_background=True)
    obr.segmo3d(iou_threshold=iou_threshold, save_semantics=True, explicit_background=True)


if export_las_option:
    print("Exporting LAS files...")
    image_id = export_semantics_id
    #save_semantics_path = os.path.join(associations_folder_path, 'semantics', 'semantics_{}.npy'.format(image_id))
    #semantics_ids_path = os.path.join(associations_folder_path, 'semantics', '{}_segmentation_ids.npy'.format(image_id))
    #semantics_probs_path = os.path.join(associations_folder_path, 'semantics', '{}_segmentation_probs.npy'.format(image_id))

    #pc_segmentation_ids = np.load(semantics_ids_path)
    #pc_segmentation_probs = np.load(semantics_probs_path)

    #max_prob_indices = np.argmax(pc_segmentation_probs, axis=1)
    #semantics = pc_segmentation_ids[np.arange(len(max_prob_indices)), max_prob_indices]
    #np.save(save_semantics_path, semantics)
    
    semantics_folder_path = os.path.join(associations_folder_path, 'semantics', 'semantics_{}.npy'.format(image_id))
    save_las_path = os.path.join(associations_folder_path, 'semantics', 'semantics_{}.las'.format(image_id))
    add_semantics_to_pointcloud(pointcloud_path, semantics_folder_path, save_las_path, remove_small_N=30, nearest_interpolation=0)

    semantic_pc_file_path = save_las_path
    post_processing = PostProcessing(semantic_pc_file_path)
    post_processing.sort_semantic_ids(exclude_largest_semantic=False)
    save_las_path = os.path.join(associations_folder_path, 'semantics', 'semantics_{}_sorted.las'.format(image_id))
    post_processing.save_semantic_pointcloud(save_las_path)

