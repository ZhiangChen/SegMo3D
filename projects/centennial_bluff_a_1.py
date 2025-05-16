import os
from ssfm.image_segmentation import ImageSegmentation
from ssfm.probabilistic_projection import *
from ssfm.segmo3d import *
from ssfm.post_processing import *
import time


scene_dir = '../data/centennial_bluff/mission_a'
pointcloud_path = os.path.join(scene_dir, 'SfM_products', 'a_downsampled_1.las')
associations_folder_path = os.path.join(scene_dir, 'associations_1')
segmentations_folder_path = os.path.join(scene_dir, 'segmentations')
photos_folder_path = os.path.join(scene_dir, 'DJI_photos')
camera_path = os.path.join(scene_dir, 'SfM_products', 'a.xml')
mesh_path = os.path.join(scene_dir, 'SfM_products', 'a.obj')
scene_name = 'mission_a_1'


image_path_list = [os.path.join(photos_folder_path, image) for image in os.listdir(photos_folder_path)]

# sort images based on the values of keyimages in file names
image_path_list = sorted(image_path_list, key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))

image_list = [image for image in os.listdir(photos_folder_path)]

# sort images based on the values of keyimages in file names
image_list = sorted(image_list, key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))



# pointcloud_projector = PointcloudProjection(depth_filtering_threshold=0.05, effective_depth = np.inf)

# pointcloud_projector.read_camera_parameters(camera_path)
# pointcloud_projector.read_mesh(mesh_path)
# pointcloud_projector.read_pointcloud(pointcloud_path)

# pointcloud_projector.parallel_batch_project_joblib(image_list, associations_folder_path, num_workers=16, save_depth=True)

segmentations_folder_path = "../data/centennial_bluff/mission_a/segmentations_class_filter"

# from ssfm.keyimage_associations_builder import *

# smc_solver = KeyimageAssociationsBuilder(image_list, associations_folder_path, segmentations_folder_path)

# smc_solver.build_associations()

# smc_solver.build_graph(20, device='cuda:3')
# smc_solver.add_camera_to_graph([camera_path], camera_type="Agisoft")

obr = SegMo3D(pointcloud_path, segmentations_folder_path, associations_folder_path, image_list=image_list, using_graph=True, radius=2, decaying=1, scene_name=scene_name)

# Run object registration
obr.segmo3d(iou_threshold=0.2, save_semantics_all=True, explicit_background=True)