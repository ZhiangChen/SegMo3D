from ssfm.files import *
from ssfm.depth_image_rendering import *
import os
from numba import jit
import concurrent.futures
import time
import yaml

from joblib import Parallel, delayed
from tqdm import tqdm


# association on all points
@jit(nopython=True)
def update_image_and_associations(image, z_buffer, points, colors, height, width):
    pixel2point = np.full((height, width), -1, dtype=np.int32)

    N_points = points.shape[0]

    for i in range(N_points):
        x, y, z = points[i]
        color = colors[i]
        px, py = int(x), int(y)

        if 0 <= px < width and 0 <= py < height:
            if 0 < z < z_buffer[py, px]:
                z_buffer[py, px] = z
                pixel2point[py, px] = i
                image[py, px] = color

    # get the pixel of valid associations
    u_indices, v_indices = np.where(pixel2point != -1)
    point2pixel = np.full((N_points, 2), -1, dtype=np.int16)
    for idx in range(len(u_indices)):
        point2pixel[pixel2point[u_indices[idx], v_indices[idx]]] = np.array([u_indices[idx], v_indices[idx]])

    return image, z_buffer, pixel2point, point2pixel


@jit(nopython=True)
def inquire_semantics(u, v, padded_segmentation, normalized_likelihoods, likelihoods, radius=3):
    """
    Arguments:
        u (int): The u coordinate of the pixel in the original image.
        v (int): The v coordinate of the pixel in the original image.
        padded_segmentation (np.array): The segmentation image padded by the size of radius with -1.
        normalized_likelihoods (np.array): A 1D array of shape (N,) where N is the number of semantic classes. The index is the semantic.
        likelihoods (np.array): A 1D array of shape (N,) where N is the number of pixels with center of (u, v) and given radius. 
            The index is the sequence number of the pixel and the value is the precomputed likelihood.
        radius (int): The radius of the circle in segmentation image size. 

    Returns:
        normalized_likelihoods (np.array): A 1D array of shape (N,) where N is the number of semantic classes. The index is the semantic 
            id and the value is the normalized likelihood. 
    """
    # reset normalized_likelihoods
    normalized_likelihoods[:] = 0
    
    # Extract semantic ids and corresponding likelihoods
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            semantic_id = padded_segmentation[u + j + radius, v + i + radius]
            if semantic_id != -1:
                normalized_likelihoods[int(semantic_id)] += likelihoods[int((i + radius) * (2 * radius + 1) + (j + radius))]

    # Normalize the likelihoods
    total_likelihood = np.sum(normalized_likelihoods)
    if total_likelihood > 0:
        normalized_likelihoods /= total_likelihood

    return normalized_likelihoods


@jit(nopython=True)
def compute_gaussian_likelihood(radius, decaying):
    """
    Arguments:
        radius (int): The radius of the circle in segmentation image size. 
        decaying (float): The decaying factor.

    Returns:
        likelihoods (np.array): A 1D array of shape (N,) where N is the number of pixels with center of (u, v) and given radius. 
            The index is the sequence number of the pixel and the value is the precomputed likelihood.
    """
    likelihoods = np.zeros((2*radius+1, 2*radius+1))
    
    for i in range(2*radius+1):
        for j in range(2*radius+1):
            likelihoods[i, j] = np.exp(-decaying * np.sqrt((i-radius)**2 + (j-radius)**2))
    likelihoods = likelihoods.reshape(-1)
    return likelihoods


@jit(nopython=True)
def add_color_to_points(point2pixel, original_colors, segmentation, image_height, image_width, radius=2, decaying=2):
    """
    Arguments:
        point2pixel (np.array): A 2D array of shape (N, 2) where N is the number of points. Each row is (u, v). 
        colors (np.array): A 2D array of shape (N, 3) where N is the number of points.
        segmentation (np.array): The segmentation image.
        image_height (int): The height of the original image.
        image_width (int): The width of the original image.
        radius (int): The radius of the circle in segmentation image size.
        decaying (float): The decaying factor.

    Returns:
        colors (np.array): A 2D array of shape (N, 3) where N is the number of points. The index is the sequence number of the point and the value is the color.
    """
    colors = original_colors.copy()
    # precompute likelihoods
    likelihoods = compute_gaussian_likelihood(radius, decaying)

    # Vectorized random color generation
    N = int(segmentation.max() + 1)
    random_colors = np.random.randint(0, 255, (N, 3)).astype(np.int64)
    
    # pad segmentation image by the size of radius with -1
    padded_segmentation = -np.ones((2*radius+image_height+2, 2*radius+image_width+2))
    padded_segmentation[radius+1:radius+image_height+1, radius+1:radius+image_width+1] = segmentation

    # Count the number of points for each semantic class
    #point_count = np.zeros(N, dtype=np.int64)
    #for i in range(associations.shape[0]):
    for point_index in range(point2pixel.shape[0]):
        u, v = point2pixel[point_index]
        if u == -1 or v == -1:
            continue
        
        normalized_likelihoods = np.zeros(N, dtype=np.float64)
        normalized_likelihoods = inquire_semantics(u, v, padded_segmentation, normalized_likelihoods, likelihoods, radius=radius)

        if normalized_likelihoods.max() > 0:
            semantic_id = np.argmax(normalized_likelihoods)
            colors[point_index] = random_colors[int(semantic_id) - 1]
            #point_count[int(semantic_id)] += 1
    
    #print('Point count: ', point_count)
    return colors


class PointcloudProjection(DepthImageRendering):
    """
    This class is used to project the point cloud onto the image using projection. 
    1. Read the point cloud and camera parameters: PointcloudProjection.read_pointcloud and PointcloudProjection.read_camera_parameters
    2. Project the point cloud onto the image: PointcloudProjection.project
    """
    def __init__(self, depth_filtering_threshold=0) -> None:
        super().__init__()
        self.sfm_software = None
        self.tri_mesh = None
        self.depth_filtering_threshold= depth_filtering_threshold

    def read_pointcloud(self, pointcloud_path):
        """
        Arguments:
            pointcloud_path (str): Path to the pointcloud file.
        """
        assert self.sfm_software is not None, 'Please read the camera parameters before reading point cloud.'
        self.points, self.colors = read_las_file(pointcloud_path)
        if self.sfm_software == 'Agisoft':
            self.colors = (self.colors / 256).astype(np.uint8)
        elif self.sfm_software == 'Scannet':
            self.colors = (self.colors / 256).astype(np.uint8)
            self.tri_mesh = "depth_images"

    def read_scannet_mesh(self, mesh_vertices_color_path):
        """
        Arguments:
            mesh_vertices_color_path (str): Path to the mesh vertices color file.
        """
        mesh_vertices_color = np.load(mesh_vertices_color_path)
        self.points = mesh_vertices_color[0]
        self.colors = mesh_vertices_color[1]
        # the color is normalized from 0 to 1. Convert it to 0 to 255
        self.colors = (self.colors * 255).astype(np.int8)

        self.tri_mesh = "depth_images"
            

    def read_camera_parameters(self, camera_parameters_path, additional_camera_parameters_path=None):
        """
        Arguments:
            camera_parameters_path (str): Path to the camera parameters file.
            additional_camera_parameters_path (str): Path to the additional camera parameters file.
        """
        if additional_camera_parameters_path is not None:
            # WebODM
            self.cameras = read_camera_parameters_webodm(camera_parameters_path, additional_camera_parameters_path)
            self.sfm_software = "WebODM"
        else:
            # Agisoft
            self.cameras = read_camera_parameters_agisoft(camera_parameters_path)
            self.sfm_software = "Agisoft"

        self.camera_intrinsics_color = self.cameras['K'] 
        self.image_height = self.cameras['height']
        self.image_width = self.cameras['width']
        
    def read_scannet_camera_parameters(self, ssfm_scene_folder_path):
        """
        Arguments:
            ssfm_scene_folder_path (str): Path to the scene folder.
        """
        self.ssfm_scene_folder_path = ssfm_scene_folder_path
        self.cameras = read_camera_scannet(ssfm_scene_folder_path)
        self.camera_intrinsics_color = self.cameras['intrinsic_color']  # 4 by 4 matrix
        self.camera_intrinsics_color = self.camera_intrinsics_color[:3, :3]
        self.camera_intrinsics_depth = self.cameras['intrinsic_depth']
        self.camera_intrinsics_depth = self.camera_intrinsics_depth[:3, :3]
        self.image_height = self.cameras['height']
        self.image_width = self.cameras['width']

        self.sfm_software = "Scannet"
        

    def read_segmentation(self, segmentation_path):
        """
        Arguments:
            segmentation_path (str): Path to the segmentation file.
        """
        assert os.path.exists(segmentation_path), 'Segmentation path does not exist.'
        self.segmentation = np.load(segmentation_path)  # for adding color to points
   
    def project(self, frame_key):
        """
        Arguments:
            frame_key (str): The original image names to be projected onto. E.g., 'DJI_0313.JPG'.
            depth_filtering_threshold (float): The threshold for depth filtering. depth_point (> depth_mesh + depth_filtering_threshold) will be filtered out.
        """
        assert self.tri_mesh is not None, 'Please read the mesh first.'
        assert self.sfm_software is not None, 'Please read the camera parameters first.'

        self.extrinsic_matrix = self.cameras[frame_key]

        # Transform the point cloud using the extrinsic matrix
        points_homogeneous = np.hstack((self.points, np.ones((len(self.points), 1))))

        extrinsic_matrix_inv = np.linalg.inv(self.extrinsic_matrix)

        points_transformed = np.matmul(points_homogeneous, extrinsic_matrix_inv.T)

        # Project the points using the intrinsic matrix
        # Drop the homogeneous component (w)
        points_camera_space = points_transformed[:, :3]

        points_projected_d = np.matmul(points_camera_space, self.camera_intrinsics_color.T)
        points_projected = points_projected_d / points_projected_d[:, -1].reshape(-1, 1)
        # replace depth 
        points_projected[:, 2] = points_projected_d[:, 2]

        # Initialize image (2D array) and z-buffer
        image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)

        if self.depth_filtering_threshold == 0:
            z_buffer = np.full((self.image_height, self.image_width), np.inf)
        else:
            if self.tri_mesh == "depth_images":
                depth_image_path = os.path.join(self.ssfm_scene_folder_path, 'associations/depth', frame_key[:-4] + '.npy')
                depth_mesh = np.load(depth_image_path)
                # resize the depth image to the size of the original image
                depth_mesh = cv2.resize(depth_mesh, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)

            else:
                # Get depth image from mesh
                depth_mesh = self.render_depth_image(frame_key)
        
        z_buffer = depth_mesh + self.depth_filtering_threshold
            
        image, z_buffer, pixel2point, point2pixel = update_image_and_associations(image, z_buffer, points_projected, self.colors, self.image_height, self.image_width)


        return z_buffer, pixel2point, point2pixel
    

    def process_frame(self, file_name, save_folder_path, depth_folder_path=None):
        print('Processing: {}'.format(file_name))
        t1 = time.time()
        depth_image, pixel2point, point2pixel = self.project(file_name)
        t2 = time.time()
        print('Time for projection: ', t2 - t1)

        pixel2point_folder_path = os.path.join(save_folder_path, 'pixel2point')

        point2pixel_folder_path = os.path.join(save_folder_path, 'point2pixel')

        # save pixel2point as npy
        pixel2point_path = os.path.join(pixel2point_folder_path, file_name[:-4] + '.npy')
        np.save(pixel2point_path, pixel2point)
        # save point2pixel as npy
        point2pixel_path = os.path.join(point2pixel_folder_path, file_name[:-4] + '.npy')
        np.save(point2pixel_path, point2pixel)

        if depth_folder_path is not None:
            depth_folder_path = os.path.join(depth_folder_path, file_name[:-4] + '.npy')
            np.save(depth_folder_path, depth_image)

        return point2pixel

    
    def parallel_batch_project_joblib(self, frame_keys, save_folder_path, num_workers=8, save_depth=False):
        """
        Execute the processing of frames in parallel using joblib and display a progress bar with tqdm.

        Arguments:
            frame_keys (list): A list of frame keys.
            save_folder_path (str): The path to save the images.
            num_workers (int): The number of worker processes.
        """
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

        pixel2point_folder_path = os.path.join(save_folder_path, 'pixel2point')
        if not os.path.exists(pixel2point_folder_path):
            os.makedirs(pixel2point_folder_path)

        point2pixel_folder_path = os.path.join(save_folder_path, 'point2pixel')
        if not os.path.exists(point2pixel_folder_path):
            os.makedirs(point2pixel_folder_path)

        if save_depth:
            depth_folder_path = os.path.join(save_folder_path, 'depth')
            if not os.path.exists(depth_folder_path):
                os.makedirs(depth_folder_path)

            # Wrap the iterable (frame_keys) with tqdm for the progress bar
            # tqdm provides a description and the total count for better progress visibility
            tasks = tqdm(frame_keys, desc="Processing frames", total=len(frame_keys))

            # Use Joblib for parallel processing with the tqdm-wrapped iterable
            Parallel(n_jobs=num_workers)(
                delayed(self.process_frame)(frame_key, save_folder_path, depth_folder_path) for frame_key in tasks)
        else:
            tasks = tqdm(frame_keys, desc="Processing frames", total=len(frame_keys))

            # Use Joblib for parallel processing with the tqdm-wrapped iterable
            Parallel(n_jobs=num_workers)(
                delayed(self.process_frame)(frame_key, save_folder_path) for frame_key in tasks)
            

    def parallel_batch_project_joblib_scannet(self, ):
        pass


if __name__ == "__main__":
    from ssfm.image_segmentation import *
    import time
    site = 'scannet'  # 'box_canyon', 'courtright', 'scannet'

    single_projection_flag = True  # True: project semantics from one image to the point cloud.

    if single_projection_flag:
        if site == 'box_canyon':
            # segment the images
            #image_segmentor = ImageSegmentation(sam_params)        
            #image_path = '../../data/mission_2/DJI_0246.JPG'
            #masks = image_segmentor.predict(image_path)
            #image_segmentor.save_npy(masks, '../../data/DJI_0246.npy')

            # project the point cloud
            pointcloud_projector = PointcloudProjection(depth_filtering_threshold=0.2)
            pointcloud_projector.read_camera_parameters('../../data/box_canyon_park/SfM_products/agisoft_cameras.xml')
            pointcloud_projector.read_mesh('../../data/box_canyon_park/SfM_products/model.obj')
            pointcloud_projector.read_pointcloud('../../data/box_canyon_park/SfM_products/agisoft_model.las')
            
            
            pointcloud_projector.read_segmentation('../../data/box_canyon_park/segmentations/DJI_0313.npy')
            image, pixel2point, point2pixel = pointcloud_projector.project('DJI_0313.JPG')

            # add color to points
            t1 = time.time()
            radius = 2
            decaying = 2
            colors = add_color_to_points(point2pixel, pointcloud_projector.colors, pointcloud_projector.segmentation, pointcloud_projector.image_height, pointcloud_projector.image_width, radius, decaying)
            t2 = time.time()
            print('Time for adding colors: ', t2 - t1)

            write_las(pointcloud_projector.points, colors, "../../data/box_canyon_park/313_depth_filter_segmentation.las")

        elif site == 'courtright':
            # project the point cloud
            t0 = time.time()
            pointcloud_projector = PointcloudProjection(depth_filtering_threshold=0.05)
            pointcloud_projector.read_camera_parameters('../../data/courtright/SfM_products/agisoft_cameras.xml')
            pointcloud_projector.read_mesh('../../data/courtright/SfM_products/agisoft_model.obj')
            pointcloud_projector.read_pointcloud('../../data/courtright/SfM_products/agisoft_model.las')
            pointcloud_projector.read_segmentation('../../data/courtright/segmentations_filtered/DJI_0576.npy')
            t1 = time.time()
            print('Time for reading data: ', t1 - t0)
            image, pixel2point, point2pixel  = pointcloud_projector.project('DJI_0576.JPG')
            t2 = time.time()
            print('Time for projection: ', t2 - t1)
            
            # add color to points
            radius = 2
            decaying = 2
            colors = add_color_to_points(point2pixel, pointcloud_projector.colors, pointcloud_projector.segmentation, pointcloud_projector.image_height, pointcloud_projector.image_width, radius, decaying)
            t3 = time.time()
            print('Time for adding colors: ', t3 - t2)

            write_las(pointcloud_projector.points, colors, "../../data/courtright/courtright_576.las")
            t4 = time.time()
            print('Time for writing las: ', t4 - t3)

        elif site == 'scannet':
            # project the point cloud
            t0 = time.time()
            pointcloud_projector = PointcloudProjection(depth_filtering_threshold=10.05)
            pointcloud_projector.read_scannet_camera_parameters('../../data/scene0000_00')
            #pointcloud_projector.read_scannet_mesh('../../data/scene0000_00/reconstructions/mesh_vertices_color.npy')
            pointcloud_projector.read_pointcloud('../../data/scene0000_00/reconstructions/scene0000_00.las')
            pointcloud_projector.read_segmentation('../../data/scene0000_00/segmentations/0.npy')
            t1 = time.time()
            print('Time for reading data: ', t1 - t0)
            image, pixel2point, point2pixel  = pointcloud_projector.project('0.jpg')
            t2 = time.time()
            print('Time for projection: ', t2 - t1)
            # add color to points
            radius = 2
            decaying = 2
            colors = add_color_to_points(point2pixel, pointcloud_projector.colors, pointcloud_projector.segmentation, pointcloud_projector.image_height, pointcloud_projector.image_width, radius, decaying)
            t3 = time.time()
            print('Time for adding colors: ', t3 - t2)

            write_las(pointcloud_projector.points, colors, "../../data/scene0000_00/scene0000_00.las")
            t4 = time.time()
            print('Time for writing las: ', t4 - t3)

    else:
        pass



