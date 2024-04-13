from ssfm.files import *
from ssfm.depth_image_rendering import *
import os
from numba import jit
import concurrent.futures
import time

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
            if z < z_buffer[py, px]:
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
def add_color_to_points(point2pixel, colors, segmentation, image_height, image_width, radius=2, decaying=2):
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
    def __init__(self) -> None:
        super().__init__()
        self.sfm_software = None
        self.tri_mesh = None

    def read_pointcloud(self, pointcloud_path):
        """
        Arguments:
            pointcloud_path (str): Path to the pointcloud file.
        """
        assert self.sfm_software is not None, 'Please read the camera parameters before reading point cloud.'
        self.points, self.colors = read_las_file(pointcloud_path)
        if self.sfm_software == 'Agisoft':
            self.colors = self.colors / 256
            

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

        self.camera_intrinsics = self.cameras['K'] 
        self.image_height = self.cameras['height']
        self.image_width = self.cameras['width']

    def read_segmentation(self, segmentation_path):
        """
        Arguments:
            segmentation_path (str): Path to the segmentation file.
        """
        assert os.path.exists(segmentation_path), 'Segmentation path does not exist.'
        self.segmentation = np.load(segmentation_path)  # for adding color to points
   
    def project(self, frame_key, depth_filtering_threshold=0.1):
        """
        Arguments:
            frame_key (str): The key of the camera frame to be projected onto.
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

        points_projected_d = np.matmul(points_camera_space, self.camera_intrinsics.T)
        points_projected = points_projected_d / points_projected_d[:, -1].reshape(-1, 1)
        # replace depth 
        points_projected[:, 2] = points_projected_d[:, 2]

        # Initialize image (2D array) and z-buffer
        image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)

        # Get depth image from mesh
        depth_mesh = self.render_depth_image(frame_key)

        # Update image and get associations
        image, z_buffer, pixel2point, point2pixel = update_image_and_associations(image, depth_mesh+depth_filtering_threshold, points_projected, self.colors, self.image_height, self.image_width)

        return image, pixel2point, point2pixel 
    
    def batch_project(self, frame_keys, save_folder_path):
        """
        Arguments:
            frame_keys (list): A list of frame keys. 
            save_folder_path (str): The path to save the images.
        """
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

        pixel2point_folder_path = os.path.join(save_folder_path, 'pixel2point')
        if not os.path.exists(pixel2point_folder_path):
            os.makedirs(pixel2point_folder_path)

        point2pixel_folder_path = os.path.join(save_folder_path, 'point2pixel')
        if not os.path.exists(point2pixel_folder_path):
            os.makedirs(point2pixel_folder_path)

        total = len(frame_keys)
        for i, frame_key in enumerate(frame_keys):
            print('Processing {}/{}: {}'.format(i+1, total, frame_key))
            t1 = time.time()
            image, pixel2point, point2pixel = self.project(frame_key)
            t2 = time.time()
            print('Time for projection: ', t2 - t1)

            # save pixel2point as npy
            pixel2point_path = os.path.join(pixel2point_folder_path, frame_key[:-4] + '.npy')
            np.save(pixel2point_path, pixel2point)
            # save point2pixel as npy
            point2pixel_path = os.path.join(point2pixel_folder_path, frame_key[:-4] + '.npy')
            np.save(point2pixel_path, point2pixel)

    def process_frame(self, frame_key, save_folder_path):
        print('Processing: {}'.format(frame_key))
        t1 = time.time()
        image, pixel2point, point2pixel = self.project(frame_key)
        t2 = time.time()
        print('Time for projection: ', t2 - t1)

        pixel2point_folder_path = os.path.join(save_folder_path, 'pixel2point')
        if not os.path.exists(pixel2point_folder_path):
            os.makedirs(pixel2point_folder_path)

        point2pixel_folder_path = os.path.join(save_folder_path, 'point2pixel')
        if not os.path.exists(point2pixel_folder_path):
            os.makedirs(point2pixel_folder_path)

        # save pixel2point as npy
        pixel2point_path = os.path.join(pixel2point_folder_path, frame_key[:-4] + '.npy')
        np.save(pixel2point_path, pixel2point)
        # save point2pixel as npy
        point2pixel_path = os.path.join(point2pixel_folder_path, frame_key[:-4] + '.npy')
        np.save(point2pixel_path, point2pixel)

        return point2pixel

    def parallel_batch_project(self, frame_keys, save_folder_path, num_workers=8):
        """
        Execute the processing of frames in parallel using multiprocessing.

        Arguments:
            frame_keys (list): A list of frame keys.
            save_folder_path (str): The path to save the images.
        """
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.process_frame, frame_key, save_folder_path) 
                       for frame_key in frame_keys]

            try:
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    # Process results or handle exceptions if any
                    print(f'Task {i+1}/{len(frame_keys)} completed')

            except KeyboardInterrupt:
                print("Interrupted by user, cancelling tasks...")
                # futures cancellation if needed
            except Exception as e:
                print(f"An error occurred: {e}")
            finally:
                print("Executor shut down.")

    def process_frame_joblib(self, frame_key, save_folder_path):
        # This is a wrapper for the original `process_frame` method, adapted for use with joblib.
        # It's essentially the same as `process_frame` but could be adjusted if needed.
        return self.process_frame(frame_key, save_folder_path)
    
    def parallel_batch_project_joblib(self, frame_keys, save_folder_path, num_workers=8):
        """
        Execute the processing of frames in parallel using joblib and display a progress bar with tqdm.

        Arguments:
            frame_keys (list): A list of frame keys.
            save_folder_path (str): The path to save the images.
            num_workers (int): The number of worker processes.
        """
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

        # Wrap the iterable (frame_keys) with tqdm for the progress bar
        # tqdm provides a description and the total count for better progress visibility
        tasks = tqdm(frame_keys, desc="Processing frames", total=len(frame_keys))

        # Use Joblib for parallel processing with the tqdm-wrapped iterable
        Parallel(n_jobs=num_workers)(
            delayed(self.process_frame_joblib)(frame_key, save_folder_path) for frame_key in tasks)

    def build_associations_keyimage(self, read_folder_path, segmentation_folder_path):
        """
        Arguments:
            read_folder_path (str): The path to the folder containing the point2pixel files.
            pixel2point_folder_path (str): The path to the folder containing the pixel2point files.
            segmentation_folder_path (str): The path to the folder containing the segmentation files.
        """
        pixel2point_folder_path = os.path.join(read_folder_path, 'pixel2point')
        assert os.path.exists(pixel2point_folder_path), 'Pixel2point folder path does not exist.'
        pixel2point_file_paths = [os.path.join(pixel2point_folder_path, f) for f in os.listdir(pixel2point_folder_path) if f.endswith('.npy')]
        pixel2point_file_paths.sort()
        N_images = len(pixel2point_file_paths)
        
        # get the number of points
        point2pixel_folder_path = os.path.join(read_folder_path, 'point2pixel')  
        assert os.path.exists(point2pixel_folder_path), 'Point2pixel folder path does not exist.'  
        point2pixel_file_paths = [os.path.join(point2pixel_folder_path, f) for f in os.listdir(point2pixel_folder_path) if f.endswith('.npy')]  
        point2pixel = np.load(point2pixel_file_paths[0])
        N_points = point2pixel.shape[0]

        assert os.path.exists(segmentation_folder_path), 'Segmentation folder path does not exist.'
        segmentation_file_paths = [os.path.join(segmentation_folder_path, f) for f in os.listdir(segmentation_folder_path) if f.endswith('.npy')]
        segmentation_file_paths.sort()

        # check if the number of images is the same
        assert len(segmentation_file_paths) == N_images, 'The number of images is not the same as the number of segmentation files.'
        segmentation_pixel2point_pairs = []
        for i in range(len(segmentation_file_paths)):
            segmentation_file_path = segmentation_file_paths[i]
            pixel2point_file_path = pixel2point_file_paths[i]
            # check if the base of the file name is the same
            assert os.path.basename(segmentation_file_path)[:-4] == os.path.basename(pixel2point_file_path)[:-4], 'The base of the file name is not the same.'
            segmentation_pixel2point_pairs.append((segmentation_file_path, pixel2point_file_path))
        
        # build associations
        associations_keyimage = np.full((N_points, N_images), False, dtype=bool)

        for i in range(len(point2pixel_file_paths)):
            segmentation = np.load(segmentation_pixel2point_pairs[i][0])
            pixel2point = np.load(segmentation_pixel2point_pairs[i][1])
            pixel2point[segmentation == -1] = -1
            valid_point_ids = pixel2point[pixel2point != -1]
            associations_keyimage[valid_point_ids, i] = True

        # save associations
        save_file_path = os.path.join(read_folder_path, 'associations_keyimage.npy')
        np.save(save_file_path, associations_keyimage)


if __name__ == "__main__":
    from ssfm.image_segmentation import *
    import time
    site = 'box_canyon'

    flag = True  # True: project semantics from one image to the point cloud.
    batch_flag = False  # True: save the associations of all images in sequence.
    Parallel_batch_flag = False # True: save the associations of all images in parallel.
    keyimage_flag = False  # True: build associations_keyimage.npy

    if flag:
        if site == 'box_canyon':
            # segment the images
            #image_segmentor = ImageSegmentation(sam_params)        
            #image_path = '../../data/mission_2/DJI_0246.JPG'
            #masks = image_segmentor.predict(image_path)
            #image_segmentor.save_npy(masks, '../../data/DJI_0246.npy')

            # project the point cloud
            pointcloud_projector = PointcloudProjection()
            pointcloud_projector.read_camera_parameters('../../data/box_canyon_park/SfM_products/agisoft_cameras.xml')
            pointcloud_projector.read_pointcloud('../../data/box_canyon_park/SfM_products/agisoft_model.las')
            
            pointcloud_projector.read_mesh('../../data/box_canyon_park/SfM_products/model.obj')
            pointcloud_projector.read_segmentation('../../data/box_canyon_park/segmentations/DJI_0246.npy')
            image, pixel2point, point2pixel = pointcloud_projector.project('DJI_0246.JPG')

            # add color to points
            t1 = time.time()
            radius = 0
            decaying = 2
            colors = add_color_to_points(point2pixel, pointcloud_projector.colors, pointcloud_projector.segmentation, pointcloud_projector.image_height, pointcloud_projector.image_width, radius, decaying)
            t2 = time.time()
            print('Time for adding colors: ', t2 - t1)

            write_las(pointcloud_projector.points, colors, "../../data/box_canyon_park/depth_filter_segmentation.las")

        elif site == 'courtwright':
            # project the point cloud
            pointcloud_projector = PointcloudProjection()
            #pointcloud_projector.read_camera_parameters('../../data/courtright/SfM_products/cameras.json', '../../data/courtright/SfM_products/shots.geojson')
            #pointcloud_projector.read_pointcloud('../../data/courtright/SfM_products/model.las')
            pointcloud_projector.read_camera_parameters('../../data/courtright/SfM_products/agisoft_cameras.xml')
            pointcloud_projector.read_pointcloud('../../data/courtright/SfM_products/agisoft_model.las')
            pointcloud_projector.read_segmentation('../../data/courtright/segmentations/DJI_0595.npy')
            t1 = time.time()
            image, pixel2point, point2pixel  = pointcloud_projector.project('DJI_0595.JPG')
            t2 = time.time()
            print('Time for projection: ', t2 - t1)

            # save image
            image_path = '../../data/DJI_0595_agisoft.png'
            cv2.imwrite(image_path, image)

            
            # add color to points
            t1 = time.time()
            radius = 2
            decaying = 2
            colors = add_color_to_points(point2pixel, pointcloud_projector.colors, pointcloud_projector.segmentation, pointcloud_projector.image_height, pointcloud_projector.image_width, radius, decaying)
            t2 = time.time()
            print('Time for adding colors: ', t2 - t1)

            write_las(pointcloud_projector.points, colors, "../../data/courtright_test_agisoft.las")

    else:
        pass


    if batch_flag:
        # project the point cloud
        pointcloud_projector = PointcloudProjection()
        pointcloud_projector.read_pointcloud('../../data/model.las')
        pointcloud_projector.read_camera_parameters('../../data/camera.json', '../../data/shots.geojson')

        # batch project
        image_folder_path = '../../data/mission_2'
        save_folder_path = '../../data/mission_2_associations_'

        image_list = [f for f in os.listdir(image_folder_path) if f.endswith('.JPG')]
        pointcloud_projector.batch_project(image_list, save_folder_path)

    else:
        pass


    if Parallel_batch_flag:
        if site == 'box_canyon':
            # project the point cloud
            pointcloud_projector = PointcloudProjection()
            pointcloud_projector.read_pointcloud('../../data/box_canyon_park/SfM_products/model.las')
            pointcloud_projector.read_camera_parameters('../../data/box_canyon_park/SfM_products/camera.json', '../../data/box_canyon_park/SfM_products/shots.geojson')

            # batch project
            image_folder_path = '../../data/box_canyon_park/DJI_photos'
            save_folder_path = '../../data/box_canyon_park/associations'

            image_list = [f for f in os.listdir(image_folder_path) if f.endswith('.JPG')]
            pointcloud_projector.parallel_batch_project(image_list, save_folder_path)

        elif site == 'courtwright':
            # project the point cloud
            pointcloud_projector = PointcloudProjection()
            pointcloud_projector.read_pointcloud('../../data/courtwright/courtwright.las')
            pointcloud_projector.read_camera_parameters('../../data/courtwright/cameras.json', '../../data/courtwright/shots.geojson')

            # batch project
            image_folder_path = '../../data/courtwright/photos'
            save_folder_path = '../../data/courtwright/associations'

            image_list = [f for f in os.listdir(image_folder_path) if f.endswith('.JPG')]
            pointcloud_projector.parallel_batch_project(image_list, save_folder_path)

    else:
        pass

    if keyimage_flag:
        pointcloud_projector = PointcloudProjection()
        t1 = time.time()
        pointcloud_projector.build_associations_keyimage('../../data/mission_2_associations_parallel', '../../data/mission_2_segmentations')
        t2 = time.time()
        print('Time for building associations: ', t2 - t1)
