from ssfm.files import *
import os
from numba import jit
import concurrent.futures

@jit(nopython=True)
def update_image_and_associations(image, z_buffer, points, colors, height, width):
    """
    Arguments:
        image (np.array): The image to be updated. 2D array of shape (height, width, 3) with zeros.
        z_buffer (np.array): The z-buffer to be updated. 2D array of shape (height, width) with inf.
        points (np.array): The points to be projected onto the image. 2D array of shape (N, 3) where N is the number of points.
        colors (np.array): The colors of the points. 2D array of shape (N, 3) where N is the number of points.
        height (int): The height of the image. 
        width (int): The width of the image. 

    Returns:
        image (np.array): The updated image. 
        z_buffer (np.array): The updated z-buffer. 
        associations (np.array): A 2D array of shape (N, 3) where N is the number of valid points that are projected onto the image. For each point, the first two elements are the u, v coordinate of the point in the image and the third element is the index of the point in the original points array. u is the axis of width and v is the axis of height.
    """
    associations_pixel = []
    associations_point = []

    for i in range(points.shape[0]):
        x, y, z = points[i]
        color = colors[i]
        px, py = int(x), int(y)

        if 0 <= px < width and 0 <= py < height:
            if z < z_buffer[py, px]:
                if z_buffer[py, px] == np.inf:
                    associations_pixel.append([px, py])
                    associations_point.append(i)
                else:
                    index = associations_pixel.index([px, py])
                    associations_point[index] = i

                z_buffer[py, px] = z
                image[py, px] = color                

    associations = np.hstack((np.array(associations_pixel), np.array(associations_point).reshape(-1, 1)))

    return image, z_buffer, associations



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

    # Extract semantic ids and corresponding likelihoods
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            semantic_id = padded_segmentation[v + j + radius, u + i + radius]
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
def add_color_to_points(associations, colors, segmentation, image_height, image_width, radius=2, decaying=2):
    """
    Arguments:
        associations (np.array): A 2D array of shape (N, 3) where N is the number of valid points. For each point, 
            the first two elements are the u, v coordinate of the point in the image and the third element is the 
            index of the point in the original points array. u is the axis of width and v is the axis of height.
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
    N = int(np.ceil(segmentation.max()))
    random_colors = np.random.randint(0, 255, (N, 3)).astype(np.int64)
    
    # pad segmentation image by the size of radius with -1
    padded_segmentation = -np.ones((2*radius+image_height+2, 2*radius+image_width+2))
    padded_segmentation[radius+1:radius+image_height+1, radius+1:radius+image_width+1] = segmentation

    # Allocate normalized_likelihoods outside the loop
    normalized_likelihoods = np.zeros(int(segmentation.max() + 1), dtype=np.float64)

    for i in range(associations.shape[0]):
        u, v, point_index = associations[i]
        normalized_likelihoods = inquire_semantics(u, v, padded_segmentation, normalized_likelihoods, likelihoods, radius=2)

        if normalized_likelihoods.max() > 0:
            semantic_id = np.argmax(normalized_likelihoods)
            colors[point_index] = random_colors[int(semantic_id) - 1]
    
    return colors



class PointcloudProjection(object):
    """
    This class is used to project the point cloud onto the image using projection. 
    1. Read the point cloud and camera parameters: PointcloudProjection.read_pointcloud and PointcloudProjection.read_camera_parameters
    2. Project the point cloud onto the image: PointcloudProjection.project
    """
    def __init__(self) -> None:
        pass

    def read_pointcloud(self, pointcloud_path):
        """
        Arguments:
            pointcloud_path (str): Path to the pointcloud file.
        """
        self.points, self.colors = read_las_file(pointcloud_path)

    def read_camera_parameters(self, camera_parameters_path, additional_camera_parameters_path=None):
        """
        Arguments:
            camera_parameters_path (str): Path to the camera parameters file.
            additional_camera_parameters_path (str): Path to the additional camera parameters file.
        """
        if additional_camera_parameters_path is not None:
            # WebODM
            self.cameras = read_camera_parameters_webodm(camera_parameters_path, additional_camera_parameters_path)
        else:
            # Agisoft
            self.cameras = read_camera_parameters_agisoft(camera_parameters_path)

    def read_segmentation(self, segmentation_path):
        """
        Arguments:
            segmentation_path (str): Path to the segmentation file.
        """
        assert os.path.exists(segmentation_path), 'Segmentation path does not exist.'
        self.segmentation = np.load(segmentation_path)

    def project(self, frame_key):
        """
        Arguments:
            frame_key (str): The key of the camera frame to be projected onto.
        """
        camera_intrinsics = self.cameras['K'] 
        extrinsic_matrix = self.cameras[frame_key]
        image_height = self.cameras['height']
        image_width = self.cameras['width']

        self.image_height = image_height
        self.image_width = image_width

        # Transform the point cloud using the extrinsic matrix
        points_homogeneous = np.hstack((self.points, np.ones((len(self.points), 1))))

        extrinsic_matrix_inv = np.linalg.inv(extrinsic_matrix)

        points_transformed = np.matmul(points_homogeneous, extrinsic_matrix_inv.T)

        # Project the points using the intrinsic matrix
        # Drop the homogeneous component (w)
        points_camera_space = points_transformed[:, :3]

        points_projected = np.matmul(points_camera_space, camera_intrinsics.T)
        points_projected /= points_projected[:, -1].reshape(-1, 1)

        # Initialize image (2D array) and z-buffer
        image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        z_buffer = np.full((image_height, image_width), np.inf)

        # Assuming points_projected and colors are NumPy arrays
        x, y, z = points_projected.T
        px, py = x.astype(int), y.astype(int)

        # Update image and get associations
        image, z_buffer, associations  = update_image_and_associations(image, z_buffer, points_projected, self.colors, image_height, image_width)

        return image, associations
    
    def batch_project(self, frame_keys, save_folder_path):
        """
        Arguments:
            frame_keys (list): A list of frame keys.
            save_folder_path (str): The path to save the images.
        """
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

        total = len(frame_keys)
        for i, frame_key in enumerate(frame_keys):
            print('Processing {}/{}: {}'.format(i+1, total, frame_key))
            t1 = time.time()
            image, associations = self.project(frame_key)
            t2 = time.time()
            print('Time for projection: ', t2 - t1)

            associations_path = os.path.join(save_folder_path, frame_key[:-4] + '.npy')
            # save associations as npy
            np.save(associations_path, associations)

    def process_frame(self, frame_key, save_folder_path):
        print('Processing: {}'.format(frame_key))
        t1 = time.time()
        image, associations = self.project(frame_key)
        t2 = time.time()
        print('Time for projection: ', t2 - t1)

        associations_path = os.path.join(save_folder_path, frame_key[:-4] + '.npy')
        # save associations as npy
        np.save(associations_path, associations)

    def parallel_batch_project(self, frame_keys, save_folder_path, num_workers=8):
        """
        Arguments:
            frame_keys (list): A list of frame keys.
            save_folder_path (str): The path to save the images.
        """
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
        futures = [executor.submit(self.process_frame, frame_key, save_folder_path) for frame_key in frame_keys]

        try:
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                # Process results or handle exceptions if any
                print(f'Task {i+1}/{len(frame_keys)} completed')

        except KeyboardInterrupt:
            print("Interrupted by user, cancelling tasks...")
            for future in futures:
                future.cancel()
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            executor.shutdown(wait=False)
            print("Executor shut down.")


if __name__ == "__main__":
    from ssfm.image_segmentation import *
    import time
    
    flag = False  # True: project semantics from one image to the point cloud.
    if flag:
        # segment the images
        image_segmentor = ImageSegmentation(sam_params)        
        image_path = '../../data/mission_2/DJI_0247.JPG'
        masks = image_segmentor.predict(image_path)
        image_segmentor.save_npy(masks, '../../data/mission_2/DJI_0247.npy')

        # project the point cloud
        pointcloud_projector = PointcloudProjection()
        pointcloud_projector.read_pointcloud('../../data/model.las')
        pointcloud_projector.read_camera_parameters('../../data/camera.json', '../../data/shots.geojson')
        pointcloud_projector.read_segmentation('../../data/mission_2/DJI_0247.npy')
        image, associations = pointcloud_projector.project('DJI_0247.JPG')

        # add color to points
        t1 = time.time()
        colors = add_color_to_points(associations, pointcloud_projector.colors, pointcloud_projector.segmentation, pointcloud_projector.image_height, pointcloud_projector.image_width)
        t2 = time.time()
        print('Time for adding colors: ', t2 - t1)

        write_las(pointcloud_projector.points, colors, "../../data/test.las")

    else:
        pass

    batch_flag = False  # True: save the associations of all images.
    if batch_flag:
        # project the point cloud
        pointcloud_projector = PointcloudProjection()
        pointcloud_projector.read_pointcloud('../../data/model.las')
        pointcloud_projector.read_camera_parameters('../../data/camera.json', '../../data/shots.geojson')

        # batch project
        image_folder_path = '../../data/mission_2'
        save_folder_path = '../../data/mission_2_associations'

        image_list = [f for f in os.listdir(image_folder_path) if f.endswith('.JPG')]
        pointcloud_projector.batch_project(image_list, save_folder_path)

    else:
        pass

    Parallel_batch_flag = True
    if Parallel_batch_flag:
        # project the point cloud
        pointcloud_projector = PointcloudProjection()
        pointcloud_projector.read_pointcloud('../../data/model.las')
        pointcloud_projector.read_camera_parameters('../../data/camera.json', '../../data/shots.geojson')

        # batch project
        image_folder_path = '../../data/mission_2'
        save_folder_path = '../../data/mission_2_associations_parallel'

        image_list = [f for f in os.listdir(image_folder_path) if f.endswith('.JPG')]
        pointcloud_projector.parallel_batch_project(image_list, save_folder_path)

    else:
        pass






