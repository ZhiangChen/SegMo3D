from ssfm.files import *
import os
from numba import jit

@jit(nopython=True)
def update_image_and_associations(image, z_buffer, points, colors, height, width):
    # Initialize an array for associations with the same length as points array
    associations = np.full((points.shape[0], 2), -1, dtype=np.int32)  # -1 indicates no valid association

    for i in range(points.shape[0]):
        x, y, z = points[i]
        color = colors[i]
        px, py = int(x), int(y)

        if 0 <= px < width and 0 <= py < height:
            if z < z_buffer[py, px]:
                z_buffer[py, px] = z
                image[py, px] = color
                associations[i] = [px, py]

    return image, z_buffer, associations

@jit(nopython=True)
def inquire_semantics(u, v, resize_factor, padded_segmentation, normalized_likelihoods, likelihoods, radius=3):
    """
    Arguments:
        u (int): The u coordinate of the pixel in the original image.
        v (int): The v coordinate of the pixel in the original image.
        resize_factor (float): The factor to resize the segmentation image.
        padded_segmentation (np.array): The segmentation image padded by the size of radius with -1.
        likelihoods (np.array): A 1D array of shape (N,) where N is the number of pixels with center of (u, v) and given radius. 
            The index is the sequence number of the pixel and the value is the precomputed likelihood.
        radius (int): The radius of the circle in segmentation image size. 

    Returns:
        normalized_likelihoods (np.array): A 1D array of shape (N,) where N is the number of semantic classes. The index is the semantic 
            id and the value is the normalized likelihood.
    """
    u, v = int(u * resize_factor), int(v * resize_factor)

    # Extract semantic ids and corresponding likelihoods
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            semantic_id = padded_segmentation[v + j + radius, u + i + radius]
            if semantic_id != -1:
                likelihood_index = (i + radius) * (2 * radius + 1) + (j + radius)
                normalized_likelihoods[int(semantic_id)] += likelihoods[int(likelihood_index)]

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
def add_color_to_points(associations, colors, random_colors, segmentation, image_height, image_width):
    """
    Arguments:
        associations (np.array): A 2D array of shape (N, 2) where N is the number of points.
        colors (np.array): A 2D array of shape (N, 3) where N is the number of points.
        random_colors (list): A list of random colors.
        segmentation (np.array): The segmentation image.
        cameras (dict): The camera parameters.
    """
    resize_segmentation = 1000
    radius = 2
    decaying = 2
    likelihoods = compute_gaussian_likelihood(radius, decaying)

    if image_height > resize_segmentation or image_width > resize_segmentation:
        resize_factor = resize_segmentation / max(image_height, image_width)
        resize_height, resize_width = int(image_height * resize_factor), int(image_width * resize_factor)
    else:
        resize_height, resize_width = image_height, image_width
        resize_factor = 1

    # pad segmentation image by the size of radius with -1
    padded_segmentation = -np.ones((2*radius+resize_height+2, 2*radius+resize_width+2))
    padded_segmentation[radius+1:radius+resize_height+1, radius+1:radius+resize_width+1] = segmentation
    
    # pad segmentation image by the size of radius with -1
    padded_segmentation = -np.ones((2*radius+resize_height+2, 2*radius+resize_width+2))
    padded_segmentation[radius+1:radius+resize_height+1, radius+1:radius+resize_width+1] = segmentation

    normalized_likelihoods = np.zeros(int(segmentation.max() + 1), dtype=np.float64)


    for i in range(associations.shape[0]):
        u, v = associations[i]
        if u != -1 and v != -1:
            normalized_likelihoods = inquire_semantics(u, v, resize_factor, padded_segmentation, normalized_likelihoods, likelihoods, radius=2)

            if normalized_likelihoods.max() > 0:
                semantic_id = np.argmax(normalized_likelihoods)
                colors[i] = random_colors[int(semantic_id) - 1]

    return colors


class ProbabilisticProjection(object):
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
        image, z_buffer, associations = update_image_and_associations(image, z_buffer, points_projected, self.colors, image_height, image_width)

        return image, associations


if __name__ == "__main__":
    from ssfm.image_segmentation import *
    import time
    
    segmented = True
    if not segmented:
        image_segmentor = ImageSegmentation(sam_params)        
        image_path = '../../data/mission_2/DJI_0247.JPG'
        masks = image_segmentor.predict(image_path)
        image_segmentor.save_npy(masks, '../../data/mission_2/DJI_0247.npy')
    else:
        probabilistic_projector = ProbabilisticProjection()
        probabilistic_projector.read_pointcloud('../../data/model.las')

        probabilistic_projector.read_camera_parameters('../../data/camera.json', '../../data/shots.geojson')

        probabilistic_projector.read_segmentation('../../data/mission_2/DJI_0247.npy')

        image, associations = probabilistic_projector.project('DJI_0247.JPG')

        # generate colors to semantic ids
        random_colors = []
        segmentation = probabilistic_projector.segmentation
        N = int(np.ceil(segmentation.max()))
        for i in range(N):
            # generate a random color   
            color = np.random.randint(0, 255, (3,)).tolist()
            random_colors.append(color)
        random_colors = np.array(random_colors)
        

        # add color to points
        t1 = time.time()
        colors = add_color_to_points(associations, probabilistic_projector.colors, random_colors, segmentation, probabilistic_projector.image_height, probabilistic_projector.image_width)
        t2 = time.time()
        print('Time for adding colors: ', t2 - t1)

        write_las(probabilistic_projector.points, colors, "test.las")




