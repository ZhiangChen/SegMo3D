# read tif file 
from ssfm.files import *
import numpy as np
from numba import jit

import matplotlib.pyplot as plt
import trimesh
import os


@jit(nopython=True)
def calculate_depth_image(z_buffer, points, height, width):
    pixel2point = np.full((height, width), -1, dtype=np.int32)

    N_points = points.shape[0]

    for i in range(N_points):
        x, y, z = points[i]
        px, py = int(x), int(y)

        if 0 <= px < width and 0 <= py < height:
            if z < z_buffer[py, px]:
                z_buffer[py, px] = z
                pixel2point[py, px] = i

    return z_buffer, pixel2point


@jit(nopython=True)
def edge_function(a, b, c):
    return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])

@jit(nopython=True)
def rasterize_depth_image(faces, points_projected, image_height, image_width):
    depth_image = np.full((image_height, image_width), np.inf)

    for face in faces:
        v0, v1, v2 = points_projected[face[0]], points_projected[face[1]], points_projected[face[2]]

        min_x = max(0, int(np.floor(min([v0[0], v1[0], v2[0]]))))
        max_x = min(image_width-1, int(np.ceil(max([v0[0], v1[0], v2[0]]))))
        min_y = max(0, int(np.floor(min([v0[1], v1[1], v2[1]]))))
        max_y = min(image_height-1, int(np.ceil(max([v0[1], v1[1], v2[1]]))))

        area = edge_function(v0, v1, v2)
        if area == 0:
            continue

        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                p = np.array([x + 0.5, y + 0.5])
                w0 = edge_function(v1, v2, p)
                w1 = edge_function(v2, v0, p)
                w2 = edge_function(v0, v1, p)

                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    w0 /= area
                    w1 /= area
                    w2 /= area

                    depth = w0 * v0[2] + w1 * v1[2] + w2 * v2[2]
                    if depth < depth_image[y, x]:
                        depth_image[y, x] = depth

    return depth_image

class DepthImageRendering(object):
    def __init__(self) -> None:
        pass

    def read_mesh(self, mesh_path):
        # assert mesh_path exists
        assert os.path.exists(mesh_path), f"Mesh path {mesh_path} does not exist."
        # read mesh
        self.tri_mesh = trimesh.load(mesh_path)
        self.vertices = self.tri_mesh.vertices
        self.faces = self.tri_mesh.faces

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


    def render_depth_image(self, frame_key):
        """
        Arguments:
            frame_key (int): Key of the frame to render the depth image for. e.g., "DJI_0123.JPG"
        """
        # get camera parameters
        self.extrinsic_matrix = self.cameras[frame_key]

        points_homogeneous = np.hstack((self.vertices, np.ones((len(self.vertices), 1))))

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
        z_buffer = np.full((self.image_height, self.image_width), np.inf)

        # Update image and get associations
        z_buffer, pixel2point = calculate_depth_image(z_buffer, points_projected, self.image_height, self.image_width)

        # get vertices that are associated with pixels on the image
        ids_valid_vertices = np.unique(pixel2point[pixel2point != -1])

        # Check if each vertex in each face is in ids
        overlap_mask = np.isin(self.faces, ids_valid_vertices)

        # Determine if each face has any vertex that overlaps with ids
        mask_valid_faces = np.any(overlap_mask, axis=1)

        depth_image_mesh = rasterize_depth_image(self.faces[mask_valid_faces], points_projected, self.image_height, self.image_width)

        return depth_image_mesh

