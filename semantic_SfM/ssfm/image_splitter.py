import os
import cv2
import yaml
from tqdm import tqdm
from joblib import Parallel, delayed

class ImageSplitter(object):
    def __init__(self, image_folder_path) -> None:
        self.image_folder_path = image_folder_path
        # assert if the folder path exists
        assert os.path.exists(image_folder_path), f"Folder path {image_folder_path} does not exist"

        self.image_files = [os.path.join(image_folder_path, f) for f in os.listdir(image_folder_path) if f.endswith('.JPG')]

        # print the number of images found in the path
        print(f"Found {len(self.image_files)} images in the folder {image_folder_path}")


    def process_image(self, image_f):
        """
        Split the image into N x N patches with overlap
        """
        image = cv2.imread(image_f)
        image_name = os.path.basename(image_f)
        image_name = image_name.split('.')[0]

        # get the height and width of the image
        height, width = image.shape[:2]

        # get the patch height and width
        patch_height = int((height - self.overlap) / self.N + self.overlap)
        patch_width = int((width - self.overlap) / self.N + self.overlap)

        # split the image into patches
        for i in range(self.N):
            for j in range(self.N):
                # get the patch, considering overlap
                patch = image[i*patch_height - i*self.overlap:(i+1)*patch_height - i*self.overlap, j*patch_width - j*self.overlap:(j+1)*patch_width - j*self.overlap]

                # write the patch to the folder
                patch_name = f"{image_name}_{i}_{j}.JPG"
                patch_path = os.path.join(self.image_write_folder_path, patch_name)
                cv2.imwrite(patch_path, patch)


    def split(self, N, overlap, image_write_folder_path, num_workers=8):
        """
        Args:
            N (int): Number of patches in each row and column
            overlap (int): Overlap between patches (in pixels)
            image_write_folder_path (str): Folder path to write the image patches
        """
        if not os.path.exists(image_write_folder_path):
            os.makedirs(image_write_folder_path)

        self.N = N
        self.overlap = overlap
        self.image_write_folder_path = image_write_folder_path

        self.data = {}

        self.data['N'] = N
        self.data['overlap'] = overlap


        with tqdm(total=len(self.image_files), desc='Processing Images') as progress_bar:
            Parallel(n_jobs=num_workers)(
                delayed(self.process_image)(image_f)
                for image_f in self.image_files
            )
            progress_bar.update(len(self.image_files))
            
        self.yaml_file = os.path.join(image_write_folder_path, 'image_patches.yaml')
        with open(self.yaml_file, 'w') as f:
            yaml.dump(self.data, f)
        print(f"Image patches are written to {image_write_folder_path}")

if __name__ == "__main__":
    image_folder_path = "../../data/box_canyon_park/DJI_photos"
    image_write_folder_path = "../../data/box_canyon_park/DJI_photos_split"

    N = 2
    overlap = 0 

    image_splitter = ImageSplitter(image_folder_path)
    image_splitter.split(N, overlap, image_write_folder_path, num_workers=8)

