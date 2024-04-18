import os
import cv2
import yaml
from tqdm import tqdm

class ImageSplitter(object):
    def __init__(self, image_folder_path) -> None:
        self.image_folder_path = image_folder_path
        # assert if the folder path exists
        assert os.path.exists(image_folder_path), f"Folder path {image_folder_path} does not exist"

        self.image_files = [os.path.join(image_folder_path, f) for f in os.listdir(image_folder_path) if f.endswith('.JPG')]

        # print the number of images found in the path
        print(f"Found {len(self.image_files)} images in the folder {image_folder_path}")

        # create a yaml file to store the image paths and the corresponding patches
        self.yaml_file = os.path.join(image_folder_path, "image_patches.yaml")
        if os.path.exists(self.yaml_file):
            # read the yaml file
            with open(self.yaml_file, 'r') as f:
                self.data = yaml.load(f, Loader=yaml.FullLoader)
            
            # print success message
            print(f"Read the yaml file {self.yaml_file}")

        else:
            self.data = {}


    def split(self, N, overlap, image_write_folder_path):
        """
        Split the image into N x N patches with overlap

        Args:
            N (int): Number of patches in each row and column
            overlap (int): Overlap between patches (in pixels)
            image_write_folder_path (str): Folder path to write the image patches
        """
        if not os.path.exists(image_write_folder_path):
            os.makedirs(image_write_folder_path)

        # put N and overlap in the yaml file
        self.data['N'] = N
        self.data['overlap'] = overlap

        for image_f in tqdm(self.image_files, desc='Processing Images'):
            image = cv2.imread(image_f)
            image_name = os.path.basename(image_f)
            image_name = image_name.split('.')[0]

            # get the height and width of the image
            height, width = image.shape[:2]

            # get the patch height and width
            patch_height = int((width - overlap) / N + overlap)
            patch_width = int((height - overlap) / N + overlap)

            
            if not os.path.exists(image_write_folder_path):
                os.makedirs(image_write_folder_path)


            # split the image into patches
            for i in range(N):
                for j in range(N):
                    # get the patch, considering overlap
                    patch = image[i*patch_height - i*overlap:(i+1)*patch_height - i*overlap, j*patch_width - j*overlap:(j+1)*patch_width - j*overlap]

                    # write the patch to the folder
                    patch_name = f"{image_name}_{i}_{j}.png"
                    patch_path = os.path.join(image_write_folder_path, patch_name)
                    cv2.imwrite(patch_path, patch)

                    # store the patch path in the yaml file
                    if image_name not in self.data:
                        self.data[image_name] = {}
                    self.data[image_name][f"{i}_{j}"] = patch_path

        # write the yaml file
        with open(self.yaml_file, 'w') as f:
            yaml.dump(self.data, f)

        print(f"Image patches are written to {image_write_folder_path}")

if __name__ == "__main__":
    image_folder_path = "../../data/box_canyon_park/DJI_photos"
    image_write_folder_path = "../../data/box_canyon_park/DJI_photos_split"

    N = 2
    overlap = 0 

    image_splitter = ImageSplitter(image_folder_path)
    image_splitter.split(N, overlap, image_write_folder_path)

