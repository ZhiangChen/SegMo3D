from ssfm.files import *
from ssfm.image_segmentation import *
from ssfm.probabilistic_projection import *
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor


class BatchPreparation(object):
    def __init__(self, segmentation_folder, association_folder, save_folder) -> None:
        # assert the exist of the segmentation folder and association folder
        assert os.path.exists(segmentation_folder), "The segmentation folder does not exist."
        assert os.path.exists(association_folder), "The association folder does not exist."
        self.segmentation_folder = segmentation_folder
        self.association_folder = association_folder

        # get the segmentation file paths and association file paths
        self.segmentation_file_paths = [os.path.join(self.segmentation_folder, f) for f in os.listdir(self.segmentation_folder) if f.endswith('.npy')]
        self.association_file_paths = [os.path.join(self.association_folder, f) for f in os.listdir(self.association_folder) if f.endswith('.npy')]

        self.segmentation_file_paths.sort()  
        self.association_file_paths.sort()

        # Check if the number of segmentation files and association files are the same
        assert len(self.segmentation_file_paths) == len(self.association_file_paths), 'The number of segmentation files and association files are not the same.'

        # create segmentation-association pairs
        self.segmentation_association_pairs = []
        for i in range(len(self.segmentation_file_paths)):
            self.segmentation_association_pairs.append((self.segmentation_file_paths[i], self.association_file_paths[i]))

        # print the number of segmentation-association pairs
        print('Number of segmentation-association pairs: {}'.format(len(self.segmentation_association_pairs)))

        self.save_folder = save_folder
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
    """
    def start_batch_preparation(self, num_workers=8):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
        futures = [executor.submit(self.create_associations, segmentation_file_path, association_file_path) for (segmentation_file_path, association_file_path) in self.segmentation_association_pairs]

        try:
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                # Process results or handle exceptions if any
                print(f'Task {i+1}/{len(self.segmentation_association_pairs)} completed')

        except KeyboardInterrupt:
            print("Interrupted by user, cancelling tasks...")
            for future in futures:
                future.cancel()
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            executor.shutdown(wait=False)
            print("Executor shut down.")
        pass
    """

    def start_batch_preparation(self, num_workers=8):
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.create_associations, segmentation_file_path, association_file_path) 
                       for (segmentation_file_path, association_file_path) in self.segmentation_association_pairs]

            try:
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    # Process results or handle exceptions if any
                    print(f'Task {i+1}/{len(self.segmentation_association_pairs)} completed')

            except KeyboardInterrupt:
                print("Interrupted by user, cancelling tasks...")
                # futures cancellation if needed
            except Exception as e:
                print(f"An error occurred: {e}")
            finally:
                print("Executor shut down.")


    def create_associations(self, segmentation_file_path, association_file_path):
        print("processing {}".format(segmentation_file_path))
        # load the segmentation and association
        segmented_objects_image1 = np.load(segmentation_file_path).transpose(1, 0).astype(np.int16)  # transpose from (height, width) to (width, height)
        associations1 = np.load(association_file_path)  # association1 is 2D array for point-pixel association, A 2D array of shape (N, 3) where N is the number of valid points that are projected onto the image. Each row is (u, v, point_index). 
        print("loaded {}".format(segmentation_file_path)) 

        pixel_objects_image1 = associations1[:, :2]  # 2D array of shape (N, 2), where each row is a pixel coordinate (u, v)
        associations1_pixel2point = {tuple(association[:2]): association[2] for association in associations1}
        associations1_point2pixel = {association[2]: tuple(association[:2]) for association in associations1}
        #print("created associations1_pixel2point and associations1_point2pixel for {}".format(segmentation_file_path))

        pixels_array = np.array(list(pixel_objects_image1))
        rows, cols = pixels_array[:, 0], pixels_array[:, 1]
        mask1 = np.zeros(segmented_objects_image1.shape, dtype=bool)
        mask1[rows, cols] = True
        #print("created mask1 for {}".format(segmentation_file_path))

        # create a folder for the current segmentation-association pair
        folder_name = os.path.splitext(os.path.basename(segmentation_file_path))[0]
        folder_path = os.path.join(self.save_folder, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # save mask1
        mask1_path = os.path.join(folder_path, 'mask1.npy')
        np.save(mask1_path, mask1)

        # save associations1_pixel2point, associations1_point2pixel
        associations1_pixel2point_path = os.path.join(folder_path, 'associations1_pixel2point.npy')
        np.save(associations1_pixel2point_path, associations1_pixel2point)
        associations1_point2pixel_path = os.path.join(folder_path, 'associations1_point2pixel.npy')
        np.save(associations1_point2pixel_path, associations1_point2pixel)

        N_objects = int(segmented_objects_image1.max() + 1)
        for j in range(N_objects):
            pixel_object1_image1 = np.argwhere(segmented_objects_image1 == j)  # 2D array of shape (N_pixels, 2), where each row is a pixel coordinate (u, v)
            point_object1_image1 = [associations1_pixel2point[tuple((p[0], p[1]))] for p in pixel_object1_image1 if mask1[p[0], p[1]]]

            # save point_object1_image1 with the name of j_point_object1_image1.npy
            point_object1_image1_path = os.path.join(folder_path, '{}_point_object1_image1.npy'.format(j))
            np.save(point_object1_image1_path, point_object1_image1)




if __name__ == '__main__':
    segmentation_folder = '../../data/mission_2_segmentations'
    association_folder = '../../data/mission_2_associations'
    save_folder = '../../data/mission_2_temp_associations'
    batch_preparation = BatchPreparation(segmentation_folder, association_folder, save_folder)
    batch_preparation.start_batch_preparation(num_workers=16)
