import numpy as np
import os

class ClassAssignment:
    def __init__(self, semantics_path, pixel2point_folder, class_folder):
        # assert exists 
        assert os.path.exists(semantics_path), "Semantics path does not exist"
        assert os.path.exists(pixel2point_folder), "Pixel2point folder does not exist"
        assert os.path.exists(class_folder), "Class folder does not exist"

        # load semantics
        self.semantics = np.load(semantics_path)

        # load pixel2point
        self.pixel2point_files = [f for f in os.listdir(pixel2point_folder) if f.endswith('.npy')]
        self.pixel2point_folder = pixel2point_folder

        # load class
        self.class_files = [f for f in os.listdir(class_folder) if f.endswith('.npy')]
        self.class_folder = class_folder

        # check if class file list is a subset of pixel2point file list
        assert set(self.class_files).issubset(set(self.pixel2point_files)), "Class files are not a subset of pixel2point files"

    
    def select_instances(self, output_path):
        assert output_path.endswith('.npy'), "Output path must be a .npy file"
        selected_instances = []
        for class_file in self.class_files:
            print(f"Processing {class_file}")
            # load class file
            class_path = os.path.join(self.class_folder, class_file)
            class_data = np.load(class_path)

            N_objects = int(class_data.max() + 1)
            print(f"Number of objects: {N_objects}")

            # load pixel2point file
            pixel2point_path = os.path.join(self.pixel2point_folder, class_file)
            pixel2point_data = np.load(pixel2point_path)

            for object_id in range(N_objects):
                pixels = class_data == object_id
                points = pixel2point_data[pixels]
                points = points[points != -1]
                if len(points) == 0:
                    continue
                instance_ids = self.semantics[points]
                instance_id = np.argmax(np.bincount(instance_ids))
                selected_instances.append(instance_id)

        selected_instances = np.unique(selected_instances)

        print(f"Selected instances: {selected_instances}")

        # save new semantics: remove all semantics that are not in selected_instances
        new_semantics = np.zeros_like(self.semantics) - 1
        for i, instance_id in enumerate(selected_instances):
            new_semantics[self.semantics == instance_id] = i

        np.save(output_path, new_semantics)



    def assign_classes(self, output_path):
        assert output_path.endswith('.ymal'), "Output path must be a .yaml file"
        # create a dictionary to store class assignments
        instance_class = {}
        for class_file in self.class_files:
            print(f"Processing {class_file}")
            # load class file
            class_path = os.path.join(self.class_folder, class_file)
            class_data = np.load(class_path)

            N_objects = int(class_data.max() + 1)
            print(f"Number of objects: {N_objects}")

            # load pixel2point file
            pixel2point_path = os.path.join(self.pixel2point_folder, class_file)
            pixel2point_data = np.load(pixel2point_path)

            for object_id in range(N_objects):
                pixels = class_data == object_id
                points = pixel2point_data[pixels]
                points = points[points != -1]
                if len(points) == 0:
                    continue
                instance_ids = self.semantics[points]
                instance_id = np.argmax(np.bincount(instance_ids))
                if instance_id not in instance_class:
                    instance_class[instance_id] = []
                instance_class[instance_id].append(object_id)

        # save instance_class as a yaml file 
        with open(output_path, 'w') as f:
            yaml.dump(instance_class, f)


                
                
        



if __name__ == "__main__":
    semantics_path = "../../data/granite_dells/associations/semantics/semantics_315.npy"
    pixel2point_folder = "../../data/granite_dells/associations/pixel2point"
    class_folder = "../../data/granite_dells/segmentations_classes"
    output_path = "../../data/granite_dells/associations/semantics/semantics_315_class.npy"
    pointcloud_path = '../../data/granite_dells/SfM_products/granite_dells_downsampled.las'
    save_las_path = "../../data/granite_dells/associations/semantics/semantics_315_class.las"

    ca = ClassAssignment(semantics_path, pixel2point_folder, class_folder, output_path)

    ca.select_instances()

    from ssfm.post_processing import *

    add_semantics_to_pointcloud(pointcloud_path, output_path, save_las_path)


