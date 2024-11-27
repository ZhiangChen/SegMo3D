#!/usr/bin/env python3

# Uasge: Copy this script to the directory: ScanNet/SensReader/python, and run the following command:
# python3 raw_scene_extractor.py

import os
import subprocess
import random

# Define the base directory for the scans_test
base_dir = "/home/zchen256/semantic_SfM/data/scannet/scans"
random_extraction = False


scene_folders = os.listdir(base_dir)
files = []

for scene_folder in scene_folders:
    scene_folder_path = os.path.join(base_dir, scene_folder)
    scene_file = scene_folder + ".sens"
    if scene_file in os.listdir(scene_folder_path):
        files.append(os.path.join(scene_folder_path, scene_file))

if random_extraction:
    random.shuffle(files)
    # Extract 10% of the files
    files = files[:int(0.1 * len(files))]

for file in files:
    if file.endswith(".sens"):
        # Get the root directory
        root = os.path.dirname(file)
        # Construct the output path
        output_path = os.path.join(root, "output")
        # Create the output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        # Construct the command
        command = [
            "python3", "reader.py",
            "--filename", file,
            "--output_path", output_path,
            "--export_depth_images",
            "--export_color_images",
            "--export_poses",
            "--export_intrinsics"
        ]
        # Run the command
        print(command)
        subprocess.run(command)
