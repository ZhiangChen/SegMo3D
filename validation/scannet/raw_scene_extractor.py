#!/usr/bin/env python3

# Uasge: Copy this script to the directory: ScanNet/SensReader/python, and run the following command:
# python3 raw_scene_extractor.py

import os
import subprocess

# Define the base directory for the scans_test
base_dir = "/home/zchen256/semantic_SfM/data/scannet/scans_test"

# Iterate over all files in the directory
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".sens"):
            # Construct the full path to the .sens file
            sens_file = os.path.join(root, file)
            # Construct the output path
            output_path = os.path.join(root, "output")
            # Create the output directory if it doesn't exist
            os.makedirs(output_path, exist_ok=True)
            # Construct the command
            command = [
                "python3", "reader.py",
                "--filename", sens_file,
                "--output_path", output_path,
                "--export_depth_images",
                "--export_color_images",
                "--export_poses",
                "--export_intrinsics"
            ]
            # Run the command
            print(command)
            subprocess.run(command)
            