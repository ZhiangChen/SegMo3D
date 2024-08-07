import cv2
import os
import yaml
from tqdm import tqdm
from joblib import Parallel, delayed
import queue

# Function to calculate the Laplacian variance of an image
def calculate_laplacian_var(photo_folder_path, f):
    img = cv2.imread(os.path.join(photo_folder_path, f))
    if img is None:
        return f, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return f, laplacian_var


def select_scannet_keyimages(scene_dir, ratio=0.2, threshold=180, file_cluster_size=50, file_select_window=50, n_jobs=8):
    """
    Select keyimages from the photos in the scene directory based on the laplacian variance of the images.
    
    Args:
        scene_dir (str): The path to the scene directory.
        ratio (float): The ratio of keyimages to select.
        threshold (int): The threshold for the laplacian variance. The greater the threshold, the clearer the image.
        file_cluster_size (int): The maximum number of images in each cluster.
        file_select_window (int): A file with the highest laplacian variance in each window of size file_select_window will be selected.
        n_jobs (int): The number of jobs to run in parallel.
    """
    assert os.path.exists(scene_dir), 'Scene directory does not exist!'

    photo_folder_path = os.path.join(scene_dir, 'photos')
    # get all photos in photo_folder_path
    photos = [f for f in os.listdir(photo_folder_path) if f.endswith('.jpg')]
    # sort photos based the values of the numbers in the filenames
    photos = sorted(photos, key=lambda x: int(x.split('_')[-1].split('.')[0]))


    # Calculate Laplacian variances in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(calculate_laplacian_var)(photo_folder_path, f) for f in tqdm(photos, desc="Processing images")
    )

    # Convert the results to a dictionary, filtering out any None values
    laplacian_vars = {f: var for f, var in results}

    # keep only the photos with laplacian_var > threshold
    keyimages_threshold = [f for f in photos if laplacian_vars[f] > threshold]

    # sort the photos based on the laplacian_vars
    sorted_photos = sorted(photos, key=lambda x: laplacian_vars[x], reverse=True)

    # select the keyimages
    keyimages_ratio = sorted_photos[:int(ratio * len(sorted_photos))]

    # keyimages is the union of keyimages_threshold and keyimages_ratio
    keyimages = list(set(keyimages_threshold).union(set(keyimages_ratio)))

    # sort keyimages based the values of the numbers in the filenames
    keyimages = sorted(keyimages, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    keyimages_laplacian_vars = [laplacian_vars[f] for f in keyimages]

    # cluster keyimages based on the continuous numbers in the filenames
    clusters = []
    cluster = [keyimages[0]]
    for i in range(1, len(keyimages)):
        prev = int(keyimages[i-1].split('_')[-1].split('.')[0])
        curr = int(keyimages[i].split('_')[-1].split('.')[0])
        if curr - prev == 1:
            cluster.append(keyimages[i])
        else:
            clusters.append(cluster)
            cluster = [keyimages[i]]
    clusters.append(cluster)

    # select the keyimage in each cluster with the highest laplacian variance
    selected_keyimages = []
    q = queue.Queue()

    # Add initial clusters to the queue
    for cluster in clusters:
        q.put(cluster)

    while not q.empty():
        cluster = q.get()

        if len(cluster) == 1:
            selected_keyimages.append(cluster[0])
        else:
            if len(cluster) < file_cluster_size:
                cluster_laplacian_vars = [keyimages_laplacian_vars[keyimages.index(f)] for f in cluster]
                selected_keyimages.append(cluster[cluster_laplacian_vars.index(max(cluster_laplacian_vars))])
            else:
                if len(cluster) < file_cluster_size*2:
                    split_idx = len(cluster) // 2
                    cluster1 = cluster[:split_idx]
                    cluster2 = cluster[split_idx:]
                    q.put(cluster1)
                    q.put(cluster2)
                else:
                    cluster1 = cluster[:file_cluster_size]
                    cluster2 = cluster[file_cluster_size:]
                    q.put(cluster1)
                    q.put(cluster2)


    window_keyimages = []
    # sort laplacian_vars based on the values of the numbers in the filenames
    sorted_laplacian_vars = [laplacian_vars[f] for f in photos]
    # select keyimages in each window of size file_select_window with the highest laplacian variance
    for i in range(0, len(sorted_laplacian_vars), file_select_window):
        window = sorted_laplacian_vars[i:i+file_select_window]
        idx = window.index(max(window))
        window_keyimages.append(photos[i+idx])
    
    # add the keyimages in each window to the selected_keyimages
    selected_keyimages = list(set(selected_keyimages).union(set(window_keyimages)))

    # print the number of keyimages and total images
    print('Number of keyimages_threshold: ', len(keyimages_threshold))
    print('Number of keyimages_ratio: ', len(keyimages_ratio))
    print('Number of keyimages: ', len(keyimages))
    print('Number of selected keyimages from window: ', len(window_keyimages))
    print('Number of selected keyimages: ', len(selected_keyimages))
    print('Total images: ', len(photos))

    # change .jpg to .npy
    selected_keyimages = [f.replace('.jpg', '.npy') for f in selected_keyimages]

    # save the keyimages to a yaml file
    save_path = os.path.join(scene_dir, 'associations', 'keyimages.yaml')
    with open(save_path, 'w') as f:
        yaml.dump(selected_keyimages, f)

    print('Keyimages saved to: ', save_path)


if __name__ == '__main__':
    scene_dir = '../../data/scene0000_00'
    select_scannet_keyimages(scene_dir)