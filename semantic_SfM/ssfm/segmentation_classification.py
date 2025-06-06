import os
import numpy as np
import shutil
import cv2
import tqdm
import natsort 

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import pandas as pd
import matplotlib.pyplot as plt

import random
from sklearn.model_selection import train_test_split
from collections import Counter

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets.folder import default_loader
from torchvision import models
from sklearn.utils import resample
from datetime import datetime

from ssfm.files import *

# set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def extract_segmentation(image_folder_path, segmentation_folder_path, save_folder_path, keep_option, keep_image_N=1, random_image=False, keep_segmentation_N=1, random_segmentation=False, camera_parameter_file=None):
    """
    Extracts the segmentation masks from the segmentation files and saves the cropped images
    :param image_folder_path: path to the folder containing the images
    :param segmentation_folder_path: path to the folder containing the segmentation files
    :param save_folder_path: path to the folder where the cropped images will be saved
    :param keep_option: 'foreground' - only keep the image where the mask is present
                        'all' - keep the entire image
                        'all+mask' - keep the entire image and save the mask as well
    :param keep_image_N: number of images to keep
    :param keep_segmentation_N: number of segmentations per image to keep
    """
    # check if the image folder exists
    assert os.path.exists(image_folder_path), "Image folder does not exist"
    image_files = [f for f in os.listdir(image_folder_path) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg') or f.endswith('.JPG')]
    image_extension = image_files[0].split('.')[-1]
    print("Number of image files: ", len(image_files))

    # check if the segmentation folder exists
    assert os.path.exists(segmentation_folder_path), "Segmentation folder does not exist"
    segmentation_files = [f for f in os.listdir(segmentation_folder_path) if f.endswith('.npy')]
    if random_image:
        np.random.shuffle(segmentation_files)
    else:
        segmentation_files = natsort.natsorted(segmentation_files)
    print("Number of segmentation files: ", len(segmentation_files))

    # check if keep_image_N is greater than the number of segmentation files
    if keep_image_N > len(segmentation_files):
        keep_image_N = len(segmentation_files)
    
    # create the save folder if it does not exist
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    if camera_parameter_file is not None:
        cameras = read_camera_parameters_agisoft(camera_parameters_file)
        distortion_params = cameras['distortion_params']
        matrix_intrinsics = cameras['K']
        print("Camera parameters loaded")

    i_image = 0
    # iterate over the segmentation files
    for seg_file in tqdm.tqdm(segmentation_files):
        masks = np.load(os.path.join(segmentation_folder_path, seg_file))
        image_file = seg_file.replace('.npy', '.' + image_extension)
        # assert if the image file exists for the segmentation file and output the segmentation file
        assert image_file in image_files, "Image file does not exist for {}".format(seg_file)
        image = cv2.imread(os.path.join(image_folder_path, image_file))
        assert image is not None, "Image file could not be read"

        if camera_parameter_file is not None:
            image = cv2.undistort(image, matrix_intrinsics, distortion_params)

        # masks is a 2D numpy array where each pixel value is the mask id
        # get the unique mask ids
        mask_ids = np.unique(masks)

        if random_segmentation:
            # shuffle the mask ids
            np.random.shuffle(mask_ids)
        else:
            mask_ids = natsort.natsorted(mask_ids)
            
        i_segmentation = 0

        assert keep_option in ['foreground', 'all', 'all+mask'], "Invalid keep option"

        # iterate over the masks
        for i in mask_ids:
            if i == -1:
                continue
            mask = np.zeros_like(masks)
            mask = (masks == i).astype(np.uint8)
            if keep_option == 'foreground':
                # only keep the image where the mask is present
                image_tmp = image.copy()
                image_tmp[mask == 0] = 0
                # get bounding box coordinates for the mask to crop the image
                x, y, w, h = cv2.boundingRect(mask)
                # crop the image
                cropped_image = image_tmp[y:y+h, x:x+w]
                # save the cropped image
                save_file = os.path.join(save_folder_path, seg_file.replace('.npy', '_{}.jpg'.format(i)))
                cv2.imwrite(save_file, cropped_image)
            else:
                # get bounding box coordinates for the mask to crop the image
                x, y, w, h = cv2.boundingRect(mask)
                # crop the image
                cropped_image = image[y:y+h, x:x+w]
                # save the cropped image
                save_file = os.path.join(save_folder_path, seg_file.replace('.npy', '_{}.jpg'.format(i)))
                cv2.imwrite(save_file, cropped_image)

                if keep_option == 'all+mask':
                    # save the mask as well
                    save_mask_file = os.path.join(save_folder_path, seg_file.replace('.npy', '_{}_mask.jpg'.format(i)))
                    cropped_mask = mask[y:y+h, x:x+w] * 255
                    cv2.imwrite(save_mask_file, cropped_mask)

            # enlarged cropped image
            padding = 50
            y1 = max(0, y - padding)
            y2 = min(image.shape[0], y + h + padding)
            x1 = max(0, x - padding)
            x2 = min(image.shape[1], x + w + padding)
            enlarged_cropped_image = image[y1:y2, x1:x2]
            # save the enlarged cropped image
            save_enlarged_file = os.path.join(save_folder_path, seg_file.replace('.npy', '_{}_enlarged.jpg'.format(i)))
            cv2.imwrite(save_enlarged_file, enlarged_cropped_image)

            i_segmentation += 1
            if i_segmentation >= keep_segmentation_N:
                break

        i_image += 1
        if i_image >= keep_image_N:
            break


# Label mappings
LABELS = {
    'q': 'rock',
    'w': 'no_rock',
    'e': 'NA'
}

class ImageAnnotator:
    def __init__(self, root, default_folder_path="~/semantic_SfM/data/"):
        self.root = root
        self.root.title("Simple Image Annotator")

        # UI Elements
        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.status_label = tk.Label(root, text="", font=("Arial", 12))
        self.status_label.pack(pady=5)

        self.instruction_label = tk.Label(
            root,
            text="[q] rock    [w] no_rock    [e] NA    [s] skip    [z] undo",
            font=("Courier", 12),
            fg="blue"
        )
        self.instruction_label.pack(pady=5)

        # Folder selection
        default_folder = os.path.expanduser(default_folder_path)
        self.folder_path = filedialog.askdirectory(title="Select Folder with Images", initialdir=default_folder)
        if not self.folder_path:
            print("No folder selected. Exiting.")
            root.quit()


        self.image_files = [
            f for f in os.listdir(self.folder_path)
            if f.lower().endswith('.jpg') and "mask" not in f.lower() and "enlarged" not in f.lower()
        ]
        self.image_files = natsort.natsorted(self.image_files)

        self.csv_path = os.path.join(self.folder_path, "annotations.csv")
        self.annotations = self.load_or_create_csv()


        self.current_index = self.find_next_unlabeled_index()
        self.previous_index = None  # for undo

        self.root.bind('<Key>', self.key_pressed)
        self.display_image()

    def load_or_create_csv(self):
        if os.path.exists(self.csv_path):
            csv_df = pd.read_csv(self.csv_path)

            # build a new DataFrame with the image files and labels; if label is empty, set it to ""; if label is in the csv, set it to the label in the csv
            df = pd.DataFrame({
                'id': range(len(self.image_files)),
                'image_path': [os.path.join(self.folder_path, f) for f in self.image_files],
                'label': [csv_df[csv_df['image_path'] == os.path.join(self.folder_path, f)]['label'].values[0] if os.path.join(self.folder_path, f) in csv_df['image_path'].values else "" for f in self.image_files]
            })
            

        else:
            df = pd.DataFrame({
                'id': range(len(self.image_files)),
                'image_path': [os.path.join(self.folder_path, f) for f in self.image_files],
                'label': ["" for _ in self.image_files]
            })
            df.to_csv(self.csv_path, index=False)
        return df

    def find_next_unlabeled_index(self):
        unlabeled = self.annotations[self.annotations['label'].isnull() | (self.annotations['label'].astype(str).str.strip() == "")]
        return unlabeled.index[0] if not unlabeled.empty else len(self.image_files)

    def display_image(self):
        if self.current_index >= len(self.image_files):
            self.image_label.config(image='', text="✅ All images annotated!")
            self.status_label.config(text="✅ All images annotated!")
            return

        image_path = self.annotations.loc[self.current_index, 'image_path']
        try:
            img = Image.open(image_path)
            mask_path = image_path.replace('.jpg', '_mask.jpg')
            enlarge_path = image_path.replace('.jpg', '_enlarged.jpg')
            if os.path.exists(mask_path):
                mask_img = Image.open(mask_path)
                # concatenate the image and mask horizontally
                new_img = Image.new('RGB', (img.width + mask_img.width, img.height))
                new_img.paste(img, (0, 0))
                new_img.paste(mask_img, (img.width, 0))
                img = new_img
            if os.path.exists(enlarge_path):
                enlarge_img = Image.open(enlarge_path)
                # concatenate the image and enlarged image horizontally
                new_img = Image.new('RGB', (img.width + enlarge_img.width, enlarge_img.height))
                #new_img.paste(img, (0, enlarge_img.height // 2 - img.height // 2))
                #new_img.paste(enlarge_img, (img.width, 0))
                new_img.paste(enlarge_img, (0, 0))
                new_img.paste(img, (enlarge_img.width, enlarge_img.height // 2 - img.height // 2))
                img = new_img
            # Upscale small images if width or height < 380
            min_dim = min(img.size)
            if min_dim < 380:
                scale_factor = 380 / min_dim
                new_width = int(img.width * scale_factor)
                new_height = int(img.height * scale_factor)
                img = img.resize((new_width, new_height), Image.LANCZOS)
            img.thumbnail((800, 600))
            self.tk_img = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.tk_img)
        except Exception as e:
            self.image_label.config(text=f"Error loading image: {image_path}")
            print(e)

        current_file = os.path.basename(image_path)
        self.status_label.config(
            text=f"[{self.current_index + 1}/{len(self.image_files)}] {current_file}"
        )

    def key_pressed(self, event):
        key = event.char.lower()

        if key == 'z':  # Undo
            if self.previous_index is not None:
                self.annotations.at[self.previous_index, 'label'] = ""
                self.current_index = self.previous_index
                self.previous_index = None
                self.annotations.to_csv(self.csv_path, index=False)
                self.display_image()
            return

        if key == 's':  # Skip
            self.previous_index = self.current_index
            self.current_index = self.find_next_unlabeled_index() + 1
            self.display_image()
            return

        if key in LABELS and self.current_index < len(self.image_files):
            label = LABELS[key]
            self.annotations.at[self.current_index, 'label'] = label
            self.annotations.to_csv(self.csv_path, index=False)

            self.previous_index = self.current_index
            self.current_index = self.find_next_unlabeled_index()
            self.display_image()


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        log_probs = torch.nn.functional.log_softmax(pred, dim=-1)
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (pred.size(1) - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


class RockDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.loader = default_loader

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'image_path']
        label = 1 if self.df.loc[idx, 'label'] == 'rock' else 0
        image = self.loader(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label


class RockClassifier:
    def __init__(self, model_name='efficientnet_b7', csv_path='annotations.csv', output_dir='saved_models', image_size=380, batch_size=16, num_epochs=10, lr=1e-4, device='cpu'):
        self.model_name = model_name.lower()
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr

        self.device = torch.device(device)
        self._prepare_data()
        self._prepare_model()

    def _prepare_data(self):
        df = pd.read_csv(self.csv_path)
        df = df[df['label'].isin(['rock', 'no_rock'])]

        df_rock = df[df['label'] == 'rock']
        df_no_rock = df[df['label'] == 'no_rock']
        min_len = min(len(df_rock), len(df_no_rock))

        df_rock_bal = resample(df_rock, replace=False, n_samples=min_len, random_state=42)
        df_no_rock_bal = resample(df_no_rock, replace=False, n_samples=min_len, random_state=42)
        df_balanced = pd.concat([df_rock_bal, df_no_rock_bal]).sample(frac=1, random_state=42).reset_index(drop=True)

        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

        self.dataset = RockDataset(df_balanced, transform=transform)
        train_len = int(0.8 * len(self.dataset))
        val_len = len(self.dataset) - train_len
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_len, val_len])

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        # print train and val numbers
        print(f"Train: {len(self.train_dataset)} samples")
        print(f"Val: {len(self.val_dataset)} samples")

    def _prepare_model(self):
        num_classes = 2

        if self.model_name == 'regnet_y_16gf':
            self.model = models.regnet_y_16gf(weights='IMAGENET1K_V2')
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        elif self.model_name == 'resnext101_32x8d':
            self.model = models.resnext101_32x8d(weights='IMAGENET1K_V1')
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        elif self.model_name == 'inception_v3':
            self.model = models.inception_v3(weights='IMAGENET1K_V1', aux_logits=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
            self.model.AuxLogits.fc = nn.Identity() 
            self.image_size = 299  # override

        elif self.model_name == 'wide_resnet101_2':
            self.model = models.wide_resnet101_2(weights='IMAGENET1K_V1')
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        elif self.model_name == 'resnet':
            self.model = models.resnet50(weights='IMAGENET1K_V1')
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        elif self.model_name == 'convnext_large':
            self.model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1)
            self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, num_classes)

        elif self.model_name == "efficientnet_v2_1":
            weights = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1
            self.model = models.efficientnet_v2_l(weights=weights)
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, num_classes)

        else:
            raise ValueError(f"Model {self.model_name} not supported.")

        self.model = self.model.to(self.device)

    def mixup_data(self, x, y, alpha=1.0):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def train(self, load_pretrained=None, use_mixup=True, mixup_alpha=0.4):
        criterion = LabelSmoothingLoss(smoothing=0.1)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )

        if load_pretrained is not None:
            self.model.load_state_dict(torch.load(load_pretrained))

        best_val_acc = 0
        os.makedirs(self.output_dir, exist_ok=True)

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                if use_mixup:
                    images, targets_a, targets_b, lam = self.mixup_data(images, labels, alpha=mixup_alpha)
                    outputs = self.model(images)
                    if isinstance(outputs, tuple):  # Handle InceptionOutputs
                        outputs = outputs[0]
                    loss = self.mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    outputs = self.model(images)
                    if isinstance(outputs, tuple):  # Handle InceptionOutputs
                        outputs = outputs[0]
                    loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            val_acc, val_loss = self.evaluate()
            print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")

            scheduler.step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_path = os.path.join(self.output_dir, f"{self.model_name}_best.pth")
                torch.save(self.model.state_dict(), model_path)

        print(f"\n✅ Training complete. Best Val Acc: {best_val_acc:.4f}")


    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        loss = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss += criterion(outputs, labels).item()

        return correct / total if total > 0 else 0.0, loss 

    def predict_batch(self, image_paths, batch_size=16, model_path=None):
        """
        Predict the class of multiple images (rock or no_rock) in batches.

        Args:
            image_paths (list): List of paths to images to be predicted.
            batch_size (int, optional): Number of images to process in each batch. Default is 16.

        Returns:
            list: List of predicted classes ("rock" or "no_rock") for each image.
        """
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))

        # Ensure the model is in evaluation mode
        self.model.eval()

        # Define the same transformation applied to the training data
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

        # Load and transform all images
        images = []
        for image_path in image_paths:
            image = Image.open(image_path)
            image = transform(image)
            images.append(image)

        # Stack all images into a single tensor
        images_tensor = torch.stack(images)

        # Move the images tensor to the same device as the model (e.g., GPU or CPU)
        images_tensor = images_tensor.to(self.device)

        # List to store the predictions for each image
        all_predictions = []

        # Process the images in batches
        num_batches = len(images_tensor) // batch_size + (1 if len(images_tensor) % batch_size != 0 else 0)
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(images_tensor))
            batch = images_tensor[start_idx:end_idx]

            # Perform inference on the batch
            with torch.no_grad():  # Disable gradient computation
                outputs = self.model(batch)
                _, predicted_classes = torch.max(outputs, 1)

            # Convert the numeric predictions to the corresponding classes
            class_names = ['no_rock', 'rock']
            batch_predictions = [class_names[idx.item()] for idx in predicted_classes]

            # Append the predictions for this batch
            all_predictions.extend(batch_predictions)

        return all_predictions


    def classify_masks(self, mask_folder_path, image_folder_path, output_folder_path, model_path=None, camera_parameter_file=None, batch_size=16, save_overlap=False, indices=None):
        # Create the output folder if it does not exist
        os.makedirs(output_folder_path, exist_ok=True)

        # Get list of mask files
        mask_files = [f for f in os.listdir(mask_folder_path) if f.endswith('.npy')]
        mask_files = natsort.natsorted(mask_files)
        print(f"Number of mask files: {len(mask_files)}")
        #print(f"Mask files: {mask_files}")

        # Get list of image files
        image_files = [f for f in os.listdir(image_folder_path) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg') or f.endswith('.JPG')]
        image_files = natsort.natsorted(image_files)
        print(f"Number of image files: {len(image_files)}")
        #print(f"Image files: {image_files}")

        if indices is not None:
            mask_files = [mask_files[i] for i in indices]
            image_files = [image_files[i] for i in indices]
            print(f"Number of files after index filtering: {len(mask_files)}")

        # Load the model
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
            print(f"Model loaded from {model_path}")

        # Load camera parameters
        if camera_parameter_file is not None:
            cameras = read_camera_parameters_agisoft(camera_parameter_file)
            distortion_params = cameras['distortion_params']
            matrix_intrinsics = cameras['K']
            print("Camera parameters loaded")

        # Define the same transformation applied to the training data
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

        self.model.eval()

        # Iterate over the mask files
        for i in tqdm.tqdm(range(len(mask_files))):
            mask_file = mask_files[i]
            image_file = image_files[i]
            # check if their basename is the same
            assert os.path.basename(mask_file).split('.')[0] == os.path.basename(image_file).split('.')[0], f"Mask file {mask_file} does not match image file {image_file}"

            masks = np.load(os.path.join(mask_folder_path, mask_file))            
            image = cv2.imread(os.path.join(image_folder_path, image_file))
            # convert image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # assert if the image is not None
            assert image is not None, "Image file could not be read"

            if camera_parameter_file is not None:
                image = cv2.undistort(image, matrix_intrinsics, distortion_params)

            predictions = []
            mask_ids = np.unique(masks)

            valid_mask_ids = []

            batch_images = []
            for mask_id in mask_ids:
                if mask_id < 1:
                    continue
                mask = np.zeros_like(masks)
                mask = (masks == mask_id).astype(np.uint8)

                x, y, w, h = cv2.boundingRect(mask)
                cropped_image = image[y:y+h, x:x+w]

                # Apply the same transformation as the training data
                cropped_image = Image.fromarray(cropped_image)
                cropped_image = transform(cropped_image)
                batch_images.append(cropped_image)
                valid_mask_ids.append(mask_id)

                if len(batch_images) == batch_size:
                    batch_images = torch.stack(batch_images).to(self.device)

                    with torch.no_grad():
                        outputs = self.model(batch_images)
                        _, predicted_classes = torch.max(outputs, 1)

                    class_names = ['no_rock', 'rock']
                    batch_predictions = [class_names[idx.item()] for idx in predicted_classes]


                    predictions.extend(batch_predictions)
                    batch_images = []

            if len(batch_images) > 0:
                batch_images = torch.stack(batch_images).to(self.device)

                with torch.no_grad():
                    outputs = self.model(batch_images)
                    _, predicted_classes = torch.max(outputs, 1)

                class_names = ['no_rock', 'rock']
                batch_predictions = [class_names[idx.item()] for idx in predicted_classes]

                predictions.extend(batch_predictions)
            
            # assert the length of predictions and valid_mask_ids are the same
            assert len(predictions) == len(valid_mask_ids), f"Length of predictions {len(predictions)} and valid_mask_ids {len(valid_mask_ids)} do not match"
            # filter masks with rock predictions
            rock_mask_ids = [mask_id for mask_id, pred in zip(valid_mask_ids, predictions) if pred == 'rock']
            # create a new mask with only the rock predictions
            rock_mask = np.zeros_like(masks) - 1
            for new_id, mask_id in enumerate(rock_mask_ids):
                rock_mask[masks == mask_id] = new_id

            # save the new mask
            save_file = os.path.join(output_folder_path, mask_file)
            np.save(save_file, rock_mask)

            if save_overlap:
                # create a new image with only the rock predictions
                overlap_image = np.zeros_like(image)
                unique_ids = np.unique(rock_mask)
                for mask_id in unique_ids:
                    if mask_id < 1:
                        continue
                    # get a random color for the mask
                    color = np.random.randint(0, 255, size=3).tolist()
                    overlap_image[rock_mask == mask_id] = color
                
                overlap_image = cv2.addWeighted(image, 0.7, overlap_image, 0.3, 0)
                # convert overlap_image to BGR
                overlap_image = cv2.cvtColor(overlap_image, cv2.COLOR_RGB2BGR)
                # save the new image
                save_image_file = os.path.join(output_folder_path, image_file)
                cv2.imwrite(save_image_file, overlap_image)

                




if __name__ == '__main__':
    image_folder_path = "../../data/centennial_bluff/mission_b/DJI_photos"
    segmentation_folder_path = "../../data/centennial_bluff/mission_b/segmentations_filtered"
    save_folder_path = "../../data/centennial_bluff/mission_b/segmentations_extraction"
    camera_parameters_file = "../../data/centennial_bluff/mission_b/SfM_products/b.xml"
    segmentation_filter_folder_path = "../../data/centennial_bluff/mission_b/segmentations_class_filter"
    
    extract_segmentation_option = False
    annotation_option = False
    classification_option = False
    validation_option = False
    prediction_image_option = False
    prediction_mask_option = False
    change_background_to_object_option = True

    model_names = [
        "regnet_y_16gf",
        "resnext101_32x8d",
        "inception_v3",
        "wide_resnet101_2",
        "convnext_large",
        "efficientnet_v2_1"]

    csv_path = os.path.join(save_folder_path, "annotations.csv")
    trained_models_dir = os.path.join(save_folder_path, "trained_models")
    if not os.path.exists(trained_models_dir):
        os.makedirs(trained_models_dir)

    if extract_segmentation_option:
        extract_segmentation(image_folder_path, segmentation_folder_path, save_folder_path, keep_option='all+mask', keep_image_N=100, random_image=True, keep_segmentation_N=10, random_segmentation=True, camera_parameter_file=camera_parameters_file)

    if annotation_option:
        default_folder_path = save_folder_path
        root = tk.Tk()
        app = ImageAnnotator(root, default_folder_path=default_folder_path)
        root.mainloop() 

    if classification_option:

        model_name = 'efficientnet_v2_1'  # Change this to the desired model name
        assert model_name in model_names, f"Model {model_name} not supported."
        print(f"\n🔧 Training model: {model_name}")
        trainer = RockClassifier(
            model_name=model_name,
            csv_path=csv_path,
            output_dir=trained_models_dir,
            num_epochs=40,
            batch_size=16,
            device="cuda:4",
            lr=1e-5
        )

        trainer.train()

    if validation_option:

        model_name = 'resnext101_32x8d'
        predictor = RockClassifier(
            model_name=model_name,
            csv_path=csv_path,
            output_dir=trained_models_dir,
            num_epochs=80,
            batch_size=16,
            device="cuda:4",
            lr=1e-5
        )

        model_path= os.path.join(trained_models_dir, f"{model_name}_best_mission_a.pth")
        predictor.model.load_state_dict(torch.load(model_path))

        predictor.model.eval()

        val_acc, val_loss = predictor.evaluate()
        print(f" Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")


    if prediction_image_option:
        image_paths = [os.path.join(save_folder_path, f) for f in os.listdir(save_folder_path) if f.endswith('.jpg') and "mask" not in f.lower() and "enlarged" not in f.lower()]
        # randomly select 10 images
        image_paths = random.sample(image_paths, 10)

        model_name = 'efficientnet_v2_1'  # Change this to the desired model name 
        #model_name = 'resnext101_32x8d' # 'resnext101_32x8d' is the best model for mission a
        predictor = RockClassifier(
            model_name=model_name,
            csv_path=csv_path,
            output_dir=trained_models_dir,
            num_epochs=80,
            batch_size=16,
            device="cuda:4",
            lr=1e-5
        )
        model_path= os.path.join(trained_models_dir, f"{model_name}_best.pth")
        predictions = predictor.predict_batch(image_paths, batch_size=16, model_path=model_path)
        
        # build a dictionary with the image path and the prediction
        predictions_dict = {}
        for image_path, prediction in zip(image_paths, predictions):
            predictions_dict[image_path] = prediction

        # sort image paths
        image_paths = natsort.natsorted(image_paths)
        # print the predictions
        for image_path in image_paths:
            print(f"{image_path}: {predictions_dict[image_path]}")


    if prediction_mask_option:
        model_name = 'resnext101_32x8d'  # Change this to the desired model name

        predictor = RockClassifier(
            model_name=model_name,
            csv_path=csv_path,
            output_dir=trained_models_dir,
            num_epochs=80,
            batch_size=16,
            device="cuda:6",
            lr=1e-5
        )

        mask_folder_path = segmentation_folder_path
        image_folder_path = image_folder_path
        output_folder_path = segmentation_filter_folder_path
        model_path= os.path.join(trained_models_dir, f"{model_name}_best.pth")
        camera_parameter_file=camera_parameters_file
        batch_size=16
        save_overlap=True

        mask_files = [f for f in os.listdir(mask_folder_path) if f.endswith('.npy')]
        N = 10
        step = len(mask_files) / N
        indices = list(range(len(mask_files)))
        indices_list = []
        for i in range(N-1):
            indices_list.append(indices[int(i * step):int((i + 1) * step)])
        indices_list.append(indices[int((N - 1) * step):])


        predictor.classify_masks(mask_folder_path, 
            image_folder_path,  
            output_folder_path, 
            model_path=model_path, 
            camera_parameter_file=camera_parameter_file, 
            batch_size=batch_size, 
            save_overlap=save_overlap, 
            indices=indices_list[9])
        print(f"\n✅ Classification complete. Results saved to {output_folder_path}")


    if change_background_to_object_option:
        segmentation_masks = [f for f in os.listdir(segmentation_filter_folder_path) if f.endswith('.npy')]
        for mask_file in tqdm.tqdm(segmentation_masks):
            mask_path = os.path.join(segmentation_filter_folder_path, mask_file)
            mask = np.load(mask_path)
            new_mask = mask + 2
            # save the new mask to overwrite the old mask
            np.save(mask_path, new_mask)


        