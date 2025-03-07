import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from pathlib import Path

class ConditionDataset(Dataset):
    def __init__(self, original_path, augmented_path, transform=None):
        """
        Creates ConditionDataset object that assigns city folders to certain 
        conditions (foggy, glaring, clear).

        Args:
            original_path: path to original folder (usually folder that contains `clear`)
            augmented_path: path to augmented folder
            transform: transformation function to apply to image
        """
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_mapping = {'clear': 0, 'foggy': 1, 'glaring': 2}

        # loads clear images
        for city_folder in sorted(Path(original_path).iterdir()):
            for img_path in sorted(city_folder.iterdir()):
                self.image_paths.append(img_path)
                self.labels.append(self.class_mapping['clear'])

        # loads foggy and glaring images
        for city_folder in sorted(Path(augmented_path).iterdir()):
            if 'foggy' in city_folder.name:
                label = self.class_mapping['foggy']
            elif 'glaring' in city_folder.name:
                label = self.class_mapping['glaring']
            else:
                continue # skip other folders

            for img_path in sorted(city_folder.iterdir()):
                self.image_paths.append(img_path)
                self.labels.append(label)

    def __len__(self):
        """
        Getter method for length of image paths.

        Returns:
            Number of image paths.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Getter method for a certain image and transforms it using the predefined transform
        function.

        Args:
            idx: index of image path to get

        Returns: 
            Image and label (foggy, glaring, clear). 
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        image = self.transform(image)

        label = self.labels[idx]
        return image, label
                