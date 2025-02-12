"""
This script applies foggy and glaring augmentations to the Cityscapes dataset.
It processes images in 'train', 'val', and 'test' directories and saves augmented
versions in separate folders using parallel processing (ThreadPoolExecutor).

Author: Minchan Kim
Usage: python augment_cityscapes.py
"""

from concurrent.futures import ThreadPoolExecutor
import albumentations as A
import cv2
import os
from tqdm import tqdm

# Define Paths
dataset_root = "../data/leftImg8bit"
save_root = "../data/aug_leftImg8bit"

os.makedirs(save_root, exist_ok=True)

# Define Augmentations
foggy_transform = A.Compose([A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.6, alpha_coef=0.1, p=1)])
glaring_transform = A.Compose([A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), num_flare_circles_lower=6, num_flare_circles_upper=10, src_radius=150, src_color=(255, 255, 255), p=1)])

def process_image(image_path, foggy_output_dir, glaring_output_dir):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply fog augmentation
    foggy_image = foggy_transform(image=image)["image"]
    foggy_image = cv2.cvtColor(foggy_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(foggy_output_dir, os.path.basename(image_path)), foggy_image)

    # Apply glare augmentation
    glaring_image = glaring_transform(image=image)["image"]
    glaring_image = cv2.cvtColor(glaring_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(glaring_output_dir, os.path.basename(image_path)), glaring_image)

def augment_and_save():
    for split in ["train", "val", "test"]:
        split_input_dir = os.path.join(dataset_root, split)
        split_output_dir = os.path.join(save_root, split)

        for city in os.listdir(split_input_dir):
            if city.startswith("."):  # Ignore hidden files (like .DS_Store)
                continue

            city_input_dir = os.path.join(split_input_dir, city)
            foggy_output_dir = os.path.join(split_output_dir, f"{city}_foggy")
            glaring_output_dir = os.path.join(split_output_dir, f"{city}_glaring")

            os.makedirs(foggy_output_dir, exist_ok=True)
            os.makedirs(glaring_output_dir, exist_ok=True)

            # Filter out .DS_Store and other hidden files
            image_paths = [os.path.join(city_input_dir, img) for img in os.listdir(city_input_dir) 
                           if img.endswith("_leftImg8bit.png") and not img.startswith(".")]

            # Use ThreadPoolExecutor to process images in parallel
            with ThreadPoolExecutor(max_workers=8) as executor:
                list(tqdm(executor.map(lambda img: process_image(img, foggy_output_dir, glaring_output_dir), image_paths), 
                          total=len(image_paths), desc=f"Processing {city} ({split})"))

if __name__ == "__main__":
    augment_and_save()
    print("Augmentation complete!")