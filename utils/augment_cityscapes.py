import os 
import cv2
import albumentations as A
from tqdm import tqdm
import numpy as np

dataset_root = '../data/leftImg8bit'
save_root = '../data/aug_leftImg8bit'

os.makedirs(save_root, exist_ok=True)

# Define augmentations
foggy_transform = A.Compose([
    A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.6, alpha_coef=0.1, p=1),
])
glaring_transform = A.Compose([
    A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1,
                     num_flare_circles_lower=6, num_flare_circles_upper=10,
                     src_radius=150, src_color=(255, 255, 255), p=1),
])

def augment_and_save(dataset_root, save_root):
    """
    Augment images with fog and glare effects and save them to the specified directories.
    """
    for split in ['train', 'val', 'test']:
        split_input_dir = os.path.join(dataset_root, split)
        split_output_dir = os.path.join(save_root, split)

        for city in os.listdir(split_input_dir):  # Process each city
            city_input_dir = os.path.join(split_input_dir, city)

            # Create foggy and glaring output directories for each city
            foggy_output_dir = os.path.join(split_output_dir, f"{city}_foggy")
            glaring_output_dir = os.path.join(split_output_dir, f"{city}_glaring")
            os.makedirs(foggy_output_dir, exist_ok=True)
            os.makedirs(glaring_output_dir, exist_ok=True)

            for image_name in tqdm(os.listdir(city_input_dir), desc=f"Processing {city} ({split})"):
                if image_name.endswith("_leftImg8bit.png"):  # Ensure only image files
                    image_path = os.path.join(city_input_dir, image_name)

                    # Read Image
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

                    # Apply Fog Augmentation
                    foggy_image = foggy_transform(image=image)["image"]
                    foggy_image = cv2.cvtColor(foggy_image, cv2.COLOR_RGB2BGR)  # Convert back to BGR
                    foggy_image_path = os.path.join(foggy_output_dir, image_name)
                    cv2.imwrite(foggy_image_path, foggy_image)

                    # Apply Glare Augmentation
                    glaring_image = glaring_transform(image=image)["image"]
                    glaring_image = cv2.cvtColor(glaring_image, cv2.COLOR_RGB2BGR)  # Convert back to BGR
                    glaring_image_path = os.path.join(glaring_output_dir, image_name)
                    cv2.imwrite(glaring_image_path, glaring_image)

if __name__ == "__main__":
    augment_and_save(dataset_root, save_root)
    print("Augmentation completed.")