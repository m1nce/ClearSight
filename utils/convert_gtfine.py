import os
import json
import numpy as np
from cityscapesscripts.helpers.labels import name2label

# Paths
data_path = "../data"
gtFine_path = os.path.join(data_path, "gtfine")
yolo_labels_path = os.path.join(data_path, "yolo_labels")

# Cityscapes classes (only keeping relevant ones)
class_names = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
class_mapping = {name: i for i, name in enumerate(class_names)}

# Function to convert polygons to bounding boxes
def polygon_to_bbox(polygon):
    polygon = np.array(polygon)
    x_min, y_min = np.min(polygon, axis=0)
    x_max, y_max = np.max(polygon, axis=0)
    return x_min, y_min, x_max, y_max

# Process dataset splits
for split in ["train", "val", "test"]:
    split_gt_path = os.path.join(gtFine_path, split)
    split_yolo_path = os.path.join(yolo_labels_path, split)
    os.makedirs(split_yolo_path, exist_ok=True)

    # Process each city
    for city in os.listdir(split_gt_path):
        city_gt_path = os.path.join(split_gt_path, city)
        city_yolo_path = os.path.join(split_yolo_path, city)
        os.makedirs(city_yolo_path, exist_ok=True)

        for file in os.listdir(city_gt_path):
            if file.endswith("_gtFine_polygons.json"):
                json_path = os.path.join(city_gt_path, file)

                # Read JSON annotation
                with open(json_path) as f:
                    data = json.load(f)

                img_width, img_height = data["imgWidth"], data["imgHeight"]
                yolo_labels = []

                for obj in data["objects"]:
                    label = obj["label"]

                    # Ensure label exists in class_mapping
                    if label in class_mapping:
                        x_min, y_min, x_max, y_max = polygon_to_bbox(obj["polygon"])

                        # Convert to YOLO format
                        x_center = (x_min + x_max) / 2 / img_width
                        y_center = (y_min + y_max) / 2 / img_height
                        width = (x_max - x_min) / img_width
                        height = (y_max - y_min) / img_height

                        yolo_labels.append(f"{class_mapping[label]} {x_center} {y_center} {width} {height}")

                # Save to YOLO format
                txt_file = file.replace("_gtFine_polygons.json", ".txt")
                output_path = os.path.join(city_yolo_path, txt_file)
                with open(output_path, "w") as f:
                    f.write("\n".join(yolo_labels))

print("Class Mapping (Label to ID):")
print(json.dumps(class_mapping, indent=2))