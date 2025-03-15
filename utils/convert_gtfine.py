import os
import json

# Paths
os.chdir('..')  # Move one directory up (optional)
data_path = "data"
gtFine_path = os.path.join(data_path, "gtfine")
yolo_labels_path = os.path.join(data_path, "yolo_labels")

# Cityscapes classes (only keeping the main ones for YOLO)
class_names = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
class_mapping = {name: i for i, name in enumerate(class_names)}

# Convert polygon to bounding box
def polygon_to_bbox(polygon):
    x_min = min([p[0] for p in polygon])
    y_min = min([p[1] for p in polygon])
    x_max = max([p[0] for p in polygon])
    y_max = max([p[1] for p in polygon])
    return x_min, y_min, x_max, y_max

# Process each dataset split: train, val, test
for split in ["train", "val", "test"]:
    split_gt_path = os.path.join(gtFine_path, split)
    split_yolo_path = os.path.join(yolo_labels_path, split)

    # Ensure split folders exist
    os.makedirs(split_yolo_path, exist_ok=True)

    # Process each city folder inside the split
    for city in os.listdir(split_gt_path):
        city_gt_path = os.path.join(split_gt_path, city)
        city_yolo_path = os.path.join(split_yolo_path, city)

        # Ensure city folder exists in yolo_labels
        os.makedirs(city_yolo_path, exist_ok=True)

        for file in os.listdir(city_gt_path):
            if file.endswith("_gtFine_polygons.json"):
                with open(os.path.join(city_gt_path, file)) as f:
                    data = json.load(f)

                img_width, img_height = data["imgWidth"], data["imgHeight"]
                yolo_labels = []

                for obj in data["objects"]:
                    label = obj["label"]
                    if label in class_mapping:
                        x_min, y_min, x_max, y_max = polygon_to_bbox(obj["polygon"])

                        # Convert to YOLO format
                        x_center = (x_min + x_max) / 2 / img_width
                        y_center = (y_min + y_max) / 2 / img_height
                        width = (x_max - x_min) / img_width
                        height = (y_max - y_min) / img_height

                        yolo_labels.append(f"{class_mapping[label]} {x_center} {y_center} {width} {height}")

                # Save to YOLO annotation file inside the respective city folder
                img_name = file.replace("_gtFine_polygons.json", ".txt")
                output_file_path = os.path.join(city_yolo_path, img_name)

                with open(output_file_path, "w") as f:
                    f.write("\n".join(yolo_labels))
