import os
import shutil
from collections import defaultdict
from random import sample
from pprint import pprint

def create_dataset_folders(source_directory, destination_directory, train_split=0.8):
    # if not os.path.exists(destination_directory):
    #     os.makedirs(destination_directory)

    # os.makedirs(os.path.join(destination_directory, "images", "train"))
    # os.makedirs(os.path.join(destination_directory, "images", "val"))
    # os.makedirs(os.path.join(destination_directory, "labels", "train"))
    # os.makedirs(os.path.join(destination_directory, "labels", "val"))

    all_files = os.listdir(source_directory)
    # image_files = [f for f in all_files if f.endswith(('.jpg', '.jpeg', '.png')) and (os.path.splitext(f)[0] + '.txt' in all_files)]
    label_files = [f for f in all_files if f.endswith('.txt') and (os.path.splitext(f)[0] + '.jpg' in all_files)]

    class_counts = defaultdict(list)

    # Count and store each class occurrence in the label files
    for label_file in label_files:
        with open(os.path.join(source_directory, label_file), "r") as f:
            for line in f:
                class_id = int(line.split()[0])
                class_counts[class_id].append(os.path.splitext(label_file)[0])

    train_images = set()
    val_images = set()
    #print(class_counts)
    # Perform stratified sampling for each class
    for class_id, img_ids in class_counts.items():
        num_train = int(len(img_ids) * train_split)
        train_img_ids = sample(img_ids, num_train)
        val_img_ids = list(set(img_ids) - set(train_img_ids))
        train_images.update(train_img_ids)
        val_images.update(val_img_ids)

    # print(val_img_ids)
    # print(train_img_ids)
    #print(len(train_images.intersection(val_images)))
    
    # Copy training and validation images and labels to their respective directories
    # for img_id in train_images:
    #     image_file = img_id + '.jpg'
    #     label_file = img_id + '.txt'
    #     shutil.copy(os.path.join(source_directory, image_file), os.path.join(destination_directory, "images", "train"))
    #     shutil.copy(os.path.join(source_directory, label_file), os.path.join(destination_directory, "labels", "train"))

    # for img_id in val_images:
    #     image_file = img_id + '.jpg'
    #     label_file = img_id + '.txt'
    #     shutil.copy(os.path.join(source_directory, image_file), os.path.join(destination_directory, "images", "val"))
    #     shutil.copy(os.path.join(source_directory, label_file), os.path.join(destination_directory, "labels", "val"))

if __name__ == "__main__":
    source_directory = "D:\\Osu!_object_detection_dataset\\frames"
    destination_directory = "D:\\Osu!_object_detection_dataset\\dataset"
    create_dataset_folders(source_directory, destination_directory)
