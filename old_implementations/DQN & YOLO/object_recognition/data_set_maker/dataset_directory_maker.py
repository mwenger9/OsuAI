import os
import shutil
import random

def create_dataset_folders(source_directory, destination_directory, train_split=0.8):
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    
    os.makedirs(os.path.join(destination_directory, "images", "train"))
    os.makedirs(os.path.join(destination_directory, "images", "val"))
    os.makedirs(os.path.join(destination_directory, "labels", "train"))
    os.makedirs(os.path.join(destination_directory, "labels", "val"))

    all_files = os.listdir(source_directory)
    image_files = [f for f in all_files if f.endswith(('.jpg', '.jpeg', '.png')) and (os.path.splitext(f)[0] + '.txt' in all_files)]
    label_files = [f for f in all_files if f.endswith('.txt') and (os.path.splitext(f)[0] + '.jpg' in all_files)]

    
    num_train = int(len(image_files) * train_split)
    train_images = random.sample(image_files, num_train)
    val_images = list(set(image_files) - set(train_images))

    for image_file in train_images:
        shutil.copy(os.path.join(source_directory, image_file), os.path.join(destination_directory, "images", "train"))
        label_file = os.path.splitext(image_file)[0] + '.txt'
        if label_file in label_files:
            shutil.copy(os.path.join(source_directory, label_file), os.path.join(destination_directory, "labels", "train"))

    for image_file in val_images:
        shutil.copy(os.path.join(source_directory, image_file), os.path.join(destination_directory, "images", "val"))
        label_file = os.path.splitext(image_file)[0] + '.txt'
        if label_file in label_files:
            shutil.copy(os.path.join(source_directory, label_file), os.path.join(destination_directory, "labels", "val"))

if __name__ == "__main__":
    source_directory = "D:\\Osu!_object_detection_dataset\\frames"
    destination_directory = "D:\\Osu!_object_detection_dataset\\dataset"
    create_dataset_folders(source_directory, destination_directory)
