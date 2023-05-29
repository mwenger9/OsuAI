import os
import shutil

def copy_unlabeled_images(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    all_files = os.listdir(input_directory)
    image_files = [f for f in os.listdir(input_directory) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        label_file = os.path.splitext(image_file)[0] + '.txt'
        if label_file not in all_files:
            shutil.copy(os.path.join(input_directory, image_file), os.path.join(output_directory, image_file))

if __name__ == "__main__":
    input_directory = "D:\\Osu!_object_detection_dataset\\frames"
    output_directory = "D:\\Osu!_object_detection_dataset\\unlabeled_images"
    copy_unlabeled_images(input_directory, output_directory)
