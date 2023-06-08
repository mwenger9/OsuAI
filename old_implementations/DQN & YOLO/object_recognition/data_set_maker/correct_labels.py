import os
import re

old_labels = [
    "hitcircle", "note", "slider", "slidercircle", "spinner", "cursor",
    "1_slider", "1_slidercircle", "2_sliderhitcircle", "2_slider",
    "clicked_slidercircle", "clicked_slider", "1_sliderhitcircle", "2_note",
    "2_hitcircle", "3_note", "4_note", "3_hitcircle", "4_hitcircle", "1_note",
    "5_note", "6_sliderhitcircle", "6_slider", "4_sliderhitcircle", "4_slider",
    "3_sliderhitcircle", "3_slider", "5_sliderhitcircle", "5_slider"
]

new_labels = [
    "note", "slider", "slidercircle", "sliderhitcircle","spinner", "cursor", 
    "clicked_slidercircle", "clicked_slider"
]

label_mapping = {}
readable_mapping = {}
for old_idx, old_label in enumerate(old_labels):
    basic_object = re.sub(r'\d_+', '', old_label)
    if basic_object in new_labels:
        new_idx = new_labels.index(basic_object)
        label_mapping[old_idx] = new_idx
        readable_mapping[old_label] = basic_object

# print(label_mapping)
# print(readable_mapping)
def process_labels(directory):
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            print(file)
            filepath = os.path.join(directory, file)
            with open(filepath, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split(" ")
                old_label = int(parts[0])
                if old_label in label_mapping:
                    new_label = label_mapping[old_label]
                    new_line = f"{new_label} {' '.join(parts[1:])}\n"
                    new_lines.append(new_line)
                    #print(f"{line} -> {new_line}")

            with open(filepath, "w+") as f:
                f.writelines(new_lines)
                
directory_path = "D:\\Osu!_object_detection_dataset\\frames"
process_labels(directory_path)
