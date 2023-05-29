import torch
from PIL import Image
import pandas as pd
# Model
from utils.general import xyxy2xywh
import os
from pathlib import Path
import shutil
from tqdm import tqdm


def row_xyxy_to_xywh(row, img_width, img_height):
    xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin
    
    row["xmin"] = center_x / img_width
    row["ymin"] = center_y / img_height
    row["xmax"] = width / img_width
    row["ymax"] = height / img_height
    
    return row

def process_image(image, model, confidence_threshold, unlabeled_images, pseudo_labeled_path):
    img_name = image.name
    img_full_path = os.path.join(unlabeled_images,img_name)
    shutil.copy(img_full_path,os.path.join(pseudo_labeled_path,img_name))
    # Inference
    results = model(im)
    test_xyxy = results.pandas().xyxy[0]
    test_xyxy = test_xyxy[test_xyxy["confidence"] >= confidence_threshold]
    test_xyxy = test_xyxy.apply(lambda row: row_xyxy_to_xywh(row, im.width, im.height), axis=1)
    output_path = os.path.join(pseudo_labeled_path,os.path.splitext(Path(img_name).name)[0] +".txt")
    test_xyxy.to_csv(output_path, sep=' ', index=False, mode="w+", columns=["class", "xmin", "ymin", "xmax", "ymax"], header=False)


if __name__ == "__main__":
    model = torch.hub.load('ultralytics/yolov5', 'custom',path='C:\\Users\\tehre\\Desktop\\yolov5\\runs\\train\\exp6\\weights\\best.pt')
    pseudo_labeled_path = "D:\\Osu!_object_detection_dataset\\pseudo_labeled\\"
    unlabeled_images = 'D:\\Osu!_object_detection_dataset\\unlabeled_images\\'

    confidence_threshold = 0.75
    for image in tqdm(os.scandir(unlabeled_images)):
        img_name = image.name
        img_full_path = os.path.join(unlabeled_images,img_name)
        # Image
        im = Image.open(os.path.join(unlabeled_images,img_name))
        #print(os.path.join(pseudo_labeled_path,img_name))
        shutil.copy(img_full_path,os.path.join(pseudo_labeled_path,img_name))
        # Inference
        results = model(im)
        # # results.print()
        # # results.show()
        test_xyxy = results.pandas().xyxy[0]
        test_xyxy = test_xyxy[test_xyxy["confidence"] >= confidence_threshold]
        # #print(test_xyxy)
        test_xyxy = test_xyxy.apply(lambda row: row_xyxy_to_xywh(row, im.width, im.height), axis=1)

        #print(test_xyxy)
        output_path = os.path.join(pseudo_labeled_path,os.path.splitext(Path(img_name).name)[0] +".txt")
        #print(output_path)
        test_xyxy.to_csv(output_path, sep=' ', index=False, mode="w+", columns=["class", "xmin", "ymin", "xmax", "ymax"], header=False)

    # img_name = 'D:\\Osu!_object_detection_dataset\\unlabeled_images\\1041915 KANA BOON Nai Mono Nedari played by Masaato.avi_24.jpg'
    # im = Image.open(os.path.join(unlabeled_images,img_name))
    # results = model(im)
    # test_xyxy = results.pandas().xyxy[0]
    # test_xyxy = test_xyxy[test_xyxy["confidence"] >= 0.4]
    # print(test_xyxy)


