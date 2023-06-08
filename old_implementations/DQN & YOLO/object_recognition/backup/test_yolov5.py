import torch
import cv2
import numpy as np

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='neurochan.pt', force_reload=True)

# Define the classes and colors for bounding box visualization
classes = ['circle', 'slider', 'spinner']
colors = [[0, 255, 255], [255, 0, 255], [255, 255, 0]]

# Load the input image
image_path = 'screenshot366.png'
image = cv2.imread(image_path)

# Perform object detection
results = model(image)

# Retrieve the predicted bounding boxes, scores, and labels
boxes = results.xyxy[0, :, :4].cpu().numpy()
scores = results.xyxy[0, :, 4].cpu().numpy()
labels = results.xyxy[0, :, 5].cpu().numpy().astype(int)

# Visualize the bounding boxes on the image
for box, score, label in zip(boxes, scores, labels):
    if score > 0.5:
        color = colors[label-1]
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(image, f'{classes[label-1]}: {score:.2f}', (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the output image
cv2.imshow('Output', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
