import torch
import torchvision
import torchvision.transforms as T
from PIL import Image, ImageDraw
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F



def visualize_boxes(image, boxes):
    # Convert the image to PIL format
    image = F.to_pil_image(image)

    # Create a draw object
    draw = ImageDraw.Draw(image)

    # Loop over the boxes and draw them on the image
    for box in boxes:
        draw.rectangle(box.tolist(), outline="red")

    return image



# Define the model and load pre-trained weights
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define the image transform
transform = T.Compose([T.ToTensor()])

# Define the object class labels (in order of the model's output)
labels = ['background', 'circle', 'slider', 'spinner']

# Load the input image
image_path = 'screenshot366.jpg'
image = Image.open(image_path).convert('RGB')

# Apply the image transform
image = transform(image)

# Make the prediction
with torch.no_grad():
    prediction = model([image])

# Retrieve the predicted bounding boxes, scores, and labels
output = model([image])[0]
print(output['labels'])
print(output['boxes'])

image = visualize_boxes(image, prediction[0]["boxes"])

# Show the image
image.show()