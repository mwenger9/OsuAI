import pygetwindow as gw
import pyautogui
import time
import cv2
import numpy as np
from PIL import ImageGrab
import torch



# Top left : 380,145
# Bottom left : 380,1000
# Top right : 1530,145
# Bottom right : 1530,1000

# x1 , y1 = 380,145
# x2, y2 = 1530,1015
# cropped = img[y1:y2, x1:x2]
# cv2.imshow("cropped",cropped)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# top_left = (305, 50)
# bottom_left = (305, 1070)
# top_right = (1615, 50)
# bottom_right = (1615, 1070)



def capture_osu_playfield():
    osu_window = gw.getWindowsWithTitle('osu!')[0]
    if osu_window is not None:
        top_left = (osu_window.left + 305, osu_window.top + 50)
        bottom_left = (osu_window.left + 305, osu_window.top + 1070)
        top_right = (osu_window.left + 1615, osu_window.top + 50)
        bottom_right = (osu_window.left + 1615, osu_window.top + 1070)
        screenshot = ImageGrab.grab(bbox=(top_left[0], top_left[1], bottom_right[0], bottom_right[1]))
        return screenshot
    return None



if __name__ == "__main__":
    model = torch.hub.load('ultralytics/yolov5', 'custom',path='C:\\Users\\tehre\\Desktop\\yolov5\\runs\\train\\exp6\\weights\\best.pt')

    osu_window = gw.getWindowsWithTitle('osu!')[0]
    osu_window.activate()

    while True:
        screenshot = capture_osu_playfield()
        if screenshot is not None:
            screenshot_np = np.array(screenshot)
            screenshot_np = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

            prediction = model(screenshot_np,conf=0.5)

            prediction.print()
            
            screenshot_with_boxes = prediction.render()[0]
            screenshot_with_boxes_np = np.array(screenshot_with_boxes)

            if screenshot_with_boxes is not None:
                screenshot_with_boxes_np = np.array(screenshot_with_boxes)
                cv2.imshow('osu! Window', screenshot_with_boxes_np)
            else:
                cv2.imshow('osu! Window', screenshot_np)

        key = cv2.waitKey(1)
        if key == ord('q'):  # Press 'q' to quit the loop
            break

    cv2.destroyAllWindows()
















# while True:
#     screenshot = pyautogui.screenshot(region=(osu_window.left, osu_window.top, osu_window.width, osu_window.height))
    
#     # Convert the PIL.Image.Image to a numpy array
#     screenshot_np = np.array(screenshot)
    
#     # Convert the image from RGB to BGR format (required by OpenCV)
#     screenshot_np = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
#     # Crop the image
#     cropped_screenshot = screenshot_np[top_left[1]:bottom_left[1], top_left[0]:top_right[0]]

#     # Display the cropped screenshot using OpenCV
#     cv2.imshow("Real-time Cropped Screenshot", cropped_screenshot)
    
#     # Use your object detection model to analyze the screenshot here

#     # Introduce a small delay (in milliseconds) between screenshots to reduce performance impact
#     key = cv2.waitKey(10)
    
#     # Break the loop if the 'q' key is pressed
#     if key == ord('q'):
#         break

# # Destroy the OpenCV window
# cv2.destroyAllWindows()