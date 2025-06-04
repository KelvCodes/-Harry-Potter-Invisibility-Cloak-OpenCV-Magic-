import cv2
import numpy as np

def nothing(x):
    pass  # Trackbar callback function

# ------------------- Setup -------------------
# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()
