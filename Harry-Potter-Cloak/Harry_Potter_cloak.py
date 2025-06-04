import cv2
import numpy as np

def nothing(x):
    pass  # Trackbar callback function

# ------------------- Setup -------------------
# Start video capture
cap = cv2.VideoCapture(0)
