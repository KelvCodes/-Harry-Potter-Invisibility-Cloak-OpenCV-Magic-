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
    
cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
# Trackbars for HSV thresholds
cv2.createTrackbar("Lower Hue", "Trackbars", 68, 180, nothing)
cv2.createTrackbar("Lower Saturation", "Trackbars", 55, 255, nothing)
cv2.createTrackbar("Lower Value", "Trackbars", 54, 255, nothing)
cv2.createTrackbar("Upper Hue", "Trackbars", 110, 180, nothing)
cv2.createTrackbar("Upper Saturation", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Upper Value", "Trackbars", 255, 255, nothing)

# ------------------- Capture Background -------------------
print("Warming up the camera. Please stay out of the frame...")
cv2.waitKey(2000)  # Wait before capturing background
