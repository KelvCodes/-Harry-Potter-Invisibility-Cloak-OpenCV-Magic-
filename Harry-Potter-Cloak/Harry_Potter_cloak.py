
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

while True:
    ret, background = cap.read()
    if not ret:
        continue
    background = cv2.flip(background, 1)
    cv2.imshow("Background Frame", background)
    if cv2.waitKey(1) & 0xFF == ord('b'):
        print("Background captured.")
        break

cv2.destroyWindow("Background Frame")

# ------------------- Main Loop -------------------
print("Now wear your cloak and press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to fetch frame.")
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get current HSV threshold values
    lower_h = cv2.getTrackbarPos("Lower Hue", "Trackbars")
    lower_s = cv2.getTrackbarPos("Lower Saturation", "Trackbars")
    lower_v = cv2.getTrackbarPos("Lower Value", "Trackbars")
    upper_h = cv2.getTrackbarPos("Upper Hue", "Trackbars")
    upper_s = cv2.getTrackbarPos("Upper Saturation", "Trackbars")
    upper_v = cv2.getTrackbarPos("Upper Value", "Trackbars")

    lower_bound = np.array([lower_h, lower_s, lower_v])
    upper_bound = np.array([upper_h, upper_s, upper_v])

    # Create masks
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask = cv2.medianBlur(mask, 3)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    mask_inv = cv2.bitwise_not(mask)

    # Segment cloak and visible parts
    cloak = cv2.bitwise_and(background, background, mask=mask)
    visible = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # Combine both areas
    final_output = cv2.addWeighted(cloak, 1, visible, 1, 0)

    cv2.imshow("Harry's Cloak", final_output)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Closing invisibility effect...")
        break

# ------------------- Cleanup -------------------
cap.release()
cv2.destroyAllWindows()
