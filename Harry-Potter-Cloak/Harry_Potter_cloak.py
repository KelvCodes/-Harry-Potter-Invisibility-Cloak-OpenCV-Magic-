g)
cv2.createTrackbar("Upper Value", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Lower Hue", "Trackbars", 68, 180, nothing)
cv2.createTrackbar("Lower Saturation", "Trackbars", 55, 255, nothing)
cv2.createTrackbar("Lower Value", "Trackbars", 54, 255, nothing)

# Allow the camera to warm up and capture the static background
print("Capturing background...")
while True:
    ret, background = cap.read()
    if ret:
        background = cv2.flip(background, 1)
        break
    cv2.waitKey(1000)

print("Background captured. Starting invisibility cloak...")

# Process frames continuously
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Read trackbar values
    upper_h = cv2.getTrackbarPos("Upper Hue", "Trackbars")
    upper_s = cv2.getTrackbarPos("Upper Saturation", "Trackbars")
    upper_v = cv2.getTrackbarPos("Upper Value", "Trackbars")
    lower_h = cv2.getTrackbarPos("Lower Hue", "Trackbars")
    lower_s = cv2.getTrackbarPos("Lower Saturation", "Trackbars")
    lower_v = cv2.getTrackbarPos("Lower Value", "Trackbars")

    lower_bound = np.array([lower_h, lower_s, lower_v])
    upper_bound = np.array([upper_h, upper_s, upper_v])

    # Create mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask = cv2.medianBlur(mask, 3)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    mask_inv = cv2.bitwise_not(mask)

    # Segment out the cloak and background areas
    cloak_area = cv2.bitwise_and(background, background, mask=mask)
    visible_area = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # Combine both to create final output
    final_output = cv2.addWeighted(cloak_area, 1, visible_area, 1, 0)

    cv2.imshow("Harry's Cloak", final_output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

