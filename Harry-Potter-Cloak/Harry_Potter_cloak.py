ckba
    """Callback for trackbar, does nothing but required by OpenCV."""
    pass

def create_trackbars(window_name):
    """Create HSV trackbars for color range adjustment."""
    cv2.createTrackbar("Lower Hue", window_name, 68, 180, trackbar_callback)
    cv2.createTrackbar("Lower Saturation", window_name, 55, 255, trackbar_callback)
    cv2.createTrackbar("Lower Value", window_name, 54, 255, trackbar_callback)
    cv2.createTrackbar("Upper Hue", window_name, 110, 180, trackbar_callback)
    cv2.createTrackbar("Upper Saturation", window_name, 255, 255, trackbar_callback)
    cv2.createTrackbar("Upper Value", window_name, 255, 255, trackbar_callback)

def get_hsv_bounds(window_name):
    """Retrieve HSV bounds from trackbars."""
    lower = np.array([
        cv2.getTrackbarPos("Lower Hue", window_name),
        cv2.getTrackbarPos("Lower Saturation", window_name),
        cv2.getTrackbarPos("Lower Value", window_name)
    ])
    upper = np.array([
        cv2.getTrackbarPos("Upper Hue", window_name),
        cv2.getTrackbarPos("Upper Saturation", window_name),
        cv2.getTrackbarPos("Upper Value", window_name)
    ])
    return lower, upper

# ------------------- Main Program -------------------
def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera not accessible.")
        return

    trackbar_window = "Color Controls"
    cv2.namedWindow(trackbar_window, cv2.WINDOW_NORMAL)
    create_trackbars(trackbar_window)

    # Capture background
    print("📸 Preparing background capture. Stay out of the frame!")
    cv2.waitKey(2000)

    background = None
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        cv2.imshow("Background Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('b'):
            background = frame.copy()
            print("✅ Background captured.")
            break

    cv2.destroyWindow("Background Capture")

    # Main invisibility effect loop
    print("🧥 Wear your cloak now! Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Could not read frame.")
            break

        frame = cv2.flip(frame, 1)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_bound, upper_bound = get_hsv_bounds(trackbar_window)

        # Create mask
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        mask = cv2.medianBlur(mask, 3)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        mask_inv = cv2.bitwise_not(mask)

        # Apply invisibility effect
        cloak_area = cv2.bitwise_and(background, background, mask=mask)
        visible_area = cv2.bitwise_and(frame, frame, mask=mask_inv)
        final_frame = cv2.addWeighted(cloak_area, 1, visible_area, 1, 0)

        cv2.imshow("Invisibility Effect", final_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Closing effect...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

