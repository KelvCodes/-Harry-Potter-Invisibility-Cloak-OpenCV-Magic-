, 68, 180, nothing)   # Lower Hue
    cv2.createTrackbar("LS", window_name, 55, 255, nothing)   # Lower Saturation
    cv2.createTrackbar("LV", window_name, 54, 255, nothing)   # Lower Value
    cv2.createTrackbar("UH", window_name, 110, 180, nothing)  # Upper Hue
    cv2.createTrackbar("US", window_name, 255, 255, nothing)  # Upper Saturation
    cv2.createTrackbar("UV", window_name, 255, 255, nothing)  # Upper Value

def read_trackbar_values(window_name):
    """Read current HSV range values from trackbars."""
    lower = np.array([
        cv2.getTrackbarPos("LH", window_name),
        cv2.getTrackbarPos("LS", window_name),
        cv2.getTrackbarPos("LV", window_name)
    ])
    upper = np.array([
        cv2.getTrackbarPos("UH", window_name),
        cv2.getTrackbarPos("US", window_name),
        cv2.getTrackbarPos("UV", window_name)
    ])
    return lower, upper

# ------------------- Main Function -------------------
def invisibility_cloak():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Could not access the camera.")
        return

    # Create window with trackbars
    controls = "Cloak Controls"
    cv2.namedWindow(controls, cv2.WINDOW_NORMAL)
    setup_trackbars(controls)

    # Capture background
    print("📸 Capturing background... Please move out of the frame and press 'b'")
    background = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        cv2.imshow("Background Setup", frame)

        if cv2.waitKey(1) & 0xFF == ord('b'):
            background = frame.copy()
            print("✅ Background captured!")
            break

    cv2.destroyWindow("Background Setup")

    # Apply invisibility effect
    print("🧥 Wear your cloak now! Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Could not read frame.")
            break

        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get HSV thresholds from trackbars
        lower, upper = read_trackbar_values(controls)

        # Create mask for cloak color
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.medianBlur(mask, 3)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

        # Inverted mask
        mask_inv = cv2.bitwise_not(mask)

        # Extract cloak area from background
        cloak_area = cv2.bitwise_and(background, background, mask=mask)
        # Extract visible area from current frame
        visible_area = cv2.bitwise_and(frame, frame, mask=mask_inv)

        # Combine both
        final_output = cv2.addWeighted(cloak_area, 1, visible_area, 1, 0)

        cv2.imshow("Invisibility Cloak", final_output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------------- Run -------------------
if __name__ == "__main__":
    invisibility_cloak()

