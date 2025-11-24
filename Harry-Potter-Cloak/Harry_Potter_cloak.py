
"US", window_name, 255, 255, nothing)
    cv2.createTrackbar("UV", window_name, 255, 255, nothing)

def read_trackbar_values(window_name):
    """Return lower and upper HSV values from trackbars."""
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


# ============================================================
#                    INVISIBILITY CLOAK
# ============================================================

def capture_background(cap, num_frames=30):
    """
    Capture a stable background by averaging multiple frames
    for better smoothness and reduced noise.
    """
    print("\n📸 Capturing stable background. Please move away from the camera...")

    bg_frames = []
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        bg_frames.append(frame)

        cv2.imshow("Background Setup", frame)
        cv2.waitKey(1)

    cv2.destroyWindow("Background Setup")

    print("✅ Background captured successfully!")
    return np.median(bg_frames, axis=0).astype(np.uint8)


def invisibility_cloak():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("❌ Camera not found. Please connect a webcam.")
        return

    # Setup controls window
    control_window = "Cloak Controls"
    cv2.namedWindow(control_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(control_window, 400, 300)
    setup_trackbars(control_window)

    # Capture stable background
    background = capture_background(cap)

    print("\n🧥 Wear your cloak now!")
    print("👉 Press 'q' to quit.")

    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame capture failed.")
            break

        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Read HSV thresholds
        lower, upper = read_trackbar_values(control_window)

        # Mask creation + smoothing
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        # Inverse mask
        mask_inv = cv2.bitwise_not(mask)

        # Extract areas
        cloak_part = cv2.bitwise_and(background, background, mask=mask)
        visible_part = cv2.bitwise_and(frame, frame, mask=mask_inv)

        # Final output
        output = cv2.add(cloak_part, visible_part)

        # FPS Calculation
        current_time = time.time()
        fps = 1 / (current_time - last_time)
        last_time = current_time

        cv2.putText(output, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Invisibility Cloak", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n👋 Exiting program...")
            break

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
#                         RUN
# ============================================================
if __name__ == "__main__":
    invisibility_cloak()

