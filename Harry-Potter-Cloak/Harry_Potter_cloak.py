indow_name):
    """Create HSV range trackbars for cloak color tuning."""
    cv2.createTrackbar("LH", window_name, 68, 180, nothing)
    cv2.createTrackbar("LS", window_name, 55, 255, nothing)
    cv2.createTrackbar("LV", window_name, 54, 255, nothing)

    cv2.createTrackbar("UH", window_name, 110, 180, nothing)
    cv2.createTrackbar("US", window_name, 255, 255, nothing)
    cv2.createTrackbar("UV", window_name, 255, 255, nothing)

def get_trackbar_values(window_name):
    """Return lower and upper HSV values from trackbars."""
    lower = np.array([
        cv2.getTrackbarPos("LH", window_name),
        cv2.getTrackbarPos("LS", window_name),
        cv2.getTrackbarPos("LV", window_name),
    ])

    upper = np.array([
        cv2.getTrackbarPos("UH", window_name),
        cv2.getTrackbarPos("US", window_name),
        cv2.getTrackbarPos("UV", window_name),
    ])

    return lower, upper


# ============================================================
#                   BACKGROUND CAPTURE
# ============================================================

def capture_background(cap, frames_count=50):
    """
    Capture a stable background by averaging multiple frames.
    Higher frame count = smoother + more stable background.
    """
    print("\n📸 Capturing stable background. Move away from the camera...")

    collected_frames = []

    for i in range(frames_count):
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        collected_frames.append(frame)

        cv2.imshow("Background Capturing...", frame)
        cv2.waitKey(1)

    cv2.destroyWindow("Background Capturing...")

    background = np.median(collected_frames, axis=0).astype(np.uint8)
    print("✅ Background captured successfully!")
    return background


# ============================================================
#                   INVISIBILITY CLOAK LOGIC
# ============================================================

def invisibility_cloak():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("❌ Camera not detected.")
        return

    # Control window
    controls = "Cloak Controls"
    cv2.namedWindow(controls, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(controls, 400, 300)
    setup_trackbars(controls)

    # Capture clean background
    background = capture_background(cap)

    print("\n🧥 Wear your cloak now!")
    print("👉 Press 'q' to quit.")

    prev_time = time.time()
    fps_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Could not read frame.")
            break

        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # HSV mask
        lower, upper = get_trackbar_values(controls)
        mask = cv2.inRange(hsv, lower, upper)

        # Better noise removal
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        # Inverse mask
        mask_inv = cv2.bitwise_not(mask)

        cloak_area = cv2.bitwise_and(background, background, mask=mask)
        visible_area = cv2.bitwise_and(frame, frame, mask=mask_inv)

        output = cv2.add(cloak_area, visible_area)

        # FPS Calculation (smoothed)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        fps_list.append(fps)
        if len(fps_list) > 20:
            fps_list.pop(0)

        smooth_fps = sum(fps_list) / len(fps_list)

        cv2.putText(output, f"FPS: {smooth_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Invisibility Cloak", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n👋 Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
#                         RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    invisibility_cloak()


