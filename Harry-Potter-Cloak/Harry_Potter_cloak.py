

def get_trackbar_values(window_name):
    """Return the lower and upper HSV values from trackbars."""
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
#                      BACKGROUND CAPTURE
# ============================================================

def capture_background(cap, frames_count=70):
    """Capture a stable background by averaging multiple frames."""
    print("\nCapturing stable background. Please move out of the frame...")

    frames = []
    for _ in range(frames_count):
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        frames.append(frame)

        cv2.imshow("Background Capturing", frame)
        cv2.waitKey(1)

    cv2.destroyWindow("Background Capturing")

    background = np.median(frames, axis=0).astype(np.uint8)
    print("Background captured successfully.")
    return background


# ============================================================
#                    MASK PROCESSING
# ============================================================

def process_mask(mask):
    """Improve mask using morphological operations and blur."""
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    return mask


# ============================================================
#                 INVISIBILITY CLOAK MAIN LOGIC
# ============================================================

def invisibility_cloak():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Camera not detected.")
        return

    # Create control window
    control_window = "Cloak Controls"
    cv2.namedWindow(control_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(control_window, 400, 300)
    setup_trackbars(control_window)

    # Initial background
    background = capture_background(cap)

    print("\nControls:")
    print("Press 'q'  - Quit")
    print("Press 'b'  - Recapture background")
    print("Press 'a'  - Toggle auto-detection mode")
    print("Press 's'  - Save snapshot")

    auto_detect = False
    previous_time = time.time()
    fps_history = []

    snapshot_dir = "Snapshots"
    os.makedirs(snapshot_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame could not be read.")
            break

        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Auto cloak detection mode
        if auto_detect:
            dominant_color = np.median(hsv.reshape(-1, 3), axis=0)
            lower = np.array([
                max(0, dominant_color[0] - 15), 40, 40
            ])
            upper = np.array([
                min(180, dominant_color[0] + 15), 255, 255
            ])
        else:
            lower, upper = get_trackbar_values(control_window)

        mask = cv2.inRange(hsv, lower, upper)
        mask = process_mask(mask)

        mask_inverse = cv2.bitwise_not(mask)
        cloak_area = cv2.bitwise_and(background, background, mask=mask)
        visible_area = cv2.bitwise_and(frame, frame, mask=mask_inverse)

        final_output = cv2.addWeighted(cloak_area, 1, visible_area, 1, 0)

        # FPS calculation
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        fps_history.append(fps)
        if len(fps_history) > 20:
            fps_history.pop(0)

        avg_fps = sum(fps_history) / len(fps_history)
        cv2.putText(final_output, f"FPS: {avg_fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)

        if auto_detect:
            cv2.putText(final_output, "Auto-Detect Mode",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 0), 2)

        cv2.putText(final_output, f"Lower HSV: {lower}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(final_output, f"Upper HSV: {upper}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Invisibility Cloak", final_output)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Exiting...")
            break
        elif key == ord("b"):
            print("Re-capturing background...")
            background = capture_background(cap)
        elif key == ord("a"):
            auto_detect = not auto_detect
            print("Auto-detection:", auto_detect)
        elif key == ord("s"):
            timestamp = int(time.time())
            filepath = os.path.join(snapshot_dir, f"cloak_snapshot_{timestamp}.png")
            cv2.imwrite(filepath, final_output)
            print("Snapshot saved:", filepath)

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
#                           RUN
# ============================================================

if __name__ == "__main__":
    invisibility_cloak()

