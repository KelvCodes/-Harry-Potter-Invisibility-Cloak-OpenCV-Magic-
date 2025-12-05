teTrackbar("UV", window_name, 255, 255, nothing)


def get_trackbar_values(window_name):
    """Return HSV lower and upper bounds."""
    lower = np.array([cv2.getTrackbarPos("LH", window_name),
                      cv2.getTrackbarPos("LS", window_name),
                      cv2.getTrackbarPos("LV", window_name)])
    upper = np.array([cv2.getTrackbarPos("UH", window_name),
                      cv2.getTrackbarPos("US", window_name),
                      cv2.getTrackbarPos("UV", window_name)])
    return lower, upper


# ============================================================
#                   BACKGROUND CAPTURE
# ============================================================

def capture_background(cap, frames_count=80, threshold=3):
    """Capture stable background using median of multiple frames and motion detection."""
    print("\nCapturing background. Please move out of frame...")

    frames = []
    last_frame = None

    for _ in range(frames_count):
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        # Motion detection: ignore frames with movement
        if last_frame is not None:
            diff = cv2.absdiff(frame, last_frame)
            if np.mean(diff) > threshold:
                continue
        last_frame = frame
        frames.append(frame)
        cv2.imshow("Capturing Background", frame)
        cv2.waitKey(1)

    cv2.destroyWindow("Capturing Background")
    background = np.median(frames, axis=0).astype(np.uint8)
    print("Background captured successfully.")
    return background


# ============================================================
#                   MASK PROCESSING
# ============================================================

def process_mask(mask, kernel_size=(5, 5)):
    """Smooth and clean mask using morphological operations."""
    kernel = np.ones(kernel_size, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    return mask


def draw_cloak_outline(frame, mask):
    """Draw contours of the cloak for visual feedback."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 255, 255), 2)
    return frame


# ============================================================
#                   MAIN INVISIBILITY LOGIC
# ============================================================

def invisibility_cloak():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Camera not detected.")
        return

    # HSV sliders window
    win = "Cloak Controls"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 420, 290)
    setup_trackbars(win)

    # Capture initial background
    background = capture_background(cap)

    auto_detect = False
    prev_time = time.time()
    fps_log = []
    snapshot_dir = "Snapshots"
    os.makedirs(snapshot_dir, exist_ok=True)

    print("\nCommands:")
    print("[q] Quit")
    print("[a] Toggle auto-detect color")
    print("[b] Recapture background")
    print("[s] Save snapshot\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame.")
            break

        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # -----------------------------
        # Auto-detection using center region
        # -----------------------------
        if auto_detect:
            h, w, _ = hsv.shape
            region = hsv[h // 3:h // 3 + 40, w // 3:w // 3 + 40]
            avg_color = np.median(region.reshape(-1, 3), axis=0)
            lower = np.array([max(0, avg_color[0] - 20), 50, 40])
            upper = np.array([min(180, avg_color[0] + 20), 255, 255])
        else:
            lower, upper = get_trackbar_values(win)

        # Mask and inverse
        mask = cv2.inRange(hsv, lower, upper)
        mask = process_mask(mask)
        mask_inv = cv2.bitwise_not(mask)

        cloak_area = cv2.bitwise_and(background, background, mask=mask)
        visible_area = cv2.bitwise_and(frame, frame, mask=mask_inv)

        final = cv2.addWeighted(cloak_area, 1, visible_area, 1, 0)
        final = draw_cloak_outline(final, mask)

        # -----------------------------
        # FPS calculation
        # -----------------------------
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        fps_log.append(fps)
        if len(fps_log) > 15:
            fps_log.pop(0)
        avg_fps = sum(fps_log) / len(fps_log)

        cv2.putText(final, f"FPS: {avg_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        mode_text = "AUTO" if auto_detect else "MANUAL"
        cv2.putText(final, f"Mode: {mode_text}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Invisibility Cloak", final)

        # -----------------------------
        # Keyboard controls
        # -----------------------------
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Exiting...")
            break
        elif key == ord("b"):
            print("Recapturing background...")
            background = capture_background(cap)
        elif key == ord("a"):
            auto_detect = not auto_detect
            print("Auto-detection:", auto_detect)
        elif key == ord("s"):
            filename = os.path.join(snapshot_dir, f"cloak_{int(time.time())}.png")
            cv2.imwrite(filename, final)
            print("Snapshot saved to", filename)

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
#                           RUN
# ============================================================

if __name__ == "__main__":
    invisibility_cloak()
