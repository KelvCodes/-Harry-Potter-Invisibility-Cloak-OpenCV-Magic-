rPo
                      cv2.getTrackbarPos("LV", window_name)])

    upper = np.array([cv2.getTrackbarPos("UH", window_name),
                      cv2.getTrackbarPos("US", window_name),
                      cv2.getTrackbarPos("UV", window_name)])

    return lower, upper


# ============================================================
#                 BACKGROUND CAPTURE
# ============================================================

def capture_background(cap, frames_count=80):
    """Capture background using median of multiple frames."""

    print("\n📸 Capturing background... MOVE OUT OF FRAME.")

    frames = []
    last_frame = None

    for i in range(frames_count):

        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        # Motion detection (reject moving frames)
        if last_frame is not None:
            diff = cv2.absdiff(frame, last_frame)
            if np.mean(diff) > 3:  # movement detected
                continue

        last_frame = frame
        frames.append(frame)

        cv2.imshow("Background Capturing", frame)
        cv2.waitKey(1)

    cv2.destroyWindow("Background Capturing")

    background = np.median(frames, axis=0).astype(np.uint8)

    print("✅ Background captured.")
    return background


# ============================================================
#                 MASK PROCESSING
# ============================================================

def process_mask(mask):
    """Apply smoothing and reduce noise."""
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    return mask


# ============================================================
#             DRAW CLEAN EDGE OUTLINE ON CLOAK
# ============================================================

def draw_cloak_outline(frame, mask):
    """Draw contour outline of cloak for visual feedback."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 255, 255), 2)
    return frame


# ============================================================
#                 MAIN INVISIBILITY LOGIC
# ============================================================

def invisibility_cloak():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("❌ Camera not detected.")
        return

    # HSV control window
    win = "Cloak Controls"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 420, 290)
    setup_trackbars(win)

    # Capture background
    background = capture_background(cap)

    auto_detect = False
    previous_time = time.time()
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
            print("Couldn’t read frame.")
            break

        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ---------------------------------------------------------
        # AUTO COLOR DETECTION (smarter)
        # ---------------------------------------------------------
        if auto_detect:
            # Sample center region for accuracy
            h, w, _ = hsv.shape
            sample = hsv[h // 3:h // 3 + 40, w // 3:w // 3 + 40]
            avg = np.median(sample.reshape(-1, 3), axis=0)

            lower = np.array([max(0, avg[0] - 20), 50, 40])
            upper = np.array([min(180, avg[0] + 20), 255, 255])
        else:
            lower, upper = get_trackbar_values(win)

        # Create mask
        mask = cv2.inRange(hsv, lower, upper)
        mask = process_mask(mask)

        # Areas
        inv = cv2.bitwise_not(mask)
        cloak_area = cv2.bitwise_and(background, background, mask=mask)
        visible_area = cv2.bitwise_and(frame, frame, mask=inv)

        final = cv2.add(cloak_area, visible_area)

        # Draw cloak outline
        final = draw_cloak_outline(final, mask)

        # ---------------------------------------------------------
        # FPS Calculation
        # ---------------------------------------------------------
        current = time.time()
        fps = 1 / (current - previous_time)
        previous_time = current
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

        # ---------------------------------------------------------
        # KEYBOARD CONTROLS
        # ---------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("👋 Exiting...")
            break

        elif key == ord("b"):
            print("♻ Recapturing background...")
            background = capture_background(cap)

        elif key == ord("a"):
            auto_detect = not auto_detect
            print("Auto-detection:", auto_detect)

        elif key == ord("s"):
            filename = os.path.join(snapshot_dir, f"cloak_{int(time.time())}.png")
            cv2.imwrite(filename, final)
            print("📸 Snapshot saved to", filename)

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
#                       RUN APP
# ============================================================

if __name__ == "__main__":
    invisibility_cloak()

