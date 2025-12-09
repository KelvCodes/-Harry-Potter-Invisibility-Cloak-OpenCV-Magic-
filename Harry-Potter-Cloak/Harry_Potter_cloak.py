
# ============================================

def capture_backgrou

        ret, frame = cap.read()
        if not ret:
            crame, (FRAME_WIDTH, FRAME_HEIGHT))

        if prev is not None:
            diff = cv2.absdiff(frame, prev)
            if np.mean(diff) > MOTION_THRESHOLD:
                continue

        prev = frame.copy()

        frames.append(frame)
        cv2.imshow("Background Capture", frame)
        cv2.waitKey(1)

    cv2.destroyWindow("Background Capture")

    if len(frames) == 0:
        print("❌ Background capture failed.")
        return None

    print("✔ Background captured successfully!")
    return np.median(frames, axis=0).astype(np.uint8)


# ============================================================
#                    MASK PROCESSING
# ============================================================

def process_mask(mask):
    """Clean mask with morphology & blur."""
    kernel = np.ones(MASK_KERNEL_SIZE, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, GAUSSIAN_BLUR_SIZE, 0)
    return mask


def draw_outline(frame, mask):
    """Contour outline for cloak visualization."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, OUTLINE_COLOR, OUTLINE_THICKNESS)
    return frame


# ============================================================
#                    AUTO COLOR DETECTION
# ============================================================

def get_auto_color(hsv):
    """Auto-sample cloak color from center region."""
    h, w, _ = hsv.shape
    r = SAMPLE_BOX_SIZE

    region = hsv[h//2-r:h//2+r, w//2-r:w//2+r]
    avg = np.median(region.reshape(-1, 3), axis=0)

    lower = np.array([max(0, avg[0] - 20), 50, 50])
    upper = np.array([min(180, avg[0] + 20), 255, 255])

    return lower, upper


# ============================================================
#                    MAIN INVISIBILITY LOOP
# ============================================================

def invisibility_cloak():

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ No camera detected.")
        return

    # Trackbar window
    win = "Cloak Controls"
    cv2.namedWindow(win)
    cv2.resizeWindow(win, 430, 300)
    setup_trackbars(win)

    # Capture background
    background = capture_background(cap)
    if background is None:
        return

    auto_detect = False
    fps_history = deque(maxlen=STABILITY_WINDOW)
    prev = time.time()

    snapshot_dir = "Snapshots"
    os.makedirs(snapshot_dir, exist_ok=True)

    print("\n===== KEYBOARD COMMANDS =====")
    print("[ q ] Quit")
    print("[ a ] Toggle auto-detect mode")
    print("[ b ] Re-capture background")
    print("[ s ] Save snapshot\n")

    # Main loop
    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ----- Color mode selection -----
        if auto_detect:
            lower, upper = get_auto_color(hsv)
        else:
            lower, upper = get_trackbar_values(win)

        # ----- Mask -----
        mask = cv2.inRange(hsv, lower, upper)
        mask = process_mask(mask)
        inv = cv2.bitwise_not(mask)

        # ----- Cloak effect -----
        cloak = cv2.bitwise_and(background, background, mask=mask)
        visible = cv2.bitwise_and(frame, frame, mask=inv)
        final = cv2.add(cloak, visible)

        final = draw_outline(final, mask)

        # ====================
        # Smooth FPS display
        # ====================
        now = time.time()
        fps = 1 / (now - prev)
        prev = now

        fps_history.append(fps)
        avg_fps = sum(fps_history) / len(fps_history)

        cv2.putText(final, f"FPS: {avg_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(final, f"MODE: {'AUTO' if auto_detect else 'MANUAL'}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Invisibility Cloak", final)

        # Mask previews
        if MASK_PREVIEW:
            cv2.imshow("Mask", mask)
            cv2.imshow("Mask Inverse", inv)

        # ===== Keyboard control =====
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('a'):
            auto_detect = not auto_detect
            print("Auto-detect:", auto_detect)
        elif key == ord('b'):
            background = capture_background(cap)
        elif key == ord('s'):
            path = os.path.join(snapshot_dir, f"cloak_{int(time.time())}.png")
            cv2.imwrite(path, final)
            print("Saved →", path)

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
#                       RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    invisibility_cloak()

