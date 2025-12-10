
    uppekbarPos("UH", window_name),
        cv2.getTrackbarPos("US", window_name),
        cv2.getTrackbarPos("UV", window_name),
    ])
    return lower, upper


# ============================================================
#                    BACKGROUND CAPTURE
# ============================================================

def capture_background(cap):
    print("\nCapturing background... Stand away from the camera.")

    frames = []
    prev = None

    for _ in range(BACKGROUND_FRAMES):
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

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
        print("Background capture failed.")
        return None

    print("Background captured successfully.")
    return np.median(frames, axis=0).astype(np.uint8)


# ============================================================
#                    MASK PROCESSING
# ============================================================

def process_mask(mask):
    kernel = np.ones(MASK_KERNEL_SIZE, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, GAUSSIAN_BLUR_SIZE, 0)
    return mask


def draw_outline(frame, mask):
    edges = cv2.Canny(mask, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, OUTLINE_COLOR, OUTLINE_THICKNESS)
    return frame


# ============================================================
#                    AUTO COLOR DETECTION
# ============================================================

def get_auto_color(hsv):
    h, w, _ = hsv.shape
    r = SAMPLE_BOX_SIZE

    region = hsv[h//2-r:h//2+r, w//2-r:w//2+r]
    avg = np.median(region.reshape(-1, 3), axis=0)

    lower = np.array([max(0, avg[0] - 20), 40, 40])
    upper = np.array([min(180, avg[0] + 20), 255, 255])

    return lower, upper


# ============================================================
#                    MANUAL CLICK COLOR PICKER
# ============================================================

clicked_color = None

def click_event(event, x, y, flags, param):
    global clicked_color
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_color = param[y, x]


def get_color_from_click(hsv_frame):
    if clicked_color is None:
        return None, None

    h, s, v = clicked_color
    lower = np.array([max(0, h - 20), 40, 40])
    upper = np.array([min(180, h + 20), 255, 255])

    return lower, upper


# ============================================================
#                    MAIN INVISIBILITY LOOP
# ============================================================

def invisibility_cloak():

    global clicked_color

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("No camera detected.")
        return

    win = "Cloak Controls"
    cv2.namedWindow(win)
    cv2.resizeWindow(win, 430, 300)
    setup_trackbars(win)

    background = capture_background(cap)
    if background is None:
        return

    auto_detect = False
    use_click_color = False

    fps_history = deque(maxlen=STABILITY_WINDOW)
    prev = time.time()
    last_auto_recalibrate = time.time()

    snapshot_dir = "Snapshots"
    os.makedirs(snapshot_dir, exist_ok=True)

    print("\n===== KEYBOARD COMMANDS =====")
    print("[ q ] Quit")
    print("[ a ] Toggle auto-detect mode")
    print("[ b ] Re-capture background")
    print("[ f ] Freeze background")
    print("[ c ] Click on cloak to auto-detect color")
    print("[ s ] Save snapshot")

    cv2.namedWindow("Invisibility Cloak")
    cv2.setMouseCallback("Invisibility Cloak", click_event)

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Color selection logic
        if use_click_color:
            lower, upper = get_color_from_click(hsv)
            if lower is None:
                lower, upper = get_trackbar_values(win)
        elif auto_detect:
            if time.time() - last_auto_recalibrate > AUTO_RECALIBRATE_INTERVAL:
                lower, upper = get_auto_color(hsv)
                last_auto_recalibrate = time.time()
            else:
                lower, upper = get_auto_color(hsv)
        else:
            lower, upper = get_trackbar_values(win)

        # Mask creation
        mask = cv2.inRange(hsv, lower, upper)
        mask = process_mask(mask)
        inv = cv2.bitwise_not(mask)

        # Cloak effect
        cloak = cv2.bitwise_and(background, background, mask=mask)
        visible = cv2.bitwise_and(frame, frame, mask=inv)
        final = cv2.add(cloak, visible)

        final = draw_outline(final, mask)

        # FPS smoothing
        now = time.time()
        fps = 1 / (now - prev)
        prev = now

        fps_history.append(fps)
        avg_fps = sum(fps_history) / len(fps_history)

        cv2.putText(final, f"FPS: {avg_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        mode = "CLICK" if use_click_color else "AUTO" if auto_detect else "MANUAL"
        cv2.putText(final, f"MODE: {mode}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Invisibility Cloak", final)

        if MASK_PREVIEW:
            cv2.imshow("Mask", mask)
            cv2.imshow("Mask Inverse", inv)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('a'):
            auto_detect = not auto_detect
            use_click_color = False
        elif key == ord('b'):
            background = capture_background(cap)
        elif key == ord('f'):
            background = frame.copy()
        elif key == ord('c'):
            clicked_color = None
            use_click_color = True
        elif key == ord('s'):
            path = os.path.join(snapshot_dir, f"cloak_{int(time.time())}.png")
            cv2.imwrite(path, final)
            print("Saved:", path)

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
#                       RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    invisibility_cloak()

