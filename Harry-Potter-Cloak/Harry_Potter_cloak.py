
MIO = 0.45

ALPHA_STEP = 0.05

MASK_HISTORY = 5

# ============================================================
#                     TRACKBARS
# ============================================================

def nothing(_): pass

def setup_trackbars(win):
    cv2.createTrackbar("LH", win, 60, 180, nothing)
    cv2.createTrackbar("LS", win, 50, 255, nothing)
    cv2.createTrackbar("LV", win, 40, 255, nothing)
    cv2.createTrackbar("UH", win, 120, 180, nothing)
    cv2.createTrackbar("US", win, 255, 255, nothing)
    cv2.createTrackbar("UV", win, 255, 255, nothing)

def get_trackbar_values(win):
    lower = np.array([
        cv2.getTrackbarPos("LH", win),
        cv2.getTrackbarPos("LS", win),
        cv2.getTrackbarPos("LV", win),
    ])
    upper = np.array([
        cv2.getTrackbarPos("UH", win),
        cv2.getTrackbarPos("US", win),
        cv2.getTrackbarPos("UV", win),
    ])
    return lower, upper

# ============================================================
#                     BACKGROUND
# ============================================================

def capture_background(cap):
    frames, prev = [], None
    print("Capturing background...")

    for _ in range(BACKGROUND_FRAMES):
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        if prev is not None:
            if np.mean(cv2.absdiff(frame, prev)) > MOTION_THRESHOLD:
                continue

        prev = frame.copy()
        frames.append(frame)
        cv2.imshow("Background Capture", frame)
        cv2.waitKey(1)

    cv2.destroyWindow("Background Capture")

    if not frames:
        return None

    return np.median(frames, axis=0).astype(np.uint8)

# ============================================================
#                     MASK LOGIC
# ============================================================

def refine_mask(mask):
    kernel = np.ones(MASK_KERNEL_SIZE, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 2)
    mask = cv2.GaussianBlur(mask, GAUSSIAN_BLUR_SIZE, 0)
    return mask

def stabilize_mask(mask, history):
    history.append(mask)
    return np.mean(history, axis=0).astype(np.uint8)

def mask_ratio(mask):
    return np.count_nonzero(mask) / mask.size

# ============================================================
#                 AUTO COLOR DETECTION
# ============================================================

def auto_hsv(hsv):
    h, w, _ = hsv.shape
    r = SAMPLE_BOX_SIZE
    region = hsv[h//2-r:h//2+r, w//2-r:w//2+r]
    avg = np.median(region.reshape(-1, 3), axis=0)

    lower = np.array([max(0, avg[0]-18), 40, 40])
    upper = np.array([min(180, avg[0]+18), 255, 255])
    return lower, upper

# ============================================================
#                   MAIN SYSTEM
# ============================================================

def invisibility_cloak():

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return

    win = "Controls"
    cv2.namedWindow(win)
    setup_trackbars(win)

    background = capture_background(cap)
    if background is None:
        return

    fps_log = deque(maxlen=FPS_WINDOW)
    mask_history = deque(maxlen=MASK_HISTORY)

    auto_mode = False
    alpha = 1.0
    prev_time = time.time()
    last_good_hsv = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # HSV selection
        if auto_mode:
            lower, upper = auto_hsv(hsv)
        else:
            lower, upper = get_trackbar_values(win)

        mask = cv2.inRange(hsv, lower, upper)
        mask = refine_mask(mask)

        ratio = mask_ratio(mask)

        # Adaptive correction
        if ratio < MIN_MASK_RATIO and last_good_hsv is not None:
            lower, upper = last_good_hsv
            mask = cv2.inRange(hsv, lower, upper)
        elif ratio > MAX_MASK_RATIO:
            lower[1] += 10
            mask = cv2.inRange(hsv, lower, upper)

        if MIN_MASK_RATIO < ratio < MAX_MASK_RATIO:
            last_good_hsv = (lower.copy(), upper.copy())

        mask = stabilize_mask(mask, mask_history)
        inv = cv2.bitwise_not(mask)

        cloak = cv2.bitwise_and(background, background, mask=mask)
        visible = cv2.bitwise_and(frame, frame, mask=inv)

        final = cv2.addWeighted(cloak, alpha, visible, 1-alpha, 0)

        # FPS
        now = time.time()
        fps = 1 / (now - prev_time)
        prev_time = now
        fps_log.append(fps)

        # Telemetry
        cv2.putText(final, f"FPS: {sum(fps_log)/len(fps_log):.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(final, f"Mask: {ratio*100:.1f}%", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(final, f"Alpha: {alpha:.2f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(final, f"Mode: {'AUTO' if auto_mode else 'MANUAL'}", (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        cv2.imshow("Invisibility Cloak", final)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            auto_mode = not auto_mode
        elif key == ord('b'):
            background = capture_background(cap)
        elif key == ord('['):
            alpha = max(0.0, alpha - ALPHA_STEP)
        elif key == ord(']'):
            alpha = min(1.0, alpha + ALPHA_STEP)

    cap.release()
    cv2.destroyAllWindows()

# ============================================================
#                     RUN
# ============================================================

if __name__ == "__main__":
    invisibility_cloak()
