==========================

def draw_panel(frame, x, y, w, h, alpha=0.4):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_telemetry(frame, fps_log, mask_ratio, alpha, auto_mode):
    avg_fps = sum(fps_log) / len(fps_log)

    lines = [
        f"FPS: {avg_fps:.1f}",
        f"Mask Coverage: {mask_ratio * 100:.1f}%",
        f"Alpha: {alpha:.2f}",
        f"Mode: {'AUTO' if auto_mode else 'MANUAL'}"
    ]

    for i, text in enumerate(lines):
        cv2.putText(
            frame, text, (15, 35 + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 0), 2
        )

# =============================================================================
#                           MAIN APPLICATION
# =============================================================================

def invisibility_cloak():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Camera access failed.")
        return

    control_window = "HSV Controls"
    cv2.namedWindow(control_window)
    setup_hsv_trackbars(control_window)

    background = capture_background(cap)
    if background is None:
        print("Background capture failed.")
        return

    fps_log = deque(maxlen=FPS_WINDOW)
    mask_history = deque(maxlen=MASK_HISTORY_SIZE)

    auto_mode = False
    alpha = 1.0
    last_valid_hsv = None
    prev_time = time.time()
    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = preprocess_frame(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower, upper = (
            estimate_hsv_from_center(hsv)
            if auto_mode
            else read_hsv_trackbars(control_window)
        )

        mask = cv2.inRange(hsv, lower, upper)
        mask = refine_mask(mask)

        ratio = compute_mask_ratio(mask)

        if ratio < MIN_MASK_RATIO and last_valid_hsv:
            lower, upper = last_valid_hsv
            mask = cv2.inRange(hsv, lower, upper)

        elif ratio > MAX_MASK_RATIO:
            lower[1] = min(255, lower[1] + 10)
            mask = cv2.inRange(hsv, lower, upper)

        if MIN_MASK_RATIO < ratio < MAX_MASK_RATIO:
            last_valid_hsv = (lower.copy(), upper.copy())

        mask = stabilize_mask(mask, mask_history)
        mask = feather_mask(mask)

        inverse_mask = cv2.bitwise_not(mask)

        cloak_bg = match_lighting(frame, background)
        cloak_region = cv2.bitwise_and(cloak_bg, cloak_bg, mask=mask)
        visible_region = cv2.bitwise_and(frame, frame, mask=inverse_mask)

        flow_alpha = alpha
        if prev_frame is not None:
            flow = cv2.calcOpticalFlowFarneback(
                cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            motion_mag = np.mean(np.linalg.norm(flow, axis=2))
            flow_alpha *= np.clip(1 - motion_mag / 20, 0.6, 1.0)

        output = cv2.addWeighted(
            cloak_region, flow_alpha,
            visible_region, 1 - flow_alpha, 0
        )

        current_time = time.time()
        fps = 1 / max(current_time - prev_time, 1e-6)
        prev_time = current_time
        fps_log.append(fps)

        draw_panel(output, 5, 5, 260, 130)
        draw_telemetry(output, fps_log, ratio, alpha, auto_mode)

        cv2.imshow("Invisibility Cloak", output)

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

        prev_frame = frame.copy()

    cap.release()
    cv2.destroyAllWindows()

# =============================================================================
#                               ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    invisibility_cloak()

