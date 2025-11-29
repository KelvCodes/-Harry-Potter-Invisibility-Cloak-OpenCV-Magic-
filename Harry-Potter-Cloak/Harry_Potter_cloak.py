
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
#                       BACKGROUND CAPTURE
# ============================================================

def capture_background(cap, frames_count=70):
    """Capture a stable background by averaging multiple frames."""
    print("\n📸 Capturing stable background. PLEASE move away...")
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
#                     MASK PROCESSING
# ============================================================

def process_mask(mask):
    """Improve mask using morphology + blur."""
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    return mask

# ============================================================
#                 INVISIBILITY CLOAK LOGIC
# ============================================================

def invisibility_cloak():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Camera not detected.")
        return

    # Control Window
    controls = "Cloak Controls"
    cv2.namedWindow(controls, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(controls, 400, 300)
    setup_trackbars(controls)

    # Background initialization
    background = capture_background(cap)

    print("\n🧥 Wear your cloak now!")
    print("👉 Press 'q' to Quit.")
    print("👉 Press 'b' to recapture background.")
    print("👉 Press 'a' to toggle Auto-Detection mode.")
    print("👉 Press 's' to save a snapshot.")

    auto_detect = False
    prev_time = time.time()
    fps_list = []
    snapshot_dir = "Snapshots"
    os.makedirs(snapshot_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Could not read frame.")
            break
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # -----------------------------
        # AUTO-DETECTION MODE
        # -----------------------------
        if auto_detect:
            dominant_color = np.median(hsv.reshape(-1, 3), axis=0)
            lower = np.array([max(0, dominant_color[0] - 15), 40, 40])
            upper = np.array([min(180, dominant_color[0] + 15), 255, 255])
        else:
            lower, upper = get_trackbar_values(controls)

        # Create mask and improve
        mask = cv2.inRange(hsv, lower, upper)
        mask = process_mask(mask)
        mask_inv = cv2.bitwise_not(mask)

        # Extract areas
        cloak_area = cv2.bitwise_and(background, background, mask=mask)
        visible_area = cv2.bitwise_and(frame, frame, mask=mask_inv)

        # Merge final output
        final_output = cv2.addWeighted(cloak_area, 1, visible_area, 1, 0)

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        fps_list.append(fps)
        if len(fps_list) > 20:
            fps_list.pop(0)
        smooth_fps = sum(fps_list) / len(fps_list)
        cv2.putText(final_output, f"FPS: {smooth_fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)
        if auto_detect:
            cv2.putText(final_output, "AUTO-DETECT MODE",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 0), 2)

        # Display HSV range on frame
        cv2.putText(final_output, f"Lower HSV: {lower}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(final_output, f"Upper HSV: {upper}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("Invisibility Cloak", final_output)
        key = cv2.waitKey(1) & 0xFF

        # -----------------------------
        # KEYBOARD SHORTCUTS
        # -----------------------------
        if key == ord('q'):
            print("\n👋 Exiting...")
            break
        if key == ord('b'):
            print("\n♻️ Re-capturing background...")
            background = capture_background(cap)
        if key == ord('a'):
            auto_detect = not auto_detect
            print(f"\n🔁 Auto-detection toggled: {auto_detect}")
        if key == ord('s'):
            snapshot_path = os.path.join(snapshot_dir, f"cloak_{int(time.time())}.png")
            cv2.imwrite(snapshot_path, final_output)
            print(f"📷 Snapshot saved: {snapshot_path}")

    cap.release()
    cv2.destroyAllWindows()

# ============================================================
#                         RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    invisibility_cloak()

