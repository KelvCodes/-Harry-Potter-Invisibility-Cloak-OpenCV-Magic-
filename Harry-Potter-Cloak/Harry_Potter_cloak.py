coup.re

    win = "Cloak Controls"
    cv2.namedWindow(win)
    cv2.resizeWindow(win, 430, 300)
    setup_trackbars(win)

    # Initial background capture
    background = capture_background(cap)
    if background is None:
        return

    auto_detect = False
    fps_log = []
    prev_time = time.time()

    snapshot_dir = "Snapshots"
    os.makedirs(snapshot_dir, exist_ok=True)

    print("\nKeyboard Commands:")
    print("[q] Quit")
    print("[a] Toggle auto-detect")
    print("[b] Recapture background")
    print("[s] Save snapshot\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame not read.")
            break

        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # --------------------------------------------------------
        # Auto color detection based on a small central region
        # --------------------------------------------------------
        if auto_detect:
            h, w, _ = hsv.shape
            region = hsv[h // 2 - 30:h // 2 + 30, w // 2 - 30:w // 2 + 30]
            avg = np.median(region.reshape(-1, 3), axis=0)

            lower = np.array([max(0, avg[0] - 20), 40, 40])
            upper = np.array([min(180, avg[0] + 20), 255, 255])
        else:
            lower, upper = get_trackbar_values(win)

        # --------------------------------------------------------
        # Mask Creation
        # --------------------------------------------------------
        mask = cv2.inRange(hsv, lower, upper)
        mask = process_mask(mask)
        mask_inv = cv2.bitwise_not(mask)

        cloak_area = cv2.bitwise_and(background, background, mask=mask)
        visible_area = cv2.bitwise_and(frame, frame, mask=mask_inv)
        final = cv2.add(cloak_area, visible_area)

        final = draw_cloak_outline(final, mask)

        # --------------------------------------------------------
        # FPS Calculation
        # --------------------------------------------------------
        now = time.time()
        fps = 1 / (now - prev_time)
        prev_time = now

        fps_log.append(fps)
        if len(fps_log) > 10:
            fps_log.pop(0)
        avg_fps = sum(fps_log) / len(fps_log)

        cv2.putText(final, f"FPS: {avg_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(final, f"Mode: {'AUTO' if auto_detect else 'MANUAL'}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Invisibility Cloak", final)

        # --------------------------------------------------------
        # Keyboard Controls
        # --------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Exiting program...")
            break
        elif key == ord("b"):
            print("Recapturing background...")
            background = capture_background(cap)
        elif key == ord("a"):
            auto_detect = not auto_detect
            print("Auto-detect =", auto_detect)
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

