ground
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        cv2.imshow("Background Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('b'):
            background = frame.copy()
            print("✅ Background captured.")
            break

    cv2.destroyWindow("Background Capture")

    # Main invisibility effect loop
    print("🧥 Wear your cloak now! Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Could not read frame.")
            break

        frame = cv2.flip(frame, 1)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_bound, upper_bound = get_hsv_bounds(trackbar_window)

        # Create mask
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        mask = cv2.medianBlur(mask, 3)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        mask_inv = cv2.bitwise_not(mask)

        # Apply invisibility effect
        cloak_area = cv2.bitwise_and(background, background, mask=mask)
        visible_area = cv2.bitwise_and(frame, frame, mask=mask_inv)
        final_frame = cv2.addWeighted(cloak_area, 1, visible_area, 1, 0)

        cv2.imshow("Invisibility Effect", final_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Closing effect...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

