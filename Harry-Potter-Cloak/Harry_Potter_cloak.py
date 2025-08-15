 wear your cloak and press 'q' to quit.")
 cv2.gecv2.getTrackbarPos("Lower Value", "Trackbars")
    
    # Create masks
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask = cv2.medianBlur(mask, 3)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    mask_inv = cv2.bitwise_not(mask)

    # Segment cloak and visible parts
    cloak = cv2.bitwise_and(background, background, mask=mask)
    visible = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # Combine both areas
    final_output = cv2.addWeighted(cloak, 1, visible, 1, 0)

    cv2.imshow("Harry's Cloak", final_output)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Closing invisibility effect...")
        break

# ------------------- Cleanup -------------------
cap.release()
cv2.destroyAllWindows()
