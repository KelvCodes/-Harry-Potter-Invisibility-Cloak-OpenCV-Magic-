
  
3), np.uint8), iterations=1)
    mask_inv = cv2.bitwise_not(mask)

    # Segment out the cloak and background areas
    cloak_area = cv2.bitwise_and(background, background, mask=mask)
    visible_area = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # Combine both to create final output
    final_output = cv2.addWeighted(cloak_area, 1, visible_area, 1, 0)

    cv2.imshow("Harry's Cloak", final_output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

