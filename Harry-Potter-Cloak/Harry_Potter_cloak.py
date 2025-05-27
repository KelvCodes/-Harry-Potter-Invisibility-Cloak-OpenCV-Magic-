
	r = init_frame[:,:,2]
	b = cv2.bitwise_and(b,mask)
	g = cv2.bitwise_and(g,mask)
	r = cv2.bitwise_and(r,mask)
	blanket_area = cv2.merge((b,g,r))

	final = cv2.bitwise_or(frame_inv, blanket_area)

	cv2.imshow("Harry's Cloak",final)

	if(cv2.waitKey(3) == ord('q')):
		break;

cv2.destroyAllWindows()
cap.release()




