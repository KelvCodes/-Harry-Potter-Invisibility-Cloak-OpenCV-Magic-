
        frame = cv2.flip(frame, 1)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

       
        cv2.imshow("Invisibility Effect", final_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Closing effect...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

