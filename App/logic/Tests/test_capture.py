from App.logic.capture import HandCapture

with HandCapture() as cap:
    while True:
        import cv2
        if not cap.read():
            break
        frame = cap.get_frame()
        landmarks = cap.get_landmarks()
        if landmarks:
            print(f"Mano detectada — primer punto: {landmarks[:3]}")
        cv2.imshow("test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()