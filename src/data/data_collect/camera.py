import cv2

class conn_cam:
    def __init__(self):
        self.webcam = cv2.VideoCapture(0)

        if not self.webcam.isOpened():
            print("Could not open webcam")
            exit()

    def destroy_cam(self):
        self.webcam.release()
        cv2.destroyAllWindows()

    def capture_image(self):
        status, frame = self.webcam.read()
        return frame

if __name__ == "__main__":
    cam = conn_cam()
    try:
        while cam.webcam.isOpened():
            image = cam.capture_image()
            print(image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.destroy_cam()