import time 
import cv2

from face_detection import crop_image
from camera import conn_cam

def main():
    cam = conn_cam()
    try:
        while cam.webcam.isOpened():
            image = cam.capture_image()
            cv2.imshow('video', image)
            crop_images = crop_image(image, save_mode=True)
            '''
            Send crop_images to inference docker
            '''
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            print(crop_images)
            print("github action test!")
            time.sleep(0.1)
    finally:
        cam.destroy_cam()

if __name__ == "__main__":
    main()