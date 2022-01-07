import cv2
import datetime
import os

from face_detection import crop_image
from camera import conn_cam

def main(interval, show_info):
    path = f"/home/cctv-mlops-practice/data/raw/{datetime.date.today()}"
    os.makedirs(path, exist_ok=True)

    cam = conn_cam()
    try:
        t1 = datetime.datetime(1000, 1, 1)
        while cam.webcam.isOpened():
            t2 = datetime.datetime.now()
            dt = (t2-t1).total_seconds()
            if dt > interval:
                image = cam.capture_image()
                image_path = path + '/'+ t2.strftime('%H-%M-%S') 
                cv2.imshow('video', image)
                start = datetime.datetime.now()
                crop_images = crop_image(image, image_path=image_path, save_mode=True)
                finish = datetime.datetime.now()
                '''
                Send crop_images to inference service
                '''
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if show_info:
                    print('interval :', dt)
                    print('file name:', image_path)
                    print('crop & save time:', (finish-start).total_seconds())

                t1 = t2
    finally:
        cam.destroy_cam()

if __name__ == "__main__":
    main(interval=1, show_info=True)