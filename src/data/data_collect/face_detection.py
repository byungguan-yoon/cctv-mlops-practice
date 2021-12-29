# import numpy as np
import cv2
import face_recognition

# img: numpy array
def crop_image(img, save_mode = True):
    face_locations = face_recognition.face_locations(img)
    face_images = []
    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_image = img[top:bottom, left:right]
        if save_mode:
            cv2.imwrite('./images/crop_img.png', face_image)
        face_images.append(face_image)
    return face_images

if __name__ == "__main__":
    image = np.asarray(cv2.imread("./images/moon.jpg"))
    print(crop_image(image))