import os
import cv2
from glob import glob

if __name__ == "__main__":
    data = glob(os.path.join("faces", "*.jpg"))
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    for path in data:
        name = path.split("/")[-1]

        image = cv2.imread(path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

        cv2.imwrite(f"results/{name}", image)
