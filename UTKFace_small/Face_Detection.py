import cv2
import os

# Load Haar Cascade model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

root_dir = 'E:/Pycharm/Age-Detection-Using-Neural-Network'

with open(os.path.join(root_dir, 'UTKFace_small/UTKFace_small.txt'), 'r') as f:
    image_paths = [line.strip().replace('\\', '/') for line in f if line.strip()]

output_dir = 'DATA_FACE'
os.makedirs(output_dir, exist_ok=True)

for image in image_paths:
    img_path = os.path.join(root_dir, image)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for i, (x, y, w, h) in enumerate(faces):
        if w < 72 or h < 72:
            print(f"[BỎ QUA] anh nhỏ hơn 72x72: {img_path}")
            continue

        face = img[y:y + h, x:x + w]

        base_name = os.path.basename(image)
        filename, ext = os.path.splitext(base_name)
        new_filename = os.path.join(output_dir, f"{filename}_face{i}{ext}")

        cv2.imwrite(new_filename, face)
        print(f"{new_filename} is saved.")
