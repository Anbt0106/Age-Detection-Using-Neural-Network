import cv2
import os

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

root_dir = 'E:/Pycharm/Age-Detection-Using-Neural-Network'

with open(os.path.join(root_dir, 'DATA/UTKFace/file_paths.txt'), 'r') as f:
    image_paths = [line.strip().replace('\\', '/') for line in f if line.strip()]

output_dir = 'DATA/UTKFace/Face_Detection'
os.makedirs(output_dir, exist_ok=True)

for image in image_paths:
    img_path = os.path.join(root_dir, 'DATA', image)
    img = cv2.imread(img_path)

    # Khử nhiễu
    denoised = cv2.bilateralFilter(img, d=10, sigmaColor=75, sigmaSpace=75)
    gray_denoised = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_denoised, scaleFactor=1.1, minNeighbors=5)

    for i, (x, y, w, h) in enumerate(faces):
        if w < 127 or h < 127:
            print(f"Face {i} in {image} is too small, skipping.")
            continue

        face = denoised[y:y + h, x:x + w]

        base_name = os.path.basename(image)
        filename, ext = os.path.splitext(base_name)
        new_filename = os.path.join(output_dir, f"{filename}_face{i}{ext}")

        cv2.imwrite(new_filename, face)
        print(f"{new_filename} is saved")
