import os
import pandas as pd

base_path = 'E:\\Pycharm\\Age-Detection-Using-Neural-Network\\'
folder_path = os.path.join(base_path, 'UTKFace_small', 'DATA_FACE')
output_txt = os.path.join(base_path, 'UTKFace_small', 'DATA_FACE.txt')
output_csv = os.path.join(base_path, 'UTKFace_small', 'DATA_FACE_Label.csv')

image_names = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]

image_paths = []
ages = []
genders = []
races = []

with open(output_txt, 'w') as f:
    for name in image_names:
        relative_path = os.path.join('DATA_FACE', name)
        f.write(relative_path + '\n')

        parts = name.split('_')
        if len(parts) >= 4:
            age = int(parts[0])
            gender = int(parts[1])
            race = int(parts[2])

            image_paths.append(relative_path)
            ages.append(age)
            genders.append(gender)
            races.append(race)

print(f"Đã lưu {len(image_names)} tên ảnh vào '{output_txt}'.")

df = pd.DataFrame({'image': image_paths, 'age': ages, 'gender': genders, 'races': races})
df.to_csv(output_csv, index=False)
print(f"{output_csv} is saved")
print(df.head())
