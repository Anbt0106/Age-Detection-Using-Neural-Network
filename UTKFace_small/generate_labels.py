import pandas as pd

file_path = 'UTKFace_small.txt'

image_paths = []
ages = []
genders = []
races = []

with open(file_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        filename = line.split('\\')[-1]
        parts = filename.split('_')
        age = int(parts[0])
        gender = int(parts[1])
        race = int(parts[2])

        image_paths.append(line)
        ages.append(age)
        genders.append(gender)
        races.append(race)

df = pd.DataFrame({'image': image_paths, 'age': ages, 'gender': genders, 'races' : races})
df.to_csv('UTKFace_small_label.csv', index=False)
print(df.head())
