import os
import pandas as pd
from PIL import Image

input_path = []
label = []

# Collect image paths and labels
for class_name in os.listdir('PetImages'):
    class_path = os.path.join('PetImages', class_name)
    if not os.path.isdir(class_path):
        continue
    for filename in os.listdir(class_path):
        full_path = os.path.join(class_path, filename)
        input_path.append(full_path)
        label.append(0 if class_name == 'Cat' else 1)

# Create dataframe
df = pd.DataFrame({'images': input_path, 'label': label})
df = df.sample(frac=1).reset_index(drop=True)

# Detect corrupted/unopenable images
bad_images = []
for image_path in df['images']:
    try:
        img = Image.open(image_path)
        img.verify()  # Check for corruption
    except:
        bad_images.append(image_path)

# Remove bad images from dataframe
df = df[~df['images'].isin(bad_images)]

# Save to CSV
df.to_csv('images_clf.csv', index=False)
print("Saved cleaned dataset to 'images_clf.csv'.")
print("Final dataset size:", len(df))