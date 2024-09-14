import pandas as pd
import os
from PIL import Image

df = []

labels = os.listdir("/home/shusrith/Downloads/aoml-hackathon-1/dataset/train/")
labels = {j: i for i, j in enumerate(labels)}

for i in labels:
    l = os.listdir(f"/home/shusrith/Downloads/aoml-hackathon-1/dataset/train/{i}")
    for j in l:
        df.append(
            [
                f"/home/shusrith/Downloads/aoml-hackathon-1/dataset/train/{i}/{j}",
                labels[i],
            ]
        )

df = pd.DataFrame(df, columns=["image_path", "label"])
print(labels)

for i in [1558, 2884, 8691, 8768]:
    print(df["label"][i])
    Image.open(df["image_path"][i]).show()
