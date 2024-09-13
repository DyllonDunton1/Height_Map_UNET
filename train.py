from PIL import Image
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

num_images = 5000
test_amount = 500
home_dir = "/home/dunto/Height_Map_UNET/"

train_t = []
train_h = []
test_t = []
test_h = []

print("Starting up data extraction...")
order = list(range(num_images))
random.shuffle(order)

for i, datum in enumerate(order):
    img_t = Image.open(f"{home_dir}data/terrain/{str(datum+1).zfill(4)}_t.png")
    img_h = Image.open(f"{home_dir}data/height/{str(datum+1).zfill(4)}_h.png")

    if i < test_amount:
        test_t.append(np.array(img_t))
        test_h.append(np.array(img_h))
    else:
        train_t.append(np.array(img_t))
        train_h.append(np.array(img_h))

    if i == 0:
        print("0.0% done")
    if (i+1)%(test_amount) == 0:
        print(f"{float(i+1)/50.0}% Done")

print(f"# Training terrain: {len(train_t)}")
print(f"# Training heights: {len(train_h)}")
print(f"# Testing terrain: {len(test_t)}")
print(f"# Testing heights: {len(test_h)}")
print(f"Terrain Shape: {train_t[0].shape}")
print(f"Height Shape: {train_h[0].shape}")


