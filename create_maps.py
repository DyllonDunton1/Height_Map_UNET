import os
from torchvision import io, transforms, models
from torch.optim.lr_scheduler import StepLR
import torch
from torch import nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision.transforms.functional as F
from torch.nn.functional import interpolate
import numpy as np
import csv

from zoe_depth_map.video_convert import create_pano, create_depth_map_from_pano
from segmentation_trainer.segmentation_create import segment_model

import warnings
warnings.filterwarnings("ignore")

data_path = "zoe_depth_map/video_data"
video_to_convert = "suburb.mp4"
pano_pil = create_pano(f"{data_path}/{video_to_convert}")
depth_map_tensor = create_depth_map_from_pano(pano_pil)

#Now that we have the depth tensor, we need to get split up the panorama and run the segmentation model
#First find the height and width of the pano
to_tensor = transforms.ToTensor()
shrink_tensor = transforms.Resize((224,224))
pano_tensor = to_tensor(pano_pil)
print(depth_map_tensor.shape)
print(pano_tensor.shape)

#Split image by dividing height by 2 and the width by num_images = 5
num_images = 5
num_levels = 2
pano_shape = depth_map_tensor.shape
split_width = pano_shape[1]//num_images
split_height = pano_shape[0]//num_levels
print(split_width, split_height)

classes_tensor = torch.zeros(pano_shape)
for y in range(num_levels):
    for x in range(num_images):
        top = y*split_height
        left = x*split_width
        bottom = top+split_height if y+1 != num_levels else pano_shape[0]
        right = left+split_width if x+1 != num_images else pano_shape[1]
        print(bottom, right)

        section_tensor = pano_tensor[:, top:bottom, left:right]
        small_tensor = shrink_tensor(section_tensor)
        print(small_tensor.shape)
        segmented_tensor = segment_model(small_tensor).float()
        print(segmented_tensor.shape)
        segmented_enlarged_tensor = interpolate(segmented_tensor.unsqueeze(0).unsqueeze(0), (bottom-top, right-left), mode='nearest').squeeze(0).squeeze(0).int()
        print(segmented_enlarged_tensor.shape)
        for i, col in enumerate(segmented_enlarged_tensor):
            for j, pix in enumerate(col):
                classes_tensor[i+top][j+left] = segmented_enlarged_tensor[i][j]

print(classes_tensor.shape)

colors = [[0, 0, 0],[128, 64, 128],[130, 76, 0],[0, 102, 0],[112, 103, 87],[28, 42, 168],[48, 41, 30],[0, 50, 89],[107, 142, 35],[70, 70, 70],[102, 102, 156],[254, 228, 12],[254, 148, 12],[190, 153, 153],[153, 153, 153],[255, 22, 96],[102, 51, 0],[9, 143, 150],[119, 11, 32],[51, 51, 0],[190, 250, 190],[112, 150, 146],[2, 135, 115],[255, 0, 0]]
rgb_tensor = np.zeros((pano_shape[1],pano_shape[0],3))
for y, row in enumerate(classes_tensor):
    for x, index in enumerate(row):
        rgb = colors[int(index)]
        rgb_tensor[x][y] = rgb


to_pil_image = transforms.ToPILImage()
segmented_image_pil = to_pil_image(rgb_tensor).rotate(-90, expand=True)
depth_image_pil = to_pil_image(depth_map_tensor)
segmented_image_pil.show()
depth_image_pil.show()
pano_pil.show()

         

