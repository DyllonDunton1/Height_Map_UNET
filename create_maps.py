import os
from torchvision import io, transforms, models
from torch.optim.lr_scheduler import StepLR
import torch
from torch import nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision.transforms.functional as F
from torch.nn.functional import interpolate, pad
import numpy as np
import csv
from HeightmapVisualization.pathfinding.path_gen import Pathfinding

from zoe_depth_map.video_convert import create_pano, create_depth_map_from_pano
from segmentation_trainer.segmentation_create import segment_model

import warnings
warnings.filterwarnings("ignore")

def main(filename, rot):

    data_path = "zoe_depth_map/video_data"
    video_to_convert = f"{filename}.mp4"
    pano_pil = create_pano(f"{data_path}/{video_to_convert}", rot)
    depth_map_tensor = create_depth_map_from_pano(pano_pil)

    #Now that we have the depth tensor, we need to get split up the panorama and run the segmentation model
    #First find the height and width of the pano
    to_tensor = transforms.ToTensor()
    shrink_tensor = transforms.Resize((112,112))
    to_pil_image = transforms.ToPILImage()
    pano_tensor = to_tensor(pano_pil)
    print(f"Depth Map Shape: {depth_map_tensor.shape}")
    print(f"RGB Original shape: {pano_tensor.shape}")

    #Split image by dividing height by 2 and the width by num_images = 5
    #Resize to be multiple of 224 resolution
    pano_c, pano_h, pano_w = pano_tensor.shape
    print(f"Pano Shape (c,h,w): ({pano_c}/{pano_h}/{pano_w})")
    patch_size = 200
    patches_wide = pano_w//patch_size
    working_width = patches_wide*patch_size
    patches_high = pano_h//patch_size
    working_height = patches_high*patch_size
    unsqueeze_tensor = pano_tensor.unsqueeze(0)
    print(f"Unsqueeze: {unsqueeze_tensor.shape}")
    working_tensor = interpolate(unsqueeze_tensor, size=(working_height, working_width), mode='bilinear', align_corners=False).squeeze(0)
    print(f"Patched (Wide/High): {patches_wide}/{patches_high}")
    print(f"Working tensor shape: {working_tensor.shape}")

    #We will be taking 5 sets of grid meshes for segmentation, then having the 5 layers vote. 
    #The mode on each pixel will be the vote to the final result
    votes = torch.zeros((5, working_height, working_width))
    print(f"Vote Tensor Shape: {votes.shape}")


    #Now go through each grid mesh and gather votes
    #order: close, close_shift, mid, mid_shift, far
    labels = ["unlabeled","paved-area","dirt","grass","gravel","water","rocks","pool","vegetation","roof","wall","window","door","fence","fence-pole","person","dog","car","bicycle","tree","bald-tree","ar-marker","obstacle","conflicting"]
    ideal = [0,1,23,22,21,13,14,15,16,17,18]
    drivable = [2,3,4,6]
    colors = [[0, 0, 0],[128, 64, 128],[130, 76, 0],[0, 102, 0],[112, 103, 87],[28, 42, 168],[48, 41, 30],[0, 50, 89],[107, 142, 35],[70, 70, 70],[102, 102, 156],[254, 228, 12],[254, 148, 12],[190, 153, 153],[153, 153, 153],[255, 22, 96],[102, 51, 0],[9, 143, 150],[119, 11, 32],[51, 51, 0],[190, 250, 190],[112, 150, 146],[2, 135, 115],[255, 0, 0]]
    label_color = list(zip(labels,colors))
    """
    color_key = torch.zeros((1,480,480))
    print(label_color)
    for color in range(len(colors)):
        color_key[:,color*20:(color+1)*20, :] = torch.mul(torch.ones((1,20,480)),color)

    rgb_tensor = np.zeros((480, 480, 3))
    for y, row in enumerate(color_key.permute(0,2,1).squeeze(0)):
        for x, index in enumerate(row):
            rgb = colors[int(index)]
            rgb_tensor[y][x] = rgb
    to_pil_image = transforms.ToPILImage()
    segmented_image_pil = to_pil_image(rgb_tensor).rotate(-90, expand=True)
    segmented_image_pil.show()
    """
    #close
    for v in range(5):
        print(f"Vote: {v}")
        shift_amount = int(patch_size//5)*v
        padded_working_tensor = pad(working_tensor, (shift_amount, patch_size-shift_amount, shift_amount, patch_size-shift_amount), mode='constant', value = 0)
        #pad_pil = to_pil_image(padded_working_tensor)
        #pad_pil.show()
        aggregation = torch.zeros((working_height+patch_size, working_width+patch_size))
        y_count = patches_high
        x_count = patches_wide
        if v != 0:
            y_count += 1
            x_count += 1
        for y in range(y_count):
            for x in range(x_count):
                top = y*patch_size
                bottom = top+patch_size
                left = x*patch_size
                right = left+patch_size
                
                #print(f"Top/Left-Bottom/Right: {top}/{left}-{bottom}/{right}")

                section_tensor = padded_working_tensor[:, top:bottom, left:right]
                input_tensor = interpolate(section_tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
                #print(f"input: {input_tensor.shape}")
                segmented_tensor = segment_model(input_tensor).float().unsqueeze(0)
                #print(f"segmented: {segmented_tensor.shape}")
                segmented_resize_tensor = interpolate(segmented_tensor.unsqueeze(0), size=(patch_size, patch_size), mode='nearest').squeeze(0)
                #print(f"resize: {segmented_resize_tensor.shape}")
                aggregation[top:bottom, left:right] = segmented_resize_tensor
        votes[v] = aggregation[shift_amount:working_height+shift_amount, shift_amount:working_width+shift_amount]
        """
        rgb_tensor = np.zeros((working_width, working_height, 3))
        for y, row in enumerate(votes[v].unsqueeze(0).permute(0,2,1).squeeze(0)):
            for x, index in enumerate(row):
                rgb = colors[int(index)]
                rgb_tensor[y][x] = rgb
        to_pil_image = transforms.ToPILImage()
        segmented_image_pil = to_pil_image(rgb_tensor).rotate(-90, expand=True)
        segmented_image_pil.show()
        """

    majority_votes, _ = torch.mode(votes, dim=0)
    classes_tensor = majority_votes.unsqueeze(0).permute(0,2,1)
    classes_tensor = torch.flip(classes_tensor, dims=[-2]).squeeze(0)
    #print(classes_tensor.shape)
    #print(classes_tensor)


    ideality_tensor = np.zeros((working_width, working_height, 1))
    for y, row in enumerate(classes_tensor):
        for x, index in enumerate(row):
            output_index = 0 
            if int(index) in ideal:
                output_index = 2
            elif int (index) in drivable:
                output_index = 1
            ideality_tensor[y][x] = output_index



    segmented_image_pil = to_pil_image(ideality_tensor).rotate(-90, expand=True)
    depth_image_pil = to_pil_image(depth_map_tensor)

    adder = video_to_convert.split(".")[0]
    torch.save(ideality_tensor,f"segmented_{adder}.pt")
    torch.save(depth_map_tensor,f"depth_{adder}.pt")
    torch.save(working_tensor,f"colored_{adder}.pt")

filename = "corn-field-landscape"
is_vertical_motion = True
main(filename,is_vertical_motion)
Pathfinding(filename).main_loop()

"""
segmented_image_pil.show()
depth_image_pil.show()
pano_pil.show()
#segmented_image_pil.save("segmented.png")
#depth_image_pil.save("depth.png")
#pano_pil.save("colored.png")
adder = video_to_convert.split(".")[0]
torch.save(ideality_tensor,f"segmented_{adder}.pt")
torch.save(depth_map_tensor,f"depth_{adder}.pt")
torch.save(working_tensor,f"colored_{adder}.pt")
print(torch.max(votes))
print(torch.min(votes))
print(len(labels))

"""
         

