from PIL import Image
import os
import numpy as np
from torchvision import io, transforms, models
import torch.nn.functional as F
import torch
import warnings
warnings.filterwarnings("ignore")

transform = transforms.Compose([
                transforms.CenterCrop(224),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

labels = ["unlabeled","paved-area","dirt","grass","gravel","water","rocks","pool","vegetation","roof","wall","window","door","fence","fence-pole","person","dog","car","bicycle","tree","bald-tree","ar-marker","obstacle","conflicting"]
colors = [[0, 0, 0],[128, 64, 128],[130, 76, 0],[0, 102, 0],[112, 103, 87],[28, 42, 168],[48, 41, 30],[0, 50, 89],[107, 142, 35],[70, 70, 70],[102, 102, 156],[254, 228, 12],[254, 148, 12],[190, 153, 153],[153, 153, 153],[255, 22, 96],[102, 51, 0],[9, 143, 150],[119, 11, 32],[51, 51, 0],[190, 250, 190],[112, 150, 146],[2, 135, 115],[255, 0, 0]]
label_color = list(zip(labels,colors))


current_dir = "data/train_labels_small"
output_dir = "data/train_labels_index"
dir_list = sorted(os.listdir(current_dir))

if True:
    for img_path in dir_list:
        img = Image.open(f"{current_dir}/{img_path}")
        img_array = np.array(img)
        #print(img_array.shape)
        
        output_img = np.zeros((256,256,24))
        for x, row in enumerate(img_array):
            for y, pixel in enumerate(row):
                #print(list(pixel))
                index = colors.index(list(pixel))
                output_img[x][y] = np.eye(24)[index]
        
        out_path = f"{output_dir}/{img_path.split('.')[0]}.pt"
        print(out_path)
        output_tensor = torch.from_numpy(output_img.transpose(2, 0, 1))
        print(output_tensor.shape)
        torch.save(output_tensor, out_path)

#img_tensor = torch.load(f"{output_dir}/540.pt")
#print(img_tensor.shape)
#ew_tensor = transform(img_tensor)
#print(new_tensor.shape)





