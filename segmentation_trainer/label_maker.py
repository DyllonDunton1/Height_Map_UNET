from PIL import Image
import os
import numpy as np
from torchvision import io, transforms, models
import torch.nn.functional as F
from torchvision.transforms.functional import crop
import torch
import warnings
warnings.filterwarnings("ignore")

transform = transforms.Compose([
                transforms.CenterCrop(224),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
img_transform = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

labels = ["unlabeled","paved-area","dirt","grass","gravel","water","rocks","pool","vegetation","roof","wall","window","door","fence","fence-pole","person","dog","car","bicycle","tree","bald-tree","ar-marker","obstacle","conflicting"]
colors = [[0, 0, 0],[128, 64, 128],[130, 76, 0],[0, 102, 0],[112, 103, 87],[28, 42, 168],[48, 41, 30],[0, 50, 89],[107, 142, 35],[70, 70, 70],[102, 102, 156],[254, 228, 12],[254, 148, 12],[190, 153, 153],[153, 153, 153],[255, 22, 96],[102, 51, 0],[9, 143, 150],[119, 11, 32],[51, 51, 0],[190, 250, 190],[112, 150, 146],[2, 135, 115],[255, 0, 0]]
label_color = list(zip(labels,colors))


current_dir = "data/val/labs"
output_dir = "data/val/inds"
dir_list = sorted(os.listdir(current_dir))
print(len(dir_list))
if False:
    for img_path in dir_list:
        img = Image.open(f"{current_dir}/{img_path}")
        print(f"{current_dir}/{img_path}")
        img_array = np.array(img)
        print(img_array.shape)
        #assert img_array.shape == (224,3000,3)
        for x, row in enumerate(img_array):
            for y, pixel in enumerate(row):
                assert list(pixel) in colors

if False:
    for img_path in dir_list:
        img = Image.open(f"{current_dir}/{img_path}")
        img.save(f"{current_dir}/{img_path.replace('.jpg','_con.jpg')}")
if True:
    for i, img_path in enumerate(dir_list):
        #if i <= 8600:
        #    continue
        img = Image.open(f"{current_dir}/{img_path}")
        img_array = np.array(img)
        #print(img_array.shape)
        #print(img_array)
        #print(img_array.max(), img_array.min())
        
        output_img = np.zeros((224,224,24))
        for x, row in enumerate(img_array):
            for y, pixel in enumerate(row):
                #print(list(pixel))
                index = colors.index(list(pixel))
                output_img[x][y] = np.eye(24)[index]
        
        out_path = f"{output_dir}/{img_path.replace('_con.jpg','')}.pt"
        #print(out_path)
        output_tensor = torch.from_numpy(output_img.transpose(2, 0, 1)).to(torch.uint8)
        #print(output_tensor.shape)
        torch.save(output_tensor, out_path)
        print(i)

'''
img_dir = "data/train_imgs_small"
mask_dir = "data/train_labels_index"
img_tensor = img_transform(Image.open(f"{img_dir}/000.jpg"))
mask_tensor = torch.load(f"{mask_dir}/000.pt")
print(img_tensor.shape, mask_tensor.shape)


top = int((256-224)*np.random.rand()//1)
left = int((256-224)*np.random.rand()//1)

print(top,left)
width = 224


img_crop = img_tensor[:, top:top+width, left:left+width]
mask_crop = mask_tensor[:, top:top+width, left:left+width]
print(img_crop.shape, mask_crop.shape)

# Random Vertical Flip
vert_chance = np.random.rand()
hori_chance = np.random.rand()
print(vert_chance, hori_chance)
if (vert_chance > 0.5):
    #do vertical flip
    print("VERT")
    img_vert = torch.flip(img_crop, dims=[1])
else:
    img_vert = img_crop

# Random Horizontal Flip
if (hori_chance > 0.5):
    #do horizontal flip
    print("HOR")
    img_hori = torch.flip(img_vert, dims=[2])
else:
    img_hori = img_vert

rot_count = int(4*np.random.rand())
print(rot_count)
img_rot90 = torch.rot90(img_crop, rot_count, [1,2])
mask_rot90 = torch.rot90(mask_crop, rot_count, [1,2])
print(img_rot90.shape, mask_rot90.shape)

to_pil_image = transforms.ToPILImage()
new_image = to_pil_image(img_rot90)

new_image.show()





'''