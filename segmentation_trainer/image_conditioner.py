from PIL import Image
import os
import numpy as np


"""
This file serves to take all of the 4000x6000 photo and label into a 224x224 image for the RESNET input
"""

labels = ["unlabeled","paved-area","dirt","grass","gravel","water","rocks","pool","vegetation","roof","wall","window","door","fence","fence-pole","person","dog","car","bicycle","tree","bald-tree","ar-marker","obstacle","conflicting"]
colors = [[0, 0, 0],[128, 64, 128],[130, 76, 0],[0, 102, 0],[112, 103, 87],[28, 42, 168],[48, 41, 30],[0, 50, 89],[107, 142, 35],[70, 70, 70],[102, 102, 156],[254, 228, 12],[254, 148, 12],[190, 153, 153],[153, 153, 153],[255, 22, 96],[102, 51, 0],[9, 143, 150],[119, 11, 32],[51, 51, 0],[190, 250, 190],[112, 150, 146],[2, 135, 115],[255, 0, 0]]


current_dir = "data/train_imgs_large"
output_dir = "data/train/imgs"
dir_list = sorted(os.listdir(current_dir))
print(f"Found {len(dir_list)} images")
#[12,8], [9,6]
#make 31 photos from each photo!
resolutions = [[6,4], [3,2], [1,1]]
photo_scale = 224
for i, img in enumerate(dir_list):
    image = Image.open(f"{current_dir}/{img}")
    #print(image.size)
    
    for res in resolutions:
        image_scaled = np.array(image.resize((photo_scale*res[0],photo_scale*res[1]),Image.NEAREST))
        print(image_scaled.shape)
        
        for w in range(res[0]):
            for h in range(res[1]):
                out_np = image_scaled[w*photo_scale:(w+1)*photo_scale, h*photo_scale:(h+1)*photo_scale, :]
                
                out_img = Image.fromarray(out_np)
                print(res, w, h, out_np.shape)
                #print(out_img.size,f"{output_dir}/{img}_{photo_scale*res[0]}x{photo_scale*res[1]}_{res[1]*w + h}")
                out_img.save(f"{output_dir}/{img.replace('_conditioned','')}_{photo_scale*res[0]}x{photo_scale*res[1]}_{res[1]*w + h}.jpg")
    
    print(i)
    




#pretrained_net = models.resnet34(pretrained=True)

#print(pretrained_net)