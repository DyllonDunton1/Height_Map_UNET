from PIL import Image
import os


"""
This file serves to take all of the 4000x6000 photo and label into a 224x224 image for the RESNET input
"""

TRAIN_PATH_IMAGES = "/home/dunto/cos573/Height_Map_UNET/segmentation_trainer/data/Train_Images"
TRAIN_PATH_LABELS = "/home/dunto/cos573/segmentation_trainer/data/Train_Labels"
VAL_PATH_IMAGES = "/home/dunto/cos573/segmentation_trainer/data/Validation_Images"
VAL_PATH_LABELS = "/home/dunto/cos573/segmentation_trainer/data/Validation_Labels"
DATA_PATH = "/home/dunto/cos573/segmentation_trainer/data"


dir = "val_index_large"
current_dir = f"/home/dunto/Height_Map_UNET/segmentation_trainer/data/{dir}"
dir_list = sorted(os.listdir(current_dir))
print(f"Found {len(dir_list)} images")
for i, img in enumerate(dir_list):
    image = Image.open(f"{current_dir}/{img}")
    print(image.size)
    image_small = image.resize((3000,2000),Image.NEAREST)
    assert image_small.size == (3000,2000)
    output_path = f"{current_dir}/{img}".replace(".jpg","_conditioned.jpg")
    image_small.save(output_path)
    print(i)
    




#pretrained_net = models.resnet34(pretrained=True)

#print(pretrained_net)