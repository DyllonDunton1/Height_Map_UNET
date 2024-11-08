from PIL import Image
import os


"""
This file serves to take all of the 4000x6000 photo and label into a 224x224 image for the RESNET input
"""

TRAIN_PATH_IMAGES = "/home/dunto/cos573/segmentation_trainer/Train_Images"
TRAIN_PATH_LABELS = "/home/dunto/cos573/segmentation_trainer/Train_Labels"
VAL_PATH_IMAGES = "/home/dunto/cos573/segmentation_trainer/Validation_Images"
VAL_PATH_LABELS = "/home/dunto/cos573/segmentation_trainer/Validation_Labels"
DATA_PATH = "/home/dunto/cos573/segmentation_trainer/data"


current_dir = VAL_PATH_LABELS
output_dir = "val_labels_small"
dir_list = sorted(os.listdir(current_dir))
print(f"Found {len(dir_list)} images")
for i, img in enumerate(dir_list):
    image = Image.open(f"{current_dir}/{img}")
    image_small = image.resize((256,256),Image.NEAREST)
    assert image_small.size == (256,256)
    output_path = f"{DATA_PATH}/{output_dir}/{img}"
    image_small.save(output_path)
    print(i)
    




#pretrained_net = models.resnet34(pretrained=True)

#print(pretrained_net)