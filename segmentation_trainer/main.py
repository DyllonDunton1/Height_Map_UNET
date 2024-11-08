import os
from torchvision import io, transforms, models
from torch.optim.lr_scheduler import StepLR
from termcolor import colored
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import csv
import gc

import warnings
warnings.filterwarnings("ignore")

# THINGS YOU COULD MESS WITH SINCE YOU HAVE A GOOD COMPUTER
num_workers = 5
num_epochs, lr, wd = 50, 0.001, 0
batch_size = 5
crop_size = (480, 480)

# MAKE SURE THESE MATCH UP WITH YOUR PATH TO DATASET
TRAIN_PATH_IMAGES = "/home/dunto/cos573/segmentation_trainer/Train_Images"
TRAIN_PATH_LABELS = "/home/dunto/cos573/segmentation_trainer/Train_Labels"
VAL_PATH_IMAGES = "/home/dunto/cos573/segmentation_trainer/Validation_Images"
VAL_PATH_LABELS = "/home/dunto/cos573/segmentation_trainer/Validation_Labels"
LABELS_PATH = "/home/dunto/cos573/segmentation_trainer/class_dict_seg.csv"
TRAIN_AMOUNT = len(os.listdir(TRAIN_PATH_IMAGES))
VAL_AMOUNT = len(os.listdir(VAL_PATH_IMAGES))

print(f"Train Amount: {TRAIN_AMOUNT}")
print(f"Validation Amount: {VAL_AMOUNT}")

def generate_colormap():
    colormap = []
    classes = []

    with open(LABELS_PATH) as input_file:
        for line in input_file.readlines()[1:]:
            line = line.strip()

            data = line.split(", ")
            classes.append(data[0])
            colormap.append(list(map(int, data[1:])))


    return colormap, classes

colormap, classes = generate_colormap()



# Grab label for any region of rgb values
def colormap_to_label():
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, cmap in enumerate(colormap):
        colormap2label[(cmap[0] * 256 + cmap[1]) * 256 + cmap[2]] = i

    return colormap2label

def label_indices(colormap, colormap2label):
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
            + colormap[:, :, 2])

    return colormap2label[idx]


# Cropping to maintain proper resolution
def rand_crop(feature, label, height, width):
    rect = transforms.RandomCrop.get_params(
        feature, (height, width))

    feature = transforms.functional.crop(feature, *rect)
    label = transforms.functional.crop(label, *rect)
    return feature, label

class SegDataset(torch.utils.data.Dataset):
    def __init__(self, crop_size, use_type="train", transform=None, amount=0):
        #self.transform = transform
        self.crop_size = crop_size
        self.use_type = use_type
        self.amount = amount
        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.colormap2label = colormap_to_label()

    def normalize_image(self, img):
        return self.transform(img.float() / 255)
    
    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1]
        )]

    def __getitem__(self, idx):
        if self.use_type == "train":
            image_path = TRAIN_PATH_IMAGES
            label_path = TRAIN_PATH_LABELS
        else:
            image_path = VAL_PATH_IMAGES
            label_path = VAL_PATH_LABELS
        feature = io.read_image(f"{image_path}/{sorted(os.listdir(image_path))[idx]}")
        label = io.read_image(f"{label_path}/{sorted(os.listdir(label_path))[idx]}")

        feature = self.normalize_image(feature.float() / 255.0)
        label = label.float()

        feature, label = rand_crop(feature, label, *self.crop_size)
        print(feature.size(), label.size())
        

        #if self.transform:
        #    feature = self.transform(feature)
        #    label = self.transform(label)


        return feature, label_indices(label, self.colormap2label)

  
    def __len__(self):
        return self.amount




# Need to check these and not just use textbook implementation
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight

def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='mean')

# Define your data augmentation transforms
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomVerticalFlip(),    # Randomly flip the image vertically
    transforms.RandomRotation(180),      # Randomly rotate the image by 180 degrees
])

print ("Loading Data!!")
train = SegDataset(crop_size, transform=data_transforms, use_type="train", amount=TRAIN_AMOUNT)
test = SegDataset(crop_size, transform=data_transforms, use_type="test", amount=VAL_AMOUNT)
train_iter = torch.utils.data.DataLoader(train, batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=False)
test_iter = torch.utils.data.DataLoader(test, batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=False)

print ("Creating Model")
pretrained_net = models.resnet18(pretrained=True)
net = nn.Sequential(*list(pretrained_net.children())[:-2])

X = torch.rand(size=(1, 3, crop_size[0], crop_size[1]))
num_classes = len(classes)
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                kernel_size=64, padding=16, stride=32))

W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

print("Attempting to Train!")


net = net.to(device)
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
scheduler = StepLR(trainer, step_size=5, gamma=0.1)

train_losses = []
test_losses = []
learning = []

print(torch.cuda.memory_summary(device='cuda', abbreviated=False))
for epoch in range(num_epochs):
    print("------------------------------------")
    print()
    print(f'Epoch {epoch+1}/{num_epochs}')
    print(f"Learning Rate: {scheduler.get_lr()}")
    print()
    
    train_loss = []
    test_loss = []
    
    #Get train loss
    for i, (X_batch, y_batch) in enumerate(train_iter):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        trainer.zero_grad()
        outputs = net(X_batch)
        l = loss(outputs, y_batch)
        l.backward()
        trainer.step()
        train_loss.append(l.item())
        print(train_loss[-1])
    avg_train_loss = sum(train_loss)/len(train_loss)
    print(f'Train Loss: {avg_train_loss}')
    if epoch > 0:
        print(f"Training loss change: {avg_train_loss - train_losses[-1]}")
    print(f'Trained {i+1} batches of {batch_size} images | Total: {batch_size*(i+1)}')
    print()

    #Get Validation Loss
    for i, (X_batch, y_batch) in enumerate(test_iter):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = net(X_batch)
        l = loss(outputs, y_batch)
        test_loss.append(l.item())
    avg_test_loss = sum(test_loss)/len(test_loss)
    print(f'Validation Loss: {avg_test_loss}')
    if epoch > 0:
        print(f"Validation loss change: {avg_test_loss - test_losses[-1]}")
    print(f'Trained {i+1} batches of {batch_size} images | Total: {batch_size*(i+1)}')
    print()

    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)
    learning.append(lr)
    if epoch >= 10:
        scheduler.step()

torch.save(net.state_dict(), 'segmentation_model.pth')
output = [train_losses,test_losses,learning]

with open(f"/home/dunto/cos573/segmentation_trainer/loss_curves.csv", "w+") as f:
    writer = csv.writer(f)
    writer.writerows(output)
