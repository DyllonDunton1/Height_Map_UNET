import os
from torchvision import io, transforms, models
from torch.optim.lr_scheduler import StepLR
import torch
from torch import nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision.transforms.functional as F
import numpy as np
import csv
import warnings
warnings.filterwarnings("ignore")


class ResNet34SegmentationModel(nn.Module):
    def __init__(self, num_classes=24):
        super(ResNet34SegmentationModel, self).__init__()

        self.pre_encoder = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(3),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample to 1500x1000

            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(3),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample to 750x500

            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(3),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample to 375x250

            # pool to the correct size
            nn.AdaptiveMaxPool2d((224, 224))  # Final resize to 224x224)64
        )
        self.base = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        #print(self.base)
        #print('-------------------------------------------')
        self.encoder = nn.Sequential(*list(self.base.children())[:-2])
        #print(self.encoder)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 48, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1),
            # pool to the correct size
            nn.AdaptiveMaxPool2d((720, 1080))
        )

    def forward(self, input):
        #print("input",input.shape)
        pre_encoded = self.pre_encoder(input)
        #print("pre",pre_encoded.shape)
        encoded = self.encoder(pre_encoded)
        #print("encoded",encoded.shape)
        decoded = self.decoder(encoded)
        #print("decoded",decoded.shape)
        return decoded

def segment_model(image_tensor):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet34SegmentationModel().to(device)
    print(device)
    state_dict = torch.load('segmentation_model.pth', map_location=device)
    model.load_state_dict(state_dict)    
    
    img_transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])   
    img_tensor = img_transform(image_tensor).unsqueeze(0).to(device)

    model.eval()

    print(img_tensor.shape)
    output_vector = model(img_tensor)
    print(output_vector.shape)
    output_class_index = torch.argmax(output_vector,dim=1).squeeze(0)
    print(output_class_index.shape)
    #return output_class_index
    colors = [[0, 0, 0],[128, 64, 128],[130, 76, 0],[0, 102, 0],[112, 103, 87],[28, 42, 168],[48, 41, 30],[0, 50, 89],[107, 142, 35],[70, 70, 70],[102, 102, 156],[254, 228, 12],[254, 148, 12],[190, 153, 153],[153, 153, 153],[255, 22, 96],[102, 51, 0],[9, 143, 150],[119, 11, 32],[51, 51, 0],[190, 250, 190],[112, 150, 146],[2, 135, 115],[255, 0, 0]]

    #print(output_class_index[112,112])
    output_tensor = np.zeros((720,1080,3))
    for x, row in enumerate(output_class_index):
        for y, index in enumerate(row):
            rgb = colors[index]
            output_tensor[x][y] = rgb

    
    to_pil_image = transforms.ToPILImage()
    new_image = to_pil_image(output_tensor)

    return new_image
    

train_images_path = "data/train_imgs_large"
train_labels_path = "data/train_labels_index"
val_images_path = "data/val_imgs_large"
val_labels_path = "data/val_labels_index"
batch_size = 8


img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])  
image = img_transform(Image.open(f"{train_images_path}/001.jpg"))
segmented_image = segment_model(image)
segmented_image.show()
