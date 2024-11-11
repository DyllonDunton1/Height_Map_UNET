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

        self.base = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
        self.encoder = nn.Sequential(*list(self.base.children())[:-2])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, input):
        return self.decoder(self.encoder(input))

def segment_model(image_tensor):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet34SegmentationModel().to(device)
    print(device)
    state_dict = torch.load('segmentation_trainer/segmentation_model.pth', map_location=device)
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
    return output_class_index
    '''
    #print(output_class_index[112,112])
    output_tensor = np.zeros((224,224,3))
    for x, row in enumerate(output_class_index):
        for y, index in enumerate(row):
            rgb = colors[index]
            output_tensor[x][y] = rgb

    
    to_pil_image = transforms.ToPILImage()
    new_image = to_pil_image(output_tensor)

    return output_class_index
    '''

train_images_path = "data/train_imgs_small"
train_labels_path = "data/train_labels_index"
val_images_path = "data/val_imgs_small"
val_labels_path = "data/val_labels_index"
batch_size = 8



#segmented_image = segment_model(model, f"{train_images_path}/003.jpg", device)
#segmented_image.show()
