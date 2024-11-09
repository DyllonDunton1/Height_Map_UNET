import os
from torchvision import io, transforms, models
from torch.optim.lr_scheduler import StepLR
import torch
from torch import nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision.transforms.functional as F
import warnings
warnings.filterwarnings("ignore")

#Need to fix transforms to synchronize
#Need to add in LR scheduler
#Need to add in Validation
#Need to save train loss and acc

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


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(os.listdir(image_dir))  # Assuming files are matched by name
        self.mask_files = sorted(os.listdir(mask_dir))
        self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        self.general_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
                transforms.RandomVerticalFlip(),    # Randomly flip the image vertically
                transforms.RandomRotation(180),      # Randomly rotate the image by 180 degrees
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path)

        # Load mask as tensor
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        mask = torch.load(mask_path)

        # Transform
        image = self.general_transform(self.img_transform(image))
        mask = self.general_transform(mask)

        return image, mask

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()  # Set model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            # Move data to device (e.g., GPU or CPU)
            inputs = inputs.to(device)
            labels = labels.to(device)
            #print(inputs.shape, labels.shape)
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            #print(outputs.shape)
            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # Print every 10 batches
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}")
                running_loss = 0.0

    print('Training complete')
    torch.save(model.state_dict(), 'segmentation_model.pth')





train_images_path = "data/train_imgs_small"
train_labels_path = "data/train_labels_index"
val_images_path = "data/val_imgs_small"
val_labels_path = "data/val_labels_index"
batch_size = 8

model = ResNet34SegmentationModel()
train_dataset = SegmentationDataset(train_images_path, train_labels_path)
val_dataset = SegmentationDataset(val_images_path, val_labels_path)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

train_model(model, train_loader, criterion, optimizer, device, num_epochs=10)