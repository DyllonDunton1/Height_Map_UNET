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


class TrainingLogPoint():
    def __init__(self,epoch,lr,train_loss,val_acc):
        self.epoch = epoch
        self.lr = lr
        self.train_loss = train_loss
        self.val_acc = val_acc
    
    def convert_csv_line(self):
        return [self.epoch,self.lr,self.train_loss,self.val_acc]

class TrainingLogger():
    def __init__(self, save_file):
        self.save_file = save_file
        self.log = [["Epoch", "Learning Rate", "Training Loss", "Validation Accuracy"]]
    
    def add_measurement(self, log):
        self.log.append(log.convert_csv_line())
    
    def save(self):
        print(f"Saving Log File @ {self.save_file}")
        with open(self.save_file, "w+") as save_csv:
            csvWriter = csv.writer(save_csv,delimiter=',')
            csvWriter.writerows(self.log)

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
        self.start_width = 256
        self.end_width = 224
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(os.listdir(image_dir))  # Assuming files are matched by name
        self.mask_files = sorted(os.listdir(mask_dir))
        self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])     
        

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = self.img_transform(Image.open(image_path))

        # Load mask as tensor
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        mask = torch.load(mask_path)

        # Random crop Transform
        top = int((self.start_width-self.end_width)*np.random.rand()//1)
        left = int((self.start_width-self.end_width)*np.random.rand()//1)
        image = image[:, top:top+self.end_width, left:left+self.end_width]
        mask = mask[:, top:top+self.end_width, left:left+self.end_width]

        # Random Vertical Flip
        if (np.random.rand() > 0.5):
            #do vertical flip
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[1])

        # Random Horizontal Flip
        if (np.random.rand() > 0.5):
            #do horizontal flip
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])

        # Random rot90
        rot_count = int(4*np.random.rand())
        image = torch.rot90(image, rot_count, [1,2])
        mask = torch.rot90(mask, rot_count, [1,2])

        return image, mask

def validation(model, val_loader, device):
    model.eval() # Set model to validation mode

    with torch.no_grad():
        #only one batch since batch_size is full batch
        for i, (inputs, labels) in enumerate(val_loader):
            # Move data to device (e.g., GPU or CPU)
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            labels_classified = torch.argmax(labels,dim=1)
            outputs_classified = torch.argmax(outputs,dim=1)
            #print(labels.shape, outputs.shape)
            #print(labels_classified.shape, outputs_classified.shape)

            accuracy_mask = torch.where(labels_classified == outputs_classified, 1.0, 0.0)
            accuracy = 100*(torch.sum(accuracy_mask) / torch.numel(accuracy_mask))
            
    model.train()
    return accuracy


def train_model(model, train_loader, criterion, optimizer, device, lr_scheduler, train_log, num_epochs=10):
    model.train()  # Set model to training mode

    lr_steps_taken = 0
    accuracy_lr_update_thresholds = [80, 90, 95, 98, 99, 99.5, 99.8]
    epochs_since_last_step = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        print("----------------------------------------------")
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Learning Rate: {lr_scheduler.get_lr()}")
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
        
        print(f"Train Loss: {running_loss / 20:.4f}")
        
        val_accuracy = validation(model, val_loader, device)
        print(f"Validation Accuracy: {val_accuracy:.4f}%")
        
        log_point = TrainingLogPoint(epoch+1, *scheduler.get_lr(), running_loss/20, str(val_accuracy.item()))
        train_log.add_measurement(log_point)

        running_loss = 0.0

        #Check to see if we should update LR or stop
        epochs_since_last_step += 1
        if val_accuracy >= accuracy_lr_update_thresholds[-1]:
            print("Accuracy over 99.8%. Ending training")
            break
        if lr_steps_taken < accuracy_lr_update_thresholds[-1] and (val_accuracy >= accuracy_lr_update_thresholds[lr_steps_taken] or epochs_since_last_step > 100):
            epochs_since_last_step = 0
            lr_steps_taken += 1
            lr_scheduler.step()
        

    print('-----------------------------------')
    print('Training complete')
    torch.save(model.state_dict(), 'segmentation_model.pth')
    print('Model Saved @ segmentation_model.pth')


train_images_path = "data/train_imgs_small"
train_labels_path = "data/train_labels_index"
val_images_path = "data/val_imgs_small"
val_labels_path = "data/val_labels_index"
batch_size = 18

model = ResNet34SegmentationModel()
train_dataset = SegmentationDataset(train_images_path, train_labels_path)
val_dataset = SegmentationDataset(val_images_path, val_labels_path)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=len(os.listdir(val_images_path)), shuffle=True)

train_log = TrainingLogger("training_log.csv")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

train_model(model, train_loader, criterion, optimizer, device, scheduler, train_log, num_epochs=120)
train_log.save()
