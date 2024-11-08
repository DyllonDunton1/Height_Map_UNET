import os
import torch
from torchvision import io, transforms, models
from torch import nn
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

MODEL_PATH = "/home/dunto/cos573/segmentation_trainer/segmentation_model.pth"
OUTPUT_PATH = "/home/dunto/cos573/segmentation_trainer/output_predictions"
VAL_PATH_IMAGES = "/home/dunto/cos573/segmentation_trainer/Validation_Images"
VAL_PATH_LABELS = "/home/dunto/cos573/segmentation_trainer/Validation_Labels"
LABELS_PATH = "/home/dunto/cos573/segmentation_trainer/class_dict_seg.csv"
VAL_AMOUNT = len(os.listdir(VAL_PATH_IMAGES))

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

def colormap_to_rgb(label_indices):
    rgb_image = np.zeros((label_indices.shape[0], label_indices.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(colormap):
        rgb_image[label_indices == i] = color
    return rgb_image

def rgb_to_class_indices(rgb_mask):
    label_indices = np.zeros((rgb_mask.shape[0], rgb_mask.shape[1]), dtype=np.uint8)
    for i, color in enumerate(colormap):
        mask = np.all(rgb_mask == color, axis=-1)
        label_indices[mask] = i
    return label_indices

def load_model():
    num_classes = len(classes)
    pretrained_net = models.resnet18(pretrained=False)
    net = nn.Sequential(*list(pretrained_net.children())[:-2])
    net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
    net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, padding=16, stride=32))

    net.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cuda')))
    net.eval()
    return net

def predict_image(net, image_path):
    img = io.read_image(image_path)
    
    transform = transforms.Compose([
        transforms.Resize((320, 480)),

    ])
    img = transform(img.float() / 255).unsqueeze(0)
    print(img.size())
    with torch.no_grad():
        output = net(img).squeeze(0).permute(1,2,0)
        predictions = torch.argmax(output, dim=-1)
        

    return predictions

def calculate_accuracy(predictions, ground_truth):
    predictions_flat = predictions.flatten()
    ground_truth_flat = ground_truth.flatten()
    return accuracy_score(ground_truth_flat, predictions_flat)

def resize_ground_truth(ground_truth, target_size):
    ground_truth_img = Image.fromarray(ground_truth)
    ground_truth_resized = ground_truth_img.resize(target_size, Image.NEAREST)
    return np.array(ground_truth_resized)

def save_prediction_as_rgb(predictions, file_name):
    rgb_image = colormap_to_rgb(predictions.numpy())
    Image.fromarray(rgb_image).save(file_name)

if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print("Loading model...")
    model = load_model()

    n = 5

    print("Running predictions and calculating accuracy...")
    accuracies = []
    for file_name in sorted(os.listdir(VAL_PATH_IMAGES))[:20]:
        img_path = os.path.join(VAL_PATH_IMAGES, file_name)
        label_path = os.path.join(VAL_PATH_LABELS, file_name.replace(".jpg",".png"))

        predictions = predict_image(model, img_path)
        print(predictions.size())
        ground_truth_rgb = np.array(Image.open(label_path))
        ground_truth = rgb_to_class_indices(ground_truth_rgb)

        ground_truth_resized = resize_ground_truth(ground_truth, predictions.shape[::-1])

        accuracy = calculate_accuracy(predictions.numpy(), ground_truth_resized)
        print(predictions.size(), ground_truth_resized.shape)
        accuracies.append(accuracy)

        output_file = os.path.join(OUTPUT_PATH, f"pred_{file_name}")
        save_prediction_as_rgb(predictions, output_file)
        print(f"Saved prediction for {file_name} at {output_file}, Accuracy: {accuracy:.4f}")


    avg_accuracy = np.mean(accuracies)
    print(f"Average accuracy: {avg_accuracy:.4f}")
