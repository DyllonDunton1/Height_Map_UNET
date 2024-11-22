import torch
from torchvision import transforms
from PIL import Image
import numpy as np

name = "suburb"
color_tensor = torch.load(f"colored_{name}.pt")
depth_tensor = torch.load(f"depth_{name}.pt")
class_tensor = torch.load(f"segmented_{name}.pt")

to_pil_image = transforms.ToPILImage()
color_pil = to_pil_image(color_tensor)
depth_pil = to_pil_image(depth_tensor)
class_pil = to_pil_image(class_tensor)

color_pil.show()
depth_pil.show()
class_pil.show()

print(np.unique(class_tensor))