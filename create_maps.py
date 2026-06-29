import os
from collections.abc import Callable

from torchvision import io, transforms, models
from torch.optim.lr_scheduler import StepLR
import torch
from torch import nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision.transforms.functional as F
from torch.nn.functional import interpolate, pad
import numpy as np
import csv
from HeightmapVisualization.pathfinding.path_gen import Pathfinding

from zoe_depth_map.video_convert import create_pano, create_depth_map_from_pano
from segmentation_trainer.segmentation_create import segment_model

import warnings
warnings.filterwarnings("ignore")


def segment_with_shifted_patch_voting(
    image_tensor: torch.Tensor,
    segment_fn: Callable[[torch.Tensor], torch.Tensor],
    patch_size: int = 200,
    num_shifts: int = 5,
    model_input_size: int = 224,
) -> torch.Tensor:
    """
    Segment an image using multiple shifted patch grids and majority voting.

    Large panoramic images can produce patch-boundary artifacts when each patch is
    segmented independently. This function reduces those artifacts by running the
    segmentation model over several grids with different offsets, then taking a
    per-pixel majority vote across the shifted predictions.

    Args:
        image_tensor: Input image tensor in CHW format.
        segment_fn: Function that accepts a CHW image patch and returns a 2D
            class-index segmentation map.
        patch_size: Spatial size of each patch in the stitched output.
        num_shifts: Number of shifted grids used for voting.
        model_input_size: Spatial size expected by the segmentation model.

    Returns:
        A 2D tensor of class indices with shape [working_height, working_width].
    """
    if image_tensor.ndim != 3:
        raise ValueError(
            f"Expected image_tensor in CHW format, got shape {tuple(image_tensor.shape)}"
        )

    _, image_height, image_width = image_tensor.shape

    patches_wide = image_width // patch_size
    patches_high = image_height // patch_size

    if patches_wide == 0 or patches_high == 0:
        raise ValueError(
            "Image must be at least one patch wide and one patch high. "
            f"Got image size {image_height}x{image_width} with patch_size={patch_size}."
        )

    working_width = patches_wide * patch_size
    working_height = patches_high * patch_size

    # Align the image size so the patch grid divides evenly.
    working_tensor = interpolate(
        image_tensor.unsqueeze(0),
        size=(working_height, working_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    votes = torch.empty(
        (num_shifts, working_height, working_width),
        dtype=torch.long,
        device=working_tensor.device,
    )

    shift_step = patch_size // num_shifts

    # Run segmentation over several offset grids to make the output less dependent on one tiling.
    for shift_index in range(num_shifts):
        shift_amount = shift_step * shift_index

        # Pad before slicing so shifted grids still cover the image edges.
        padded_tensor = pad(
            working_tensor,
            (
                shift_amount,
                patch_size - shift_amount,
                shift_amount,
                patch_size - shift_amount,
            ),
            mode="constant",
            value=0,
        )

        aggregation = torch.empty(
            (working_height + patch_size, working_width + patch_size),
            dtype=torch.long,
            device=working_tensor.device,
        )

        y_count = patches_high + int(shift_index != 0)
        x_count = patches_wide + int(shift_index != 0)

        for y_index in range(y_count):
            for x_index in range(x_count):
                top = y_index * patch_size
                bottom = top + patch_size
                left = x_index * patch_size
                right = left + patch_size

                patch_tensor = padded_tensor[:, top:bottom, left:right]

                model_input = interpolate(
                    patch_tensor.unsqueeze(0),
                    size=(model_input_size, model_input_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

                segmented_patch = segment_fn(model_input).float().unsqueeze(0)

                resized_segmentation = interpolate(
                    segmented_patch.unsqueeze(0),
                    size=(patch_size, patch_size),
                    mode="nearest",
                ).squeeze().long()

                aggregation[top:bottom, left:right] = resized_segmentation

        # Crop each shifted result back to the shared working image region.
        votes[shift_index] = aggregation[
            shift_amount : working_height + shift_amount,
            shift_amount : working_width + shift_amount,
        ]

    # Combine shifted predictions with a per-pixel majority vote.
    majority_votes, _ = torch.mode(votes, dim=0)
    return majority_votes


def main(filename, rot):

    data_path = "zoe_depth_map/video_data"
    video_to_convert = f"{filename}.mp4"
    pano_pil = create_pano(f"{data_path}/{video_to_convert}", rot)
    depth_map_tensor = create_depth_map_from_pano(pano_pil)

    #Now that we have the depth tensor, we need to get split up the panorama and run the segmentation model
    #First find the height and width of the pano
    to_tensor = transforms.ToTensor()
    to_pil_image = transforms.ToPILImage()
    pano_tensor = to_tensor(pano_pil)
    print(f"Depth Map Shape: {depth_map_tensor.shape}")
    print(f"RGB Original shape: {pano_tensor.shape}")

    majority_votes = segment_with_shifted_patch_voting(
        image_tensor=pano_tensor,
        segment_fn=segment_model,
        patch_size=200,
        num_shifts=5,
        model_input_size=224,
    )

    classes_tensor = majority_votes.unsqueeze(0).permute(0, 2, 1)
    classes_tensor = torch.flip(classes_tensor, dims=[-2]).squeeze(0)
    #print(classes_tensor.shape)
    #print(classes_tensor)

    labels = ["unlabeled","paved-area","dirt","grass","gravel","water","rocks","pool","vegetation","roof","wall","window","door","fence","fence-pole","person","dog","car","bicycle","tree","bald-tree","ar-marker","obstacle","conflicting"]
    ideal = [0,1,23,22,21,13,14,15,16,17,18]
    drivable = [2,3,4,6]
    colors = [[0, 0, 0],[128, 64, 128],[130, 76, 0],[0, 102, 0],[112, 103, 87],[28, 42, 168],[48, 41, 30],[0, 50, 89],[107, 142, 35],[70, 70, 70],[102, 102, 156],[254, 228, 12],[254, 148, 12],[190, 153, 153],[153, 153, 153],[255, 22, 96],[102, 51, 0],[9, 143, 150],[119, 11, 32],[51, 51, 0],[190, 250, 190],[112, 150, 146],[2, 135, 115],[255, 0, 0]]
    label_color = list(zip(labels,colors))

    working_width, working_height = classes_tensor.shape
    ideality_tensor = np.zeros((working_width, working_height, 1))
    for y, row in enumerate(classes_tensor):
        for x, index in enumerate(row):
            output_index = 0 
            if int(index) in ideal:
                output_index = 2
            elif int (index) in drivable:
                output_index = 1
            ideality_tensor[y][x] = output_index



    segmented_image_pil = to_pil_image(ideality_tensor).rotate(-90, expand=True)
    depth_image_pil = to_pil_image(depth_map_tensor)

    adder = video_to_convert.split(".")[0]
    torch.save(ideality_tensor,f"segmented_{adder}.pt")
    torch.save(depth_map_tensor,f"depth_{adder}.pt")
    torch.save(pano_tensor,f"colored_{adder}.pt")


if __name__ == "__main__":
    filename = "corn-field-landscape"
    is_vertical_motion = True

    main(filename, is_vertical_motion)
    Pathfinding(filename).main_loop()
