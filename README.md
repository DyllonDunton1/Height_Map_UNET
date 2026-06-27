# Rover Path Navigation via Aerial Imagery

A course-project repository for converting aerial drone footage into map products that can support rover path planning.

This repo contains the code used to build a perception-to-planning pipeline:

1. convert drone video into a stitched panorama
2. generate a relative height/depth map with ZoeDepth
3. run semantic segmentation over the panorama
4. reduce semantic classes into rover-friendly traversability categories
5. pass the generated map products to an A* pathfinding visualization module

The full project writeup is included as:

```text
Dunton-Walden-COS573-Final-Report.pdf
```

## Repository Contents

```text
Height_Map_UNET/
├── README.md
├── create_maps.py
├── Dunton-Walden-COS573-Final-Report.pdf
├── zoe_depth_map/
│   └── video_convert.py
├── segmentation_trainer/
│   ├── segment.py
│   └── segmentation_create.py
└── HeightmapVisualization/
    └── pathfinding visualization submodule
```

Some large local assets are intentionally excluded from Git. The `.gitignore` excludes local video data, ZoeDepth model files, segmentation training data, and trained model weights.

Expected local-only assets include:

```text
data/
zoe_depth_map/video_data/
zoe_depth_map/ZoeDepth/
segmentation_trainer/data/
segmentation_trainer/segmentation_model_91_683.pth
```

## Main Entry Point

The main project pipeline is in:

```text
create_maps.py
```

This script ties the repo together. It imports:

```python
from zoe_depth_map.video_convert import create_pano, create_depth_map_from_pano
from segmentation_trainer.segmentation_create import segment_model
from HeightmapVisualization.pathfinding.path_gen import Pathfinding
```

At a high level, `create_maps.py` does the following:

1. loads a drone video from `zoe_depth_map/video_data`
2. stitches the video frames into a panorama
3. generates a depth/height map from the panorama
4. divides the panorama into patches
5. runs the segmentation model on each patch
6. performs shifted-grid voting to reduce patch-boundary artifacts
7. converts semantic classes into simple traversability values
8. saves generated map tensors
9. launches the pathfinding visualization

The script currently uses:

```python
filename = "corn-field-landscape"
is_vertical_motion = True
main(filename, is_vertical_motion)
Pathfinding(filename).main_loop()
```

So the expected input video is:

```text
zoe_depth_map/video_data/corn-field-landscape.mp4
```

## Video to Panorama and Depth Map

The file:

```text
zoe_depth_map/video_convert.py
```

contains the helper functions for turning video into map inputs.

Important functions:

```python
convert_mp4_to_frames(vid_path, rot)
stitch_imgs(frames)
crop_img(stitched_np)
create_pano(vid_path, rot)
create_depth_map_from_pano(pano_image_pil)
```

The workflow is:

1. read an MP4 with OpenCV
2. optionally rotate frames
3. select a subset of frames
4. stitch them into a panorama with OpenCV's stitcher
5. crop away invalid black borders
6. run ZoeDepth on the panorama
7. normalize the depth output into an intensity-style tensor

The ZoeDepth model is expected to exist locally at:

```text
zoe_depth_map/ZoeDepth/
```

That directory is not tracked in Git.

## Segmentation Training

The file:

```text
segmentation_trainer/segment.py
```

contains the training code for the semantic segmentation model.

It defines:

```python
TrainingLogPoint
TrainingLogger
ResNet34SegmentationModel
SegmentationDataset
validation()
train_model()
```

The segmentation model uses a pretrained ResNet-34 backbone as an encoder and a transposed-convolution decoder to output 24 semantic classes.

The expected training data paths are:

```text
segmentation_trainer/data/train/imgs
segmentation_trainer/data/train/inds
segmentation_trainer/data/val/imgs
segmentation_trainer/data/val/inds
```

The training script saves a model as:

```text
segmentation_model.pth
```

Training logs are saved as CSV files such as:

```text
training_log_12.csv
```

## Segmentation Inference

The file:

```text
segmentation_trainer/segmentation_create.py
```

contains the inference helper used by the main pipeline.

It defines the same `ResNet34SegmentationModel` architecture and exposes:

```python
segment_model(image_tensor)
```

This function:

1. loads the trained segmentation model weights
2. normalizes the input image patch
3. runs the model in evaluation mode
4. returns the predicted class index for each pixel

The expected model file is:

```text
segmentation_trainer/segmentation_model.pth
```

## Patch-Based Panorama Segmentation

The full panorama is larger than the segmentation model input, so `create_maps.py` segments it in patches.

The current approach:

1. resizes the panorama to a working size divisible by the patch size
2. uses a patch size of `200`
3. runs segmentation over multiple shifted patch grids
4. stores predictions in a 5-layer vote tensor
5. takes the pixelwise mode across votes

This shifted-grid voting was used to reduce patch-boundary artifacts and make the final terrain layer more useful for path planning.

## Traversability Classes

After semantic segmentation, `create_maps.py` reduces raw class labels into simpler rover-focused terrain categories.

The code currently groups classes into:

```python
ideal = [0, 1, 23, 22, 21, 13, 14, 15, 16, 17, 18]
drivable = [2, 3, 4, 6]
```

The output map uses:

```text
0 = avoided / not preferred
1 = drivable
2 = ideal / preferred
```

This simplified map is saved and then used by the pathfinding visualization.

## Generated Outputs

For a given input video name, the pipeline saves tensors such as:

```text
segmented_<filename>.pt
depth_<filename>.pt
colored_<filename>.pt
```

For example, with:

```python
filename = "corn-field-landscape"
```

the expected generated outputs are:

```text
segmented_corn-field-landscape.pt
depth_corn-field-landscape.pt
colored_corn-field-landscape.pt
```

These files represent the simplified traversability map, depth/height map, and working RGB panorama tensor.

## Pathfinding Visualization

The final stage calls:

```python
Pathfinding(filename).main_loop()
```

from the `HeightmapVisualization` submodule.

That submodule handles the interactive pathfinding visualization and A* planning portion of the project. The perception side of this repository generates the map products that the pathfinder consumes.

## How to Run

This repo was built as a course project, so it is not packaged as a polished command-line tool. The main workflow is to edit the input filename at the bottom of `create_maps.py`, place the matching MP4 in `zoe_depth_map/video_data`, and run:

```bash
python3 create_maps.py
```

Required local setup includes:

- Python
- PyTorch
- torchvision
- OpenCV
- PIL / Pillow
- NumPy
- imutils
- ZoeDepth cloned or placed at `zoe_depth_map/ZoeDepth`
- trained segmentation weights placed at `segmentation_trainer/segmentation_model.pth`
- the `HeightmapVisualization` submodule available

## My Contribution

My work focused on the perception and terrain-map generation side of the project:

- drone video frame extraction and panorama stitching
- relative height/depth-map generation using ZoeDepth
- semantic segmentation training code
- segmentation inference helper
- patch-based panorama segmentation
- shifted-grid voting to reduce patch seams
- conversion of semantic labels into simple traversability categories
- integration between generated map products and the pathfinding visualization

The A* pathfinding visualization was completed with Sophie Walden.

## Technologies Used

- Python
- PyTorch
- torchvision
- OpenCV
- ZoeDepth
- ResNet-34 encoder / decoder segmentation model
- PIL / Pillow
- NumPy
- imutils
- A* path planning visualization
- Semantic Drone Dataset

## Project Status

This is an archived course project repository. It is useful as a snapshot of the working pipeline and the code structure used for the final project, but it is not currently organized as a reusable Python package.

The best files to inspect first are:

```text
create_maps.py
zoe_depth_map/video_convert.py
segmentation_trainer/segment.py
segmentation_trainer/segmentation_create.py
```
