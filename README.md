# Overview

This project consists of a pipelin of 4 parts: Stictching, HeightMapping, Segmentation, and path planning. All information about the project can be found in depth in 'Dunton-Walden-COS573-Final-Report.pdf'

# Stictching
With the use of opencv, we take an aerial video of drone footage, and stitch it into a large panorama

# HeightMapping
Using ZoeDepth, the panorama is converted into an intensity map where high intensity means a higher elevation

# Segmentation 
Using the 'Semantic Drone Dataset' on Kaggle, I was able to create a U-NET without skip connections and train it to ~92% accuracy which allowed for determining which surfaces were drivable

# Path Planning
Using pygame and A* (Completed by Sophie Walden), we are able to visualize the maps and plot a path on the rgb panorama using the height and segmentation information
