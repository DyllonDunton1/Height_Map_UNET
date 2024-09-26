import torch
from PIL import Image
import glob
import numpy as np
import cv2
import imutils
from ZoeDepth.zoedepth.utils.misc import colorize

resize = False
re_x = 1280
re_y = 720

#video_to_convert = "road-flying-above"
video_to_convert = "suburb"
data_path = "/home/dunto/Height_Map_UNET/data/"
city_path = f"{data_path}cityscapes/"
mars_path = f"{data_path}marsscapes/"
vid_tests_path = f"{data_path}videos_tests/"
output_path = f"{data_path}output/"


def glob_files_in_dir(dir_path):
    file_names = sorted(glob.glob(f"{dir_path}*.png"))
    return file_names


def convert_to_depth(img, model):
    
    
    depth_numpy = model.infer_pil(img)  # as numpy
    print(f"min: {np.min(depth_numpy)}, max: {np.max(depth_numpy)}")
    colored = colorize(depth_numpy)
    depth_pil = Image.fromarray(colored)

    return depth_pil, depth_numpy


def convert_mp4_to_frames(vid_path):
    print(vid_path)
    cap = cv2.VideoCapture(vid_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frames.append(image)
    cap.release()
    return frames

def stitch_imgs(frames):
    stitcher = cv2.Stitcher_create()
    ret, stitched = stitcher.stitch(frames)
    return ret, stitched

# Zoe_N
repo = "isl-org/ZoeDepth"
# Zoe_N
model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)

#file_names = glob_files_in_dir(city_path)
#depth_img, depth_np = convert_to_depth(file_names[0], zoe)

#print(f"Shape: {depth_np.shape}")
#print(depth_np)

#depth_img.save(f"{output_path}depth_test.png")

frames = convert_mp4_to_frames(f"{vid_tests_path}{video_to_convert}.mp4")
print(f"Frame size: {frames[0].shape}")
print(len(frames))
trimmed_frames = []
for i, frame in enumerate(frames):
    if i%(len(frames)//6) == 0:
        if resize:
            trimmed_frames.append(cv2.resize(frame, (re_x, re_y)))
        else:
            trimmed_frames.append(frame)
        print(i)
print(f"Frame size: {trimmed_frames[0].shape}")
print(len(trimmed_frames))

ret, stitched_np = stitch_imgs(trimmed_frames)
if ret:
    print("Bad Stitching")


print(f"Shape: {stitched_np.shape}")


stitched_image = cv2.copyMakeBorder(stitched_np, 10, 10, 10, 10,
			cv2.BORDER_CONSTANT, (0, 0, 0))
gray = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)
mask = np.zeros(thresh.shape, dtype="uint8")
(x, y, w, h) = cv2.boundingRect(c)
cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
minRect = mask.copy()
sub = mask.copy()
while cv2.countNonZero(sub) > 0:
    minRect = cv2.erode(minRect, None)
    sub = cv2.subtract(minRect, thresh)

cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)
(x, y, w, h) = cv2.boundingRect(c)

stitched_image = stitched_image[y:y + h, x:x + w]



stitched_image = Image.fromarray(stitched_image)
stitched_image.save(f"{output_path}test_stitch.png")

depth_stitch_pil, depth_stitch_np = convert_to_depth(stitched_image, zoe)
print(depth_stitch_np)
depth_stitch_pil.save(f"{output_path}depth_test_stitch.png")




