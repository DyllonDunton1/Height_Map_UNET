import torch
from PIL import Image
import glob
import numpy as np
import cv2
import imutils
#from ZoeDepth.zoedepth.utils.misc import colorize


import warnings
warnings.filterwarnings("ignore")



#video_to_convert = "road-flying-above"
video_to_convert = "suburb.mp4"
data_path = "video_data"

output_path = f"{data_path}/output/"



def convert_to_depth_tensor(img, model):
    
    depth_numpy = model.infer_pil(img)  # as numpy
    #colored = colorize(depth_numpy)
    #depth_pil = Image.fromarray(colored)

    #normalize to [0,255]
    depth_tensor = torch.from_numpy(depth_numpy)
    min = torch.min(depth_tensor)
    max = torch.max(depth_tensor)

    depth_tensor_subtracted = torch.sub(depth_tensor, min)
    depth_tensor_scaled = torch.mul(depth_tensor_subtracted, -255.0/(max-min))
    depth_tensor_flipped = torch.add(depth_tensor_scaled, 255)

    return depth_tensor_flipped.int()


def convert_mp4_to_frames(vid_path, rot):
    print(vid_path)
    cap = cv2.VideoCapture(vid_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if rot:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        frames.append(image)
    cap.release()
    return frames

def stitch_imgs(frames):
    stitcher = cv2.Stitcher_create()
    ret, stitched = stitcher.stitch(frames)
    return ret, stitched

def crop_img(stitched_np):
    stitched_image = cv2.copyMakeBorder(stitched_np, 10, 10, 10, 10,
                cv2.BORDER_CONSTANT, (0, 0, 0))
    gray = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
    #print(cnts)
    cnts = imutils.grab_contours(cnts)
    #print(cnts)
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

    return stitched_image

def create_pano(vid_path, rot):
    frames = convert_mp4_to_frames(vid_path, rot)
    trimmed_frames = []
    for i, frame in enumerate(frames):
        if i%(len(frames)//7) == 0:
            trimmed_frames.append(frame)

    ret, stitched_np = stitch_imgs(trimmed_frames)
    #print(stitched_np.shape)
    #img_pil = Image.fromarray(stitched_np)
    #img_pil.show()
    if ret:
        print("Bad Stitching")
    stitched_image = crop_img(stitched_np)    
    stitched_image = Image.fromarray(stitched_image)

    return stitched_image

def create_depth_map_from_pano(pano_image_pil):
    # ZoeD_N
    model_zoe_n = torch.hub.load("zoe_depth_map/ZoeDepth/", "ZoeD_N", source="local", pretrained=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    zoe = model_zoe_n.to(device)

    depth_tensor = convert_to_depth_tensor(pano_image_pil, zoe)

    return depth_tensor







