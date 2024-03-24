
#%% Imagebuild

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision as tv
import numpy as np
#import PIL.Image
import time

from processing_chain.PatchGenerator import PatchGenerator
from processing_chain.BuildImage import BuildImage
from data_comm.communicate_receiver import communicate_receiver 
from data_comm.communicate_datapacket import DataPacket
from WebCam import WebCam



# Recieve the packets from the Encoder


# control: For controlling plotting options and flow of script
# ========================================================


control = {
    "force_torch_use_cpu": True,  # force using CPU even if GPU available
    "debug_use_testimage": False,
    "plot_phosphene": False,  # show how phosphenes look like
    "plot_clocks": False,  # show how clocks look like
    "plot_contour": False,  # show contours extracted
    "plot_parameter_summary": False,  # shows geometry params overview
    "plot_deadzone": False,  # plots deadzone evolution during sparsification
    "plot_prostheticvision": True,  # shows sparsified images
}
# some constants for addressing specific components of output arrays
image_id_CONST: int = 0
overlap_index_CONST: int = 1
# First, the code to generate dictionaries
#this uses PatchGenerator

# First, you have to init and create the dict. elements. This needs: SizeDVA, the DVA that the Dict will take
# how big the phosphenes will be and how the clocks will be defined 

# display: Defines geometry of target display
# it amakes ure it fits perfectly on the screen even if it has to use one dimension,
# ex: if people are far away from the screen it makes the image bigger 
# to avoid making it look blurry or weird or something

display = {
    "size_max_x_DVA": 10.0,  # maximum x size of encoded image
    "size_max_y_DVA": 10.0,  # minimum y size of encoded image
    "PIX_per_DVA": 80.0,  # scaling factor pixels to DVA
    "scale": "same_range",  # "same_luminance" or "same_range"
}

# how the x y lenghts are going to be affected
display_size_max_x_PIX: float =  display["size_max_x_DVA"] * display["PIX_per_DVA"]
display_size_max_y_PIX: float = display["size_max_y_DVA"] * display["PIX_per_DVA"]


dictionary = {
    "size_DVA": 1.0,  # PREVIOUSLY 1.25,
    "clocks": None,  # parameters for clocks dictionary, see below
    "phosphene": None,  # paramters for phosphene dictionary, see below
}

dictionary["phosphene"]: dict[float] = {
    "sigma_width": 0.18,  # DEFAULT 0.15,  # half-width of Gaussian
}

dictionary["clocks"]: dict[int, int, float, float] = {
    "n_dir": 8,  # number of directions for clock pointer segments
    "n_open": 4,  # number of opening angles between two clock pointer segments
    "pointer_width": 0.07,  # PREVIOUSLY 0.05,  # relative width and size of tip extension of clock pointer
    "pointer_length": 0.18,  # PREVIOUSLY 0.15,  # relative length of clock pointer
}



# gabor: Defines paras of Gabor filters for contour extraction
# ==============================================================
gabor = {
    "sigma_kernel_DVA": 0.06,
    "lambda_kernel_DVA": 0.12,
    "n_orientations": 8,
}

# set patch size for both dictionaries, make sure it is odd number
dictionary_size_PIX: int = (
    1 + (int(dictionary["size_DVA"] * display["PIX_per_DVA"]) // 2) * 2
)

#Kernel
sigma_kernel_PIX: float = gabor["sigma_kernel_DVA"] * display["PIX_per_DVA"]
lambda_kernel_PIX: float = gabor["lambda_kernel_DVA"] * display["PIX_per_DVA"]

#padding
padding_PIX: int = int(max(3.0 * sigma_kernel_PIX, 1.1 * dictionary_size_PIX))
tmp = tv.transforms.Grayscale(num_output_channels=1)
tmp_value = torch.full((3, 1, 1), 0)
padding_fill: int = int(tmp(tmp_value).squeeze())




# Check if GPU is available and use it, if possible
# =================================================
default_dtype = torch.float32
torch.set_default_dtype(default_dtype)
if control["force_torch_use_cpu"]:
    torch_device: str = "cpu"
else:
    torch_device: str = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {torch_device} as TORCH device...")


# set patch size for both dictionaries, make sure it is odd number
dictionary_size_PIX: int = (
    1 + (int(dictionary["size_DVA"] * display["PIX_per_DVA"]) // 2) * 2
)
# generate dictionaries
# ---------------------







# define target video/representation width/height
multiple_of = 4
out_x = display_size_max_x_PIX + 2 * padding_PIX
out_y = display_size_max_y_PIX + 2 * padding_PIX
out_x += (multiple_of - (out_x % multiple_of)) % multiple_of
out_y += (multiple_of - (out_y % multiple_of)) % multiple_of
out_x = int(out_x)
out_y = int(out_y)
#ENE: Make sure the source of Stimulator and ENCODER are the same
# UDO: put all constants or parameters HERE!
source: str | int = "pics/level2.mp4"  # video file or device to read from
max_frame_count: int = (
    torch.inf
)  # number of frames to process (torch.inf for no limits)
output_file: str | None = "output_Stim2.avi"  # output file to write to, or None



# open source
cap = WebCam(source)
if not cap.open_cam():
    raise OSError(f"Opening source {source} failed!")

# get the video frame size, frame count and frame rate
frame_width = cap.cap_frame_width
frame_height = cap.cap_frame_height
fps = cap.cap_fps
if type(source) == str:
    frame_count = min([max_frame_count, cap.cap_frames_available])
else:
    frame_count = max_frame_count
print(f"Processing {frame_count} frames of {frame_width} x {frame_height} @ {fps} fps.")


if output_file != None:
    out = cv2.VideoWriter(
        output_file,
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (out_x, out_y),
    )
    if out == None:
        raise OSError(f"Can not open file {output_file} for writing!")


# init the video output


# generate dictionaries
# ---------------------
print("Generating dictionaries...")
patch_generator = PatchGenerator(torch_device=torch_device)
dictionary_phosphene = patch_generator.alphabet_phosphene(
    patch_size=dictionary_size_PIX,
    sigma_width=dictionary["phosphene"]["sigma_width"] * dictionary_size_PIX,
)
clocks_filter, clocks, segments = patch_generator.alphabet_clocks(
    patch_size=dictionary_size_PIX,
    n_dir=dictionary["clocks"]["n_dir"],
    n_filter=gabor["n_orientations"],
    segment_width=dictionary["clocks"]["pointer_width"] * dictionary_size_PIX,
    segment_length=dictionary["clocks"]["pointer_length"] * dictionary_size_PIX,
)
#clocks: [# of Dic_elemets, 1, X_image, Y_image], torch.float32
print("Data Initialized! Stimulation of neurons starting...")

# Define embedding to fix the image to a certain size
def embed_image(frame_torch, out_height, out_width, init_value=0):

    out_shape = torch.tensor(frame_torch.shape)

    frame_width = frame_torch.shape[-1]
    frame_height = frame_torch.shape[-2]

    frame_width_idx0 = max([0, (frame_width - out_width) // 2])
    frame_height_idx0 = max([0, (frame_height - out_height) // 2])

    select_width = min([frame_width, out_width])
    select_height = min([frame_height, out_height])

    out_shape[-1] = out_width
    out_shape[-2] = out_height

    out_torch = init_value * torch.ones(tuple(out_shape))

    out_width_idx0 = max([0, (out_width - frame_width) // 2])
    out_height_idx0 = max([0, (out_height - frame_height) // 2])

    out_torch[
        ...,
        out_height_idx0 : (out_height_idx0 + select_height),
        out_width_idx0 : (out_width_idx0 + select_width),
    ] = frame_torch[
        ...,
        frame_height_idx0 : (frame_height_idx0 + select_height),
        frame_width_idx0 : (frame_width_idx0 + select_width),
    ]

    return out_torch


# contour_collapse: torch.Size([1, H_final, W_final]) <class 'torch.Tensor'> torch.float32
#clocks: [# of Dic_elemets, 1, X_image, Y_image], torch.float32

#create the image after having change the values of the dict_elements
# 1. Be able to select the dict-elements - modify the position selection
# build the full image!

# image_clocks: torch.Size([1, 1, W, H]) <class 'torch.Tensor'> torch.float32



# %%   Gnerate a character brightness that is reinitialized everey 50ms



videostream = False
t_max= 6400

brightness = np.zeros([out_x,out_y,3])
# Loop until the elapsed time is greater than the maximum time

n=0
tau = 1# time constant
start_time = time.time() #definte the start time.

start_time = time.time() #or reload the image
previous_time = time.time()
percent_dicts_per_sec = 0.1 # the percentage of dicts that will be removed per second
# url = "tcp://*:5555"
# packet = DataPacket() # construct your DataPacket object here

stimulator = 6400

while time.time() - start_time < t_max: # until 50ms elapsed

    # init time variables
    dt= time.time() - start_time
    dict_time = time.time() - previous_time
    print('dt =',dt)
    print('dict_time =', dict_time)

    #recieve the packet and update the images
    # packet = communicate_receiver(url)
    # if packet is not None:
    #     print('Packet received. Proceeding to extraction...')
    #     contour = packet.contour
    #     position_selection = packet.positions
    # else:
    #     print("No packet was received. Continuing...")


    # Calculate the number of dicts to keep based on elapsed time and % per second
    dicts_to_keep = int(position_selection.shape[1] * (1 - percent_dicts_per_sec) ** (dict_time/3))
    ind_to_keep = torch.randperm(position_selection.shape[1])[:dicts_to_keep] # randomly select indices to keep
    position_update = position_selection[:, ind_to_keep, :] #what are the new positions
    print("generating image from dataPacket")
    # build the image
    image_clocks = BuildImage(
    canvas_size=contour.shape,
    dictionary=clocks,
    position_found=position_update,
    default_dtype=default_dtype,
    torch_device=torch_device,
)
    image_phosphenes = BuildImage(
    canvas_size=contour.shape,
    dictionary=dictionary_phosphene,
    position_found=position_selection,
    default_dtype=default_dtype,
    torch_device=torch_device,
)
    # create an image form the updated position
    image_update = image_clocks[0,0]
    image_clocks_normalized = ( image_update - image_update.min()) / (image_update.max() - image_update.min())
    # scale to [0, 255]
    image_clocks_embed= embed_image(
    image_clocks_normalized, out_height=out_y, out_width=out_x
    )
    image_clocks_scaled = np.uint8(image_clocks_embed * 255)
    image_np = np.squeeze(image_clocks_scaled)
    image_np = np.tile(image_np[:, :, np.newaxis],[1,1,3]) 

    #Add that image to our brightness tensor

    brightness = brightness * np.exp(-1 * (dict_time) / tau) + image_np  #decrease over time  
    n = n+1
    frame = brightness.astype(np.uint8)
    previous_time = time.time()
    if videostream != True:

        plt.imshow(frame)  
        plt.colorbar()
        plt.show()
    else: 
    # format = [width, height], class numpy.ndarray, type float64
        print(f"frame's format is:", frame.shape, type(frame), frame.dtype, "range:",frame.min(), frame.max())
        print(f"Output's format is:", out_y,out_x,3, "<class 'numpy.ndarray'> uint8, 0-255")

        out.write(frame) #[:,:,1]
    # (640, 480) <class 'numpy.ndarray'> uint8

print(f"number of frames:",n)
    
# format: height,width,color_RGB, <class 'numpy.ndarray'> uint8, range=0,255

out.release()
# %%
