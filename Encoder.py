# %%
#
# test_PsychophysicsEncoding.py
# ========================================================
# encode visual scenes into sparse representations using
# different kinds of dictionaries
#
# Version 1.0, 07.03.2023:
#   made pretty and introduced proper parameter definitions
#   for Elsa's psychophysics experiments
#
# Version 1.1, 07.03.2023:
#   made compatible with updated routines:
#   ContourExtract.py, PatchGenerator.py, Sparsifier.py
#   added BuildImage.py for sparse image generation
#
# Version 1.2, 09.03.2023:
#   merged with percept simulator directory, now modules
#   are found in 'processing_chain', and test images and
#   output in 'test/images' and 'test/output'


# Import Python modules
# ========================================================
# import csv
# import time
import os
import matplotlib.pyplot as plt
import torch
import torchvision as tv
from PIL import Image
import cv2
import numpy as np
import time

# Import our modules
# ========================================================
from processing_chain.ContourExtract import ContourExtract
from processing_chain.PatchGenerator import PatchGenerator
from processing_chain.Sparsifier import Sparsifier
from processing_chain.DiscardElements import discard_elements_simple
from processing_chain.BuildImage import BuildImage
from processing_chain.Yolo5Segmentation import Yolo5Segmentation
from data_comm.communicate_datapacket import DataPacket, field
from data_comm.communicate_sender import communicate_sender
from WebCam import WebCam


# Define parameters
# ========================================================
# Unit abbreviations:
#   dva = degrees of visual angle
#   pix = pixels

# display: Defines geometry of target display
# ========================================================
# The encoded image will be scaled such that it optimally uses
# the max space available. If the orignal image has a different aspect
# ratio than the display region, it will only use one spatial
# dimension (horizontal or vertical) to its full extent
#
# If one DVA corresponds to different PIX_per_DVA on the display,
# (i.e. varying distance observers from screen), it should be set
# larger than the largest PIX_per_DVA required, for avoiding
# extrapolation artefacts or blur.
#
display = {
    "size_max_x_DVA": 10.0,  # maximum x size of encoded image
    "size_max_y_DVA": 10.0,  # minimum y size of encoded image
    "PIX_per_DVA": 80.0,  # scaling factor pixels to DVA
    "scale": "same_range",  # "same_luminance" or "same_range"
}


# gabor: Defines paras of Gabor filters for contour extraction
# ==============================================================
gabor = {
    "sigma_kernel_DVA": 0.06,
    "lambda_kernel_DVA": 0.12,
    "n_orientations": 8,
}

# encoding: Defines parameters of sparse encoding process
# ========================================================
# Roughly speaking, after contour extraction dictionary elements
# will be placed starting from the position with the highest
# overlap with the contour. Elements placed can be surrounded
# by a dead or inhibitory zone to prevent placing further elements
# too closely. The procedure will map 'n_patches_compute' elements
# and then stop. For each element one obtains an overlap with the
# contour image.
#
# After placement, the overlaps found are normalized to the max
# overlap found, and then all elements with a larger normalized overlap
# than 'overlap_threshold' will be selected. These remaining
# elements will comprise a 'full' encoding of the contour.
#
# To generate even sparser representations, the full encoding can
# be reduced to a certain percentage of elements in the full encoding
# by setting the variable 'percentages'
#
# Example: n_patches_compute = 100 reduced by overlap_threshold = 0.1
# to 80 elements. Requesting a percentage of 30% yields a representation
# with 24 elements.
#

encoding = {
    "n_patches_compute": 100,  # this amount of patches will be placed
    "use_exp_deadzone": True,  # parameters of Gaussian deadzone
    "size_exp_deadzone_DVA": 1.20,  # PREVIOUSLY 1.4283
    "use_cutout_deadzone": True,  # parameters of cutout deadzone
    "size_cutout_deadzone_DVA": 0.65,  # PREVIOUSLY 0.7575
    "overlap_threshold": 0.1,  # relative overlap threshold
    "percentages": torch.tensor([100]),
}


# dictionary: Defines parameters of dictionary
# ========================================================
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

# control: For controlling plotting options and flow of script
# ========================================================
control = {
    "force_torch_use_cpu": True,  # force using CPU even if GPU available
    "debug_use_testimage": True,
    "plot_phosphene": True,  # show how phosphenes look like
    "plot_clocks": True,  # show how clocks look like
    "plot_contour": True,  # show contours extracted
    "plot_parameter_summary": True,  # shows geometry params overview
    "plot_deadzone": False,  # plots deadzone evolution during sparsification
    "plot_prostheticvision": True,  # shows sparsified images
}


# path: Path infos for input and output images
# ========================================================
path = {"output": "test/output/level1/", "input": "test/images_test/"}

# global scaling factors for all pixel-related length scales
display_size_max_x_PIX: float = display["size_max_x_DVA"] * display["PIX_per_DVA"]
display_size_max_y_PIX: float = display["size_max_y_DVA"] * display["PIX_per_DVA"]

# set patch size for both dictionaries, make sure it is odd number
dictionary_size_PIX: int = (
    1 + (int(dictionary["size_DVA"] * display["PIX_per_DVA"]) // 2) * 2
)

# convert contour-related parameters to pixel units
sigma_kernel_PIX: float = gabor["sigma_kernel_DVA"] * display["PIX_per_DVA"]
lambda_kernel_PIX: float = gabor["lambda_kernel_DVA"] * display["PIX_per_DVA"]

# Padding
# -------
padding_PIX: int = int(max(3.0 * sigma_kernel_PIX, 1.1 * dictionary_size_PIX))
tmp = tv.transforms.Grayscale(num_output_channels=1)
tmp_value = torch.full((3, 1, 1), 255)
padding_fill: int = int(tmp(tmp_value).squeeze())
#Ene's corrections

# =================================================================
# NO USER-SERVICEABLE PARTS BEYOND THAT POINT!
# =================================================================


# some constants for addressing specific components of output arrays
image_id_CONST: int = 0
overlap_index_CONST: int = 1


# Check if GPU is available and use it, if possible
# =================================================
default_dtype = torch.float32
torch.set_default_dtype(default_dtype)
if control["force_torch_use_cpu"]:
    torch_device: str = "cpu"
else:
    torch_device: str = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {torch_device} as TORCH device...")


# Make directories, if necessary: the place were we dump the new images to...
# =======================    =====================================================
 
# generate a Class to send the data 

# UDO: Ene starts here...
# UDO: Numpy is really required? - LESS (Less important)

#display the frame with imshow
def show_torch_frame(frame_torch: torch.Tensor, title: str = "", cmap: str = None):
    # Assuming frame_torch is in the shape [C, H, W] and C=3 for RGB
    # Move the channel dimension to the last axis for imshow
    frame_numpy = frame_torch.permute(1, 2, 0).cpu().numpy()  # Use permute instead of movedim
    
    # Normalize the frame to 0-255 range across all channels
    frame_min = frame_numpy.min(axis=(0, 1), keepdims=True)
    frame_max = frame_numpy.max(axis=(0, 1), keepdims=True)
    frame_numpy = (frame_numpy - frame_min) / (frame_max - frame_min) * 255
    
    # Convert to uint8 for display
    frame_numpy = frame_numpy.astype(np.uint8)

    # Display the image using plt.imshow, no cmap needed for RGB
    plt.imshow(frame_numpy,cmap=cmap)
    plt.title(title)
    plt.axis('off')  # Hide axes ticks
    plt.show()

#emb a smaller image into a large image. Preserve the aspect ratio of the smaller one. 
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



# open the video file you want to take
# UDO: include list of other classes.
available_classes: list[str] = [
        "--: None",
        "00: person",
        "01: bicycle",
        "02: car",
        "03: motorcycle",
        "04: airplane",
        "05: bus",
        "06: train",
        "07: truck",
        "08: boat",
        "09: traffic light",
        "10: fire hydrant",
        "11: stop sign",
        "12: parking meter",
        "13: bench",
        "14: bird",
        "15: cat",
        "16: dog",
        "17: horse",
        "18: sheep",
        "19: cow",
        "20: elephant",
        "21: bear",
        "22: zebra",
        "23: giraffe",
        "24: backpack",
        "25: umbrella",
        "26: handbag",
        "27: tie",
        "28: suitcase",
        "29: frisbee",
        "30: skis",
        "31: snowboard",
        "32: sports ball",
        "33: kite",
        "34: baseball bat",
        "35: baseball glove",
        "36: skateboard",
        "37: surfboard",
        "38: tennis racket",
        "39: bottle",
        "40: wine glass",
        "41: cup",
        "42: fork",
        "43: knife",
        "44: spoon",
        "45: bowl",
        "46: banana",
        "47: apple",
        "48: sandwich",
        "49: orange",
        "50: broccoli",
        "51: carrot",
        "52: hot dog",
        "53: pizza",
        "54: donut",
        "55: cake",
        "56: chair",
        "57: couch",
        "58: potted plant",
        "59: bed",
        "60: dining table",
        "61: toilet",
        "62: tv",
        "63: laptop",
        "64: mouse",
        "65: remote",
        "66: keyboard",
        "67: cell phone",
        "68: microwave",
        "69: oven",
        "70: toaster",
        "71: sink",
        "72: refrigerator",
        "73: book",
        "74: clock",
        "75: vase",
        "76: scissors",
        "77: teddy bear",
        "78: hair drier",
        "79: toothbrush",
    ]
class_cup = 41
class_person = 0
classes_detect = [class_cup]

# define target video/representation width/height
multiple_of = 4
out_x = display_size_max_x_PIX + 2 * padding_PIX
out_y = display_size_max_y_PIX + 2 * padding_PIX
out_x += (multiple_of - (out_x % multiple_of)) % multiple_of
out_y += (multiple_of - (out_y % multiple_of)) % multiple_of
out_x = int(out_x)
out_y = int(out_y)

VERBOSE = True

# UDO: put all constants or parameters HERE!
source: str | int = "pics/level2.mp4"  # video file or device to read from
max_frame_count: int = 5 #(
#    torch.inf
#)  # number of frames to process (torch.inf for no limits)
output_file: str | None = "output_level23.avi"  # output file to write to, or None

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

# open output file if we want to save frames
if output_file != None:
    out = cv2.VideoWriter(
        output_file,
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (out_x, out_y),
    )
    if out == None:
        raise OSError(f"Can not open file {output_file} for writing!")

# get an instance of the Yolo segmentation network
yolo = Yolo5Segmentation()

# Loop over the frames in the video file

# create the windows that will contain the plots
#cv2.namedWindow("Plots", cv2.WINDOW_NORMAL)
i_frame = 0
while i_frame < frame_count:

    i_frame += 1

    # format: color_RGB, height, width <class 'torch.tensor'> float, range=0,1
    frame = cap.get_frame()
    if frame == None:
        raise OSError(f"Can not capture frame {i_frame}")

    # perform segmentation
    frame_segmented = yolo(frame.unsqueeze(0), classes=classes_detect)

    # Display on image
    # display
    if VERBOSE:
        show_torch_frame(frame,title=f"Frame #{i_frame}")
        # key = cv2.waitKey(1)
        # if key == ord('q'):
        #     break

    # This extracts the frame in x to convert the mask in a video format
    if yolo.found_class_id != None:

        n_found = len(yolo.found_class_id)
        print(f"{n_found} occurrences of desired object found in frame {i_frame}!")

        mask = frame_segmented[0]
        if VERBOSE:
            show_torch_frame(
                frame_segmented,
                f"YOLO from #{i_frame}",
                cmap="gray",
            )
        # is there something in the mask?
        if not mask.sum() == 0:

            # yes, cut only the part of the frame that has our object of interest
            frame_masked = mask * frame

            x_height = mask.sum(axis=-2)
            x_indices = np.where(x_height > 0)
            x_min, x_max = x_indices[0].min(), x_indices[0].max() + 1

            y_height = mask.sum(axis=-1)
            y_indices = np.where(y_height > 0)
            y_min, y_max = y_indices[0].min(), y_indices[0].max() + 1

            # Define a margin
            margin = 20  # Adjust the margin size as needed

            # Apply margin, ensuring indices stay within image bounds
            x_min = max(x_min - margin, 0)
            x_max = min(x_max + margin, frame.shape[-1])
            y_min = max(y_min - margin, 0)
            y_max = min(y_max + margin, frame.shape[-2])

            # Crop the frame with margins
            frame_cut = frame_masked[:, y_min:y_max, x_min:x_max]
        else:
            print(f"Mask contains all zeros in frame {i_frame}!")
            frame_cut = None
    else:
        print(f"No objects found in frame {i_frame}!")
        frame_cut = None
    if frame_cut == None:
        out_torch = torch.zeros([out_y, out_x])
    else:
        if VERBOSE:
            show_torch_frame(frame_cut, f"Cut from #{i_frame}")
            # key = cv2.waitKey(1)
            # if key == ord('q'):
            #     break

        # UDO: from here on, we proceed as before, just handing
        # UDO: over the frame_cut --> image
        image = frame_cut
        start_time = time.time() #Start the counter for image decay.

        # Determine target size of image
        # image: [RGB, Height, Width], dtype= tensor.torch.uint8
        print("Computing downsampling factor image -> display")
        f_x: float = display_size_max_x_PIX / image.shape[-1]
        f_y: float = display_size_max_y_PIX / image.shape[-2]
        f_xy_min: float = min(f_x, f_y)
        downsampling_x: int = int(f_xy_min * image.shape[-1])
        downsampling_y: int = int(f_xy_min * image.shape[-2])

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

        # define contour extraction processing chain
        # ------------------------------------------
        print("Extracting contours")
        train_processing_chain = tv.transforms.Compose(
            transforms=[
                tv.transforms.Grayscale(num_output_channels=1),  # RGB to grayscale
                tv.transforms.Resize(
                    size=(downsampling_y, downsampling_x)
                ),  # downsampling
                tv.transforms.Pad(  # extra white padding around the picture
                    padding=(padding_PIX, padding_PIX),
                    fill=0, #replace by padding_fill
                ),
                ContourExtract(  # contour extraction
                    n_orientations=gabor["n_orientations"],
                    sigma_kernel=sigma_kernel_PIX,
                    lambda_kernel=lambda_kernel_PIX,
                    torch_device=torch_device,
                ),
                # CURRENTLY we do not crop in the end!
                # tv.transforms.CenterCrop(  # Remove the padding
                #     size=(center_crop_x, center_crop_y)
                # ),
            ],
        )
        # ...with and without orientation channels
        contour = train_processing_chain(image.unsqueeze(0))
        contour_collapse = train_processing_chain.transforms[-1].create_collapse(
            contour
        )

        if VERBOSE:
            show_torch_frame(contour_collapse, f"Contour from #{i_frame}", cmap="gray")
            # key = cv2.waitKey(1)
            # if key == ord('q'):
            #     break
        # ----------------------------------------------------------
        dictionary_prior = torch.ones(
            (clocks_filter.shape[0]),
            dtype=default_dtype,
            device=torch.device(torch_device),
        )
        # instantiate and execute sparsifier
        # ----------------------------------
        previous_time = time.time()
        print("Performing sparsification")
        sparsifier = Sparsifier(
            dictionary_filter=clocks_filter,
            dictionary=clocks,
            dictionary_prior=dictionary_prior,
            number_of_patches=encoding["n_patches_compute"],
            size_exp_deadzone=encoding["size_exp_deadzone_DVA"]
            * display["PIX_per_DVA"],
            plot_use_map=control["plot_deadzone"],
            deadzone_exp=encoding["use_exp_deadzone"],
            deadzone_hard_cutout=encoding["use_cutout_deadzone"],
            deadzone_hard_cutout_size=encoding["size_cutout_deadzone_DVA"]
            * display["PIX_per_DVA"],
            padding_deadzone_size_x=padding_PIX,
            padding_deadzone_size_y=padding_PIX,
            torch_device=torch_device,
        )
        sparsifier(contour)
        assert sparsifier.position_found is not None

        # extract and normalize the overlap found
        overlap_found = sparsifier.overlap_found[image_id_CONST, :, overlap_index_CONST]
        overlap_found = overlap_found / overlap_found.max()





        # get overlap above certain threshold, extract corresponding elements
        overlap_idcs_valid = torch.where(
            overlap_found >= encoding["overlap_threshold"]
        )[0]
        position_selection = sparsifier.position_found[
            image_id_CONST : image_id_CONST + 1, overlap_idcs_valid, :
        ]
        n_elements = len(overlap_idcs_valid)
        print("{} elements positioned!".format(n_elements))

        # ENE: DATA LOADING 
        if not VERBOSE:
            packet = DataPacket(contour=contour, positions=position_selection)#,positions=position_selection
            #send it to the Stimulator
            url = "tcp://localhost:5555"
            response = communicate_sender(packet, url)


        # build the full image!
        image_clocks = BuildImage(
            canvas_size=contour.shape,
            dictionary=clocks,
            position_found=position_selection,
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

        # normalize to range [0...1]
        m = image_clocks[0].max()
        if m != 0:
            image_clocks_normalized = image_clocks[0] / image_clocks[0].max()

        # embed into frame of desired output size
        out_torch = embed_image(
            image_clocks_normalized, out_height=out_y, out_width=out_x
        )
        m_phosphenes = image_phosphenes[0].max()
        if m_phosphenes != 0:
            image_phosphenes_normalized = image_phosphenes[0] / m_phosphenes

        # Embed into frame of desired output size
        out_torch_phosphenes = embed_image(
            image_phosphenes_normalized, out_height=out_y, out_width=out_x
        )
    # show, if desired
    if VERBOSE:
        show_torch_frame(
            out_torch,
            f"Clocks from #{i_frame}",
            cmap="gray",
        )


    # show, if desired
    if VERBOSE:
        show_torch_frame(
            out_torch_phosphenes,
            f"Phosphenes from #{i_frame}",
            cmap="gray",
        )

    if output_file != None:
        # out_torch = embed_image(
        #     image_clocks_normalized, out_height=out_y, out_width=out_x
        # )
        out_pixel = (
            (out_torch * torch.ones([3, 1, 1]) * 255)
            .type(dtype=torch.uint8)
            .movedim(0, -1)
            .numpy()
        )
        out.write(out_pixel)
        # if not success:
        #     print(f"Can not write frame {i_frame} to video file, continuing...")
        videostream = False
        t_max= 6400

        brightness = np.zeros([out_x,out_y,3])
        # Loop until the elapsed time is greater than the maximum time

        n=0
        tau = 1# time constant

        #or reload the image
        
        percent_dicts_per_sec = 0.1 # the percentage of dicts that will be removed per second
        # url = "tcp://*:5555"
        # packet = DataPacket() # construct your DataPacket object here

        stimulator = 6400

        # init time variables
        dt= time.time() - start_time
        dict_time = time.time() - previous_time
        print('dt =',dt)
        print('dict_time =', dict_time)


        # Calculate the number of dicts to keep based on elapsed time and % per second
        dicts_to_keep = int(position_selection.shape[1] * (1 - percent_dicts_per_sec) ** dict_time)
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


cap.close_cam()
#cv2.destroyAllWindows()

if output_file != None:
    out.release()


# %%
