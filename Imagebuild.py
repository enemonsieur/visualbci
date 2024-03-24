#%% Imagebuild

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from processing_chain.ContourExtract import ContourExtract
from processing_chain.PatchGenerator import PatchGenerator
from processing_chain.Sparsifier import Sparsifier
from processing_chain.DiscardElements import discard_elements_simple
from processing_chain.BuildImage import BuildImage
import torchvision as tv
from PIL import Image

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



encoding = {
    "n_patches_compute": 100,  # this amount of patches will be placed
    "use_exp_deadzone": True,  # parameters of Gaussian deadzone
    "size_exp_deadzone_DVA": 1.20,  # PREVIOUSLY 1.4283
    "use_cutout_deadzone": True,  # parameters of cutout deadzone
    "size_cutout_deadzone_DVA": 0.65,  # PREVIOUSLY 0.7575
    "overlap_threshold": 0.1,  # relative overlap threshold
    "percentages": torch.tensor([100]),
}

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
# global scaling factors for all pixel-related length scales
display_size_max_x_PIX: float =  display["size_max_x_DVA"] * display["PIX_per_DVA"]
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
tmp_value = torch.full((3, 1, 1), 0)
padding_fill: int = int(tmp(tmp_value).squeeze())

padding_PIX: int = int(max(3.0 * sigma_kernel_PIX, 1.1 * dictionary_size_PIX))
tmp = tv.transforms.Grayscale(num_output_channels=1)
tmp_value = torch.full((3, 1, 1), 0)
padding_fill: int = int(tmp(tmp_value).squeeze())

print("Finished defining variables!!!")
print("\n\n\n")

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



















#%% Generate the important stuff

#image: shape [3, H, W], type=torch.tensor, dtype=unit8
#Let's draw an image using PIL - Image Draw
import numpy as np
import PIL.Image


# Create image in Black BG with circle
image = PIL.Image.fromarray(np.uint8(np.zeros((480, 640, 3)) * 255))
draw = PIL.ImageDraw.Draw(image)
draw.ellipse((50, 50, 200, 200), fill=(255, 0, 0))
image = np.array(image)

#change the value etc...
image = torch.tensor(image.transpose((2, 0, 1)), dtype=torch.uint8)


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

# Determine target size of image
# image: [RGB, Height, Width], dtype= tensor.torch.uint8
print("Computing downsampling factor image -> display")
f_x: float = display_size_max_x_PIX / image.shape[-1]
f_y: float = display_size_max_y_PIX / image.shape[-2]
f_xy_min: float = min(f_x, f_y)
downsampling_x: int = int(f_xy_min * image.shape[-1])
downsampling_y: int = int(f_xy_min * image.shape[-2])

# define contour extraction processing chain
# ------------------------------------------
print("Extracting contours")
train_processing_chain = tv.transforms.Compose(
    transforms=[
        tv.transforms.Grayscale(num_output_channels=1),               # RGB to grayscale
        tv.transforms.Resize(size=(downsampling_y, downsampling_x)),  # downsampling
        tv.transforms.Pad(  # extra white padding around the picture
            padding=(padding_PIX, padding_PIX),
            fill=padding_fill,
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
#merge the 8 orientations in one
contour_collapse = train_processing_chain.transforms[-1].create_collapse(contour)

# contour_collapse: torch.Size([1, H_final, W_final]) <class 'torch.Tensor'> torch.float32

# generate a prior for mapping the contour to the dictionary
# CURRENTLY we use an uniform prior...
# ----------------------------------------------------------
dictionary_prior = torch.ones(
    (clocks_filter.shape[0]),
    dtype=default_dtype,
    device=torch.device(torch_device),
)

# instantiate and execute sparsifier
# ----------------------------------
print("Performing sparsification")
sparsifier = Sparsifier(
    dictionary_filter=clocks_filter,
    dictionary=clocks,
    dictionary_prior=dictionary_prior,
    number_of_patches=encoding["n_patches_compute"],
    size_exp_deadzone=encoding["size_exp_deadzone_DVA"] * display["PIX_per_DVA"],
    plot_use_map=control["plot_deadzone"],
    deadzone_exp=encoding["use_exp_deadzone"],
    deadzone_hard_cutout=encoding["use_cutout_deadzone"],
    deadzone_hard_cutout_size=encoding["size_cutout_deadzone_DVA"]
    * display["PIX_per_DVA"],
    padding_deadzone_size_x=padding_PIX,
    padding_deadzone_size_y=padding_PIX,
    torch_device=torch_device,
)
# image_clocks = sparsifier(contour)
sparsifier(contour)
assert sparsifier.position_found is not None


# extract and normalize the overlap found
overlap_found = sparsifier.overlap_found[image_id_CONST, :, overlap_index_CONST]
overlap_found = overlap_found / overlap_found.max()


# get overlap above certain threshold, extract corresponding elements
overlap_idcs_valid = torch.where(overlap_found >= encoding["overlap_threshold"])[0]
position_selection = sparsifier.position_found[
    image_id_CONST : image_id_CONST + 1, overlap_idcs_valid, :
]
n_elements = len(overlap_idcs_valid)
print("{} elements positioned!".format(n_elements))

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
# image_clocks: torch.Size([1, 1, W, H]) <class 'torch.Tensor'> torch.float32
# %%
