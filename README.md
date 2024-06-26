
# I-See Project: Intracortical Visual Prosthetic (ICVP) System
## Purpose: 
These scripts allow real-time conversion of a stream of images in visual phosphenes to replicate vision generated by stimulation of the visual center of the brain. It leverages the architecture of OpenCV and YOLO (a Convolutional Neural Network that  identifies and isolates objects in a frame in a relatively short period of time, ideal for realtime processes), and pytorch (to take advantage of the GPU).
## Quick Start Guide

### Prerequisites
- Ensure you have Python installed (Ideally 3.10) with all necessary libraries: PyTorch, OpenCV, and YOLO.

### Launching the System

#### Starting the Encoder:
- The Encoder (`Encoder.py`) initiates the process by capturing real-time video input through a webcam or video file.
- Key functions like `Yolo5Segmentation` from the `processing_chain` module are utilized here for object detection. This function integrates YOLO for object recognition and segmentation.
- The `ContourExtract`, `PatchGenerator`, and `Sparsifier` functions process the detected object to generate and encode visual primitives.

#### Data Communication:
- Upon encoding the visual data into primitives, the `DataPacket` class (found in `data_comm` module) is used to encapsulate the information, preparing it for transmission.
- The `communicate_sender` function then wirelessly transmits these packets to the Stimulator, simulating the transfer of visual data to a prosthetic device.

#### Stimulator Activation:
- Upon receiving data, the Stimulator (`Stimulator.py`) uses the `communicate_receiver` function to accept incoming packets.
- The `BuildImage` and `PatchGenerator` functions from the `processing_chain` module are employed to reconstruct and simulate the visual perception based on the encoded data.
- The system then generates visual stimuli, aiming to simulate how processed visual patterns would appear to the ICVP user.

### Running the System
Execute the following commands in your terminal or command prompt, ensuring you are in the project directory:

To start the Encoder:
```bash
python Encoder.py
```

To run the Stimulator (in a separate terminal or session):
```bash
python Stimulator.py
```

Ensure both scripts are running simultaneously for real-time data communication and processing.

## System Setup and Operation

### Encoder Setup
The Encoder module (`Encoder.py`) is the first step in the I-See project pipeline. It captures real-time video input and processes it through a series of functions to extract and encode visual data into a format suitable for the Stimulator module. Here’s how it works:

- **Initialization**: The system starts with setting up the necessary libraries and modules, ensuring all dependencies like PyTorch, OpenCV, and custom processing chains are correctly loaded.
- **Image Capture**: Utilizes the `WebCam` module to capture real-time video feeds from the webcam.
- **Object Detection**: Implements the YOLO deep learning algorithm (through `Yolo5Segmentation` function) for efficient and fast object detection, isolating significant visual elements in the scene.
- **Image Processing**: After detecting the objects, the image is passed through various filters and processors, such as the Gabor filter (`ContourExtract` function) for edge detection, to refine and emphasize the critical visual features.
- **Data Encoding**: The processed visual data are then encoded into a sparse representation using the `Sparsifier` function, making it easier to transmit and stimulate the visual cortex effectively.

### Stimulator Setup
Following the encoding process, the Stimulator module (`Stimulator.py`) receives the encoded data and translates it into visual stimuli. Here’s the sequence of operations:

- **Receiving Data**: Through a data communication mechanism (`communicate_receiver` function), the Stimulator receives the encoded visual packets from the Encoder.
- **Image Reconstruction**: Utilizes the `BuildImage` function to reconstruct the visual stimuli from the sparse representations, ensuring the output matches the intended phosphene or clock patterns.
- **Visualization Simulation**: Based on the reconstructed image, the system simulates how the visual primitives would appear to the visual cortex, using functions like `PatchGenerator` to map and display these primitives accurately.
- **Real-time Adjustment**: Provides real-time feedback and adjustment capabilities, ensuring the stimuli are optimized for the user's perceptual experience.
