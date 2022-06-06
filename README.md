# Convolutional-Neural-Network

## This repository contains two implementations of convolutional neural networks: 
### 1. Car Detection With YOLO
### 2. Image Segmentation With Unet
##
#### 1. Car Detection With YOLO:
"You Only Look Once" (YOLO) is a popular algorithm because it achieves high accuracy while also being able to run in real time. This algorithm "only looks once" at the image in the sense that it requires only one forward propagation pass through the network to make predictions. After non-max suppression, it then outputs recognized objects together with the bounding boxes.

Many of the ideas in this notebook are described in the two YOLO papers: [Redmon et al., 2016](https://arxiv.org/abs/1506.02640) and [Redmon and Farhadi, 2016](https://arxiv.org/abs/1612.08242).

Dataset provided by https://www.drive.ai/

If there are 80 classes you want the object detector to recognize, you can represent the class label 'c' either as an integer from 1 to 80, or as an 80-dimensional vector (with 80 numbers) one component of which is 1, and the rest of which are 0. The video lectures used the latter representation; in this notebook, you'll use both representations, depending on which is more convenient for a particular step.

<b>Inputs and outputs:
1. The input is a batch of images, and each image has the shape (m, 608, 608, 3)
2. The output is a list of bounding boxes along with the recognized classes. Each bounding box is represented by 6 numbers as explained above. If you expand  into an 80-dimensional vector, each bounding box is then represented by 85 numbers.
</b>  
Anchor boxes are chosen by exploring the training data to choose reasonable height/width ratios that represent the different classes. For this assignment, 5 anchor boxes were chosen for you (to cover the 80 classes), and stored in the file './model_data/yolo_anchors.txt'
The dimension of the encoding tensor of the second to last dimension based on the anchor boxes is .
The YOLO architecture is: IMAGE (m, 608, 608, 3) -> DEEP CNN -> ENCODING (m, 19, 19, 5, 85).
  
Here's one way to visualize what YOLO is predicting on an image:

For each of the 19x19 grid cells, find the maximum of the probability scores (taking a max across the 80 classes, one maximum for each of the 5 anchor boxes).
Color that grid cell according to what object that grid cell considers the most likely.
  
The only boxes plotted are ones for which the model had assigned a high probability to reduce the algorithm's output there is essential usage of non-max suppression.
  
Filtering with a Threshold on Class Scores: 
<br> The model gives you a total of 19x19x5x85 numbers, with each box described by 85 numbers. It's convenient to rearrange the (19,19,5,85) (or (19,19,425)) dimensional tensor into the following variables:  
1. box_confidence: tensor of shape $(19, 19, 5, 1)$ containing $p_c$ (confidence probability that there's some object) for each of the 5 boxes predicted in each of the 19x19 cells.
2. boxes: tensor of shape $(19, 19, 5, 4)$ containing the midpoint and dimensions $(b_x, b_y, b_h, b_w)$ for each of the 5 boxes in each cell.
3. box_class_probs: tensor of shape $(19, 19, 5, 80)$ containing the "class probabilities" $(c_1, c_2, ... c_{80})$ for each of the 80 classes for each of the 5 boxes per cell.

<b>Definition of a box:</b>

![image](https://user-images.githubusercontent.com/86974424/172105885-9e78f966-d2d8-4967-927a-d5528747d015.png)

<b>Encoding architecture for YOLO:</b>

![image](https://user-images.githubusercontent.com/86974424/172105975-3f2ccda3-6162-4d7c-a24a-04f91f7e8621.png)

<b>Visualizing classes:</b>

![image](https://user-images.githubusercontent.com/86974424/172106077-ec9ab376-9ba5-4b43-ac6a-30d1d64ed253.png)

<b>Visualizing bounding boxes:</b>

![image](https://user-images.githubusercontent.com/86974424/172106143-b63b1d7e-2f71-4956-92e4-bdc3e5786ac2.png)

<b>Non-max Suppression:</b> Even after filtering by thresholding over the class scores, we still end up with a lot of overlapping boxes. A second filter for selecting the right boxes is called non-maximum suppression (NMS).

![image](https://user-images.githubusercontent.com/86974424/172106289-295ce3ca-3386-4043-8650-353552adc92f.png)

<b>NOTE: YOLO's network was trained to run on 608x608 images. If you are testing this data on a different size image -- for example, the car detection dataset had 720x1280 images -- this step rescales the boxes so that they can be plotted on top of the original 720x1280 image.</b>

<b>Training the model:</b>
Training a YOLO model takes a very long time and requires a fairly large dataset of labelled bounding boxes for a large range of target classes. You are going to load an existing pre-trained Keras YOLO model stored in "yolo.h5". These weights come from the official YOLO website, and were converted using a function written by Allan Zelener.

## Output and Output scores:

![image](https://user-images.githubusercontent.com/86974424/172107128-59b47496-ee87-4bc8-96ff-7f588ac2fba8.png)
##
#### 2. Image Segmentation With Unet
This type of image classification is called semantic image segmentation. It's similar to object detection in that both ask the question: "What objects are in this image and where in the image are those objects located?," but where object detection labels objects with bounding boxes that may include pixels that aren't part of the object, semantic image segmentation allows you to predict a precise mask for each object in the image by labeling each pixel in the image with its corresponding class. The word “semantic” here refers to what's being shown, so for example the “Car” class is indicated below by the dark blue mask, and "Person" is indicated with a red mask:
![image](https://user-images.githubusercontent.com/86974424/172108089-80e92ec0-e2ce-47c1-8393-48dbe9233aeb.png)

##### Loading and Spliting the Data also Checking out some of the unmasked and masked images from the dataset:
![image](https://user-images.githubusercontent.com/86974424/172108615-17c64c0b-1728-4978-a234-cbbf182f698d.png)
 
U-Net, named for its U-shape, was originally created in 2015 for tumor detection, but in the years since has become a very popular choice for other semantic segmentation tasks.

U-Net builds on a previous architecture called the Fully Convolutional Network, or FCN, which replaces the dense layers found in a typical CNN with a transposed convolution layer that upsamples the feature map back to the size of the original input image, while preserving the spatial information. This is necessary because the dense layers destroy spatial information (the "where" of the image), which is an essential part of image segmentation tasks. An added bonus of using transpose convolutions is that the input size no longer needs to be fixed, as it does when dense layers are used.

Unfortunately, the final feature layer of the FCN suffers from information loss due to downsampling too much. It then becomes difficult to upsample after so much information has been lost, causing an output that looks rough.

U-Net improves on the FCN, using a somewhat similar design, but differing in some important ways. Instead of one transposed convolution at the end of the network, it uses a matching number of convolutions for downsampling the input image to a feature map, and transposed convolutions for upsampling those maps back up to the original input image size. It also adds skip connections, to retain information that would otherwise become lost during encoding. Skip connections send information to every upsampling layer in the decoder from the corresponding downsampling layer in the encoder, capturing finer information while also keeping computation low. These help prevent information loss, as well as model overfitting.

![image](https://user-images.githubusercontent.com/86974424/172108807-3c474ede-915d-42af-b07c-039b6e817f2e.png)

##### Encoder (Downsampling Block):
Each conv_block() is composed of 2 Conv2D layers with ReLU activations. We will apply Dropout, and MaxPooling2D to some conv_blocks, as we will verify in the following sections, specifically to the last two blocks of the downsampling.

The function will return two tensors:
1. next_layer: That will go into the next block.
2. skip_connection: That will go into the corresponding decoding block.

![image](https://user-images.githubusercontent.com/86974424/172109185-a835d0b9-8da0-4972-970d-10d595eb94d0.png)

##### Decoder (Upsampling Block):
The decoder, or upsampling block, upsamples the features back to the original image size. At each upsampling level, we'll take the output of the corresponding encoder block and concatenate it before feeding to the next decoder block.

![image](https://user-images.githubusercontent.com/86974424/172109306-6bc221e5-e2aa-4dd3-806d-51ddd6cbe971.png)

##### U-net modelling:

![image](https://user-images.githubusercontent.com/86974424/172109394-b719b0b1-ca5c-4d73-9da5-f9f3b0b1f899.png)

##### Training and Plotting model accuracy:

![image](https://user-images.githubusercontent.com/86974424/172109537-46ba6e9d-51b6-4116-8f52-7a5198df9562.png)

## Results & Prediction:
![image](https://user-images.githubusercontent.com/86974424/172109736-749591bc-a10e-46c9-a6a0-aa45079778f0.png)
![image](https://user-images.githubusercontent.com/86974424/172110824-f59d74c6-ee53-496d-9c99-5867aea8e6ae.png)
![image](https://user-images.githubusercontent.com/86974424/172110854-62b7ec0c-d18c-474c-a86c-bfb66f3fdab8.png)
![image](https://user-images.githubusercontent.com/86974424/172110865-b44de46f-a8f7-4257-afd8-a39b955d5380.png)
