S15A - Data Preperation for Depth and FG Detection Network (Jointly Done by Anilkumar N Bhatt and Maruthi Srinivas):

------------------------------------

Below are the images created as part of data preparation

1) Gdrive Location : https://drive.google.com/drive/folders/1raMnribL-gsa4FEpX8QIeyR6yP4XgmP-?usp=sharing

   Total Size : 4.9 GB

   100 BG and its corresponding 100 Flip Images, Shape : 192X192X3, Type : jpg, Folder Name : BG_and_Its Flip Images

   100 FG images, Shape : 192x192x4, Type : png, transparent background, Folder Name : FG_Images

   400K FG_BG images, Shape : 192x192x3, Type : jpg, Zip File Name : FG_BG_400K.zip

2) Statistics :

   Mean : [0.56670278 0.49779153 0.43632878], Std-Dev : [0.38389994 0.30871084 0.25551239], Size : 3 GB

   Corresponding 400K FG_BG_Masks, Shape : 192x192x1, Type : jpg, Zip File Name : FG_BG_Mask_400K.zip

   Mean : [0.20249742], Std-Dev : [0.39961225], Size : 906 MB

   400K Depth images of FG_BG, Shape : 200x200x1, Type : jpg, Zip File Name : FG_BG_Depth_400K.zip

    Mean : [0.32939295], Std-Dev : [0.24930712], Size : 1 GB

    Log files corresponding to above three zip files which have file names, image size of BG, image size of FG and bounding box coordinates of overlaid FG image.

    FG_BG_Filename_Logs.txt, FG_BG_Mask_Filename_Logs.txt, Log_FG_BG_Depth_400K.txt

    Code for statistics calculation :            [statistics](https://github.com/mmaruthi/EVA4P1_S15A_Depth_FG_Detection_DataPrep/blob/master/Statistics_FG_BG_Mask_Depth.ipynb)

3) Data preperation details are as listed below:

   This part deals with data preparation which will be later used by a network that will predict foreground from background and how far foreground is from camera w.r.to background (depth).

    Since data required for training this network is not publicly available & crowdsourcing is also not possible, data preparation strategy as follows were adopted.

    Downloaded 100 background images from web. Images of public places mostly malls & shopping complexes were downloaded. Resized them to 192x192

     Downloaded 100 foreground images. Images of people were selected. Removed the background using Microsoft power point using 'Remove Background' option thereby adding transparent layer. After that cropped this image using 'Crop' option under 'Format' tab in PPT to select object only. Then saved this image in 'png' format so that transparency (alpha channel) is retained.

     Flipped the 100 background images we created in step 3 and saved it. This makes total 200 background images (100 – Regular, 100 – Flipped) all in jpg format

Sample BG images

![Sample BG Images](https://github.com/mmaruthi/EVA4P1_S15A_Depth_FG_Detection_DataPrep/blob/master/Images_For_ReadMe/BG_Sample10.png)

Corresponding BG Flip images:

![Corresponding BG Flip Images](https://github.com/mmaruthi/EVA4P1_S15A_Depth_FG_Detection_DataPrep/blob/master/Images_For_ReadMe/BG_Flip_Sample10.png)

Sample FG Images:

![Sample FG Images](https://github.com/mmaruthi/EVA4P1_S15A_Depth_FG_Detection_DataPrep/blob/master/Images_For_ReadMe/FG_Sample10.png)

FG_BG Preparation – Overlaying Foreground on Background Image. (400K images)
Code : https://github.com/anilbhatt1/EVA4P1_S15A_Depth_FG_Detection/blob/master/EVA4P1_S15_DataPrep_V1.ipynb

For each background image, each foreground image is overlaid in 20 random positions giving 1 BG * 100 FG * 20 Positions = 20000 images

So each background with its flip is generating 20000 + 20000 = 40000 images.

Similarly 100 background images with their corresponding flip will generate 400K images.

These 400K images are saved one-by-one with naming convention like Img_fg_bg_20217.jpg in a colab folder.

While overlaying, random positions are generated in such a way to ensure that foreground object remains within the background frame. We achieved this by calculating delta between width of BG & width of FG, delta between height of BG & height of FG and generated random positions between low as 0 and high as delta values. This ensured FG images remain within BG canvas like below.

![positions](https://github.com/mmaruthi/EVA4P1_S15A_Depth_FG_Detection_DataPrep/blob/master/Images_For_ReadMe/Random_Positions.png)

Saved colab folder is zipped and then copied to gdrive location.

Sample FG_BG 

![Sample FG_BG](https://github.com/mmaruthi/EVA4P1_S15A_Depth_FG_Detection_DataPrep/blob/master/Images_For_ReadMe/FG_BG_Sample10.png)

FG_BG Mask preparation – Preparing mask of FG from FG_BG images (400K images)
Code : https://github.com/anilbhatt1/EVA4P1_S15A_Depth_FG_Detection/blob/master/EVA4P1_S15_DataPrep_V1.ipynb

Mask of FG_BG is prepared along FG_BG preparation and written in a separate colab folder.

As followed for FG_BG, this colab folder is zipped and copied to gdrive.

Foreground image that we are going to overlay over background is converted to gray scale. FG image already have a transparent background with object in it.

Hence all those pixels which represent object will have a pixel value greater than zero. We will make these pixels 255 so they will be bright and rest all as zero.

Next we will convert background image to gray scale. Here we will make all those pixels that are > 0 to 0. Result will be background turning dark.

Next we will overlay converted foreground on top of converted background.

Result will be a white mask of foreground on top of dark background.

FG_BG Mask generated is as below

![FG_BG_Mask](https://github.com/mmaruthi/EVA4P1_S15A_Depth_FG_Detection_DataPrep/blob/master/Images_For_ReadMe/FG_BG_Mask_Sample10.png)

FG_BG Depth Creation
Code : https://github.com/mmaruthi/EVA4P1_S15A_Depth_FG_Detection/blob/master/EVA4P1_S15_DepthCreation_V1.ipynb

We are taking Dense Depth model pre-trained on NYU dataset. This dataset is having similar background as chosen for FG_BG images.
FG_BG images are passed on to DenseDepth model, resized to Grayscale 200x200 , stored in colab folder.

This colab folder is zipped and copied to gdrive.

FG_BG Depth generated is as below:

![FG_BG_Depth](https://github.com/mmaruthi/EVA4P1_S15A_Depth_FG_Detection_DataPrep/blob/master/Images_For_ReadMe/FG_BG_Depth_Sample10.png)



