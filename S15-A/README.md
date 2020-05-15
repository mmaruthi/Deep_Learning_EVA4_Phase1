S15A - Data Preperation for Depth and FG Detection Network (Jointly Done by Anilkumar N Bhatt and Maruthi Srinivas):

------------------------------------

Below are the images created as part of data preparation

Gdrive Location : https://drive.google.com/drive/folders/1raMnribL-gsa4FEpX8QIeyR6yP4XgmP-?usp=sharing
Total Size : 4.9 GB
100 BG and its corresponding 100 Flip Images, Shape : 192X192X3, Type : jpg, Folder Name : BG_and_Its Flip Images
100 FG images, Shape : 192x192x4, Type : png, transparent background, Folder Name : FG_Images
400K FG_BG images, Shape : 192x192x3, Type : jpg, Zip File Name : FG_BG_400K.zip
Mean : [0.56670278 0.49779153 0.43632878], Std-Dev : [0.38389994 0.30871084 0.25551239], Size : 3 GB
Corresponding 400K FG_BG_Masks, Shape : 192x192x1, Type : jpg, Zip File Name : FG_BG_Mask_400K.zip
Mean : [0.20249742], Std-Dev : [0.39961225], Size : 906 MB
400K Depth images of FG_BG, Shape : 200x200x1, Type : jpg, Zip File Name : FG_BG_Depth_400K.zip
Mean : [0.32939295], Std-Dev : [0.24930712], Size : 1 GB
Log files corresponding to above three zip files which have file names, image size of BG, image size of FG and bounding box coordinates of overlaid FG image.
FG_BG_Filename_Logs.txt, FG_BG_Mask_Filename_Logs.txt, Log_FG_BG_Depth_400K.txt
Code for statistics calculation : https://github.com/anilbhatt1/EVA4P1_S15A_Depth_FG_Detection_DataPrep/blob/master/Statistics_FG_BG_Mask_Depth.ipynb

Data preperation details are as listed below
This part deals with data preparation which will be later used by a network that will predict foreground from background and how far foreground is from camera w.r.to background (depth).
Since data required for training this network is not publicly available & crowdsourcing is also not possible, data preparation strategy as follows were adopted.
Downloaded 100 background images from web. Images of public places mostly malls & shopping complexes were downloaded. Resized them to 192x192
Downloaded 100 foreground images. Images of people were selected. Removed the background using Microsoft power point using 'Remove Background' option thereby adding transparent layer. After that cropped this image using 'Crop' option under 'Format' tab in PPT to select object only. Then saved this image in 'png' format so that transparency (alpha channel) is retained.
Flipped the 100 background images we created in step 3 and saved it. This makes total 200 background images (100 – Regular, 100 – Flipped) all in jpg format
Sample BG images

![Sample BG Images](https://github.com/mmaruthi/EVA4P1_S15A_Depth_FG_Detection_DataPrep/blob/master/Images_For_ReadMe/BG_Sample10.png)

