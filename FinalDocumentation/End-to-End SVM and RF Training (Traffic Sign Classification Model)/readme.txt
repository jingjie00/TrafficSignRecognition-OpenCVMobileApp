Done by: Ng Jan Hui & Tan Wei Mun

*If you wish to skip segmentation and feature extraction, look at step 4

1. Copy the traffic sign dataset images and place into the "Input Dataset" folder

2. You may specify the path of the images to be trained by editing the "testInputs.txt". The correct way to specify the path is, e.g. "InputDataset/abc.png" if the image you wish to classify is abc.png. 

However doing so you may need to recompile the cpp file, as you need to edit the map data structure on line 350, you need to specify the 2nd and 3rd characters of the name of the images you wish to train. For example, if you want to train images that names start with "015", then you need to add a encoding entry in the map by sepcifying a pair as such, {"15", n+1}, where n is the largest encoded label in that map.

3. The program will loop through every image to segment and extract features from it.

===

4. Alternatively, we have conveniently extracted features of 6 traffic sign types from a sample of 1045 images. The features have been saved into a CSV file.

4a. If you wish to train your own set of images, the features extracted will also be saved to CSV file for you, in case you wish to retrain the models, it can be directly done by reading the features CSV without going thru segmentation and feature extraction again.

5. The model will be trained based on the features of the CSV file, and will be saved into the same directory as the .exe file. 2 models will be generated which are the Random Forest model and SVM model.

===

6. If you found missing values in your CSV generated, you may use the imputation python file that will conveniently impute your missing data. Follow the readme file of that folder.

===

7. After training, the labels predicted by the SVM, RF along with the ground truth labels, will be exported to a CSV file as well. These 3 files can be used in the classification analysis python script. Follow the readme file of that folder