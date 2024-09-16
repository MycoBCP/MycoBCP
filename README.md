Supplementary code for "Bacterial Cytological Profiling and Convolutional Neural Networks as Tools to Determine Antimicrobial Mechanisms in _Mycobacterium tuberculosis_"

dv_preprocessing.py
-reads microscope images from deltavision microscopes (.dv files) and processes them for use in training or testing

bsl2_network_train.py
-sets up and trains the convolutional neural network based on a library of images set up in "BSL2_Training_Data"
-access to raw data and image is not provided
-model_save_name.ckpt is provided in lieu of images to run testing

bsl2_network_test.py
-takes preprocessed dv files as input and outputs a csv with the feature vector as described in the manuscript

BSL2_Test_Run_XX_Full/BSL2_Test_Run_XX_Full_output
-Contains a few example images and resulting output
