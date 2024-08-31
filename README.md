# Intro-to-deep-learning-final-project
This repository contains all of our code in the final project of the course. This project contains 3 classification models (CNN, ViT, and a hybrid of the two), along with an ensembeling script.
The script "ResConvTrans_Cifar100.py" contains the definition, training, and results plotting of the hybrid model (residual convolutional network that feeds into a vision transformer)
The script "CNNModel.py" contains the definition, training, and results plotting of the CNN.
The script "ViT_model.py" contains the definition, training, and results plotting of the ViT
The script "Model_classes_for_ensemble.py" contains all the model classes for the ensemble. This exists since when we tried to import the models directly from their original scripts, some of the started the training loop for some reason...
The script "ensemble.py" loads the models and wheights, and calculates the mean output of the 3 models on the test dataset to make a final decision. 
