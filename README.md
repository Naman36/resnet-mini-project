# Mini-Project 1: Residual Network Design 
**Authors:** Alison Reed, David Glaser, Naman Soni

**Summary:** We implemented a modified ResNet architecture which classifies CIFAR-10 images with high accuracy. Our implementation modifies the EfficientNet architecture, which makes use of compound dimension coefficients in order to effectively increase classification accuracy with convolutional network scaling. EfficientNet is also relatively lightweight, allowing us to efficiently classify images with under 5 million parameters.

This repository is inspired by [Efficient Resnets](https://github.com/Nikunj-Gupta/Efficient_ResNets)

# Introduction 
ResNets (or Residual Networks) are one of the most commonly used models for image classification tasks. In this project, you will design and train your own ResNet model for CIFAR-10 image classification. In particular, your goal will be to maximize accuracy on the CIFAR-10 benchmark while keeping the size of your ResNet model under budget. Model size, typically measured as the number or trainable parameters, is important when models need to be stored on devices with limited storage capacity, mobile devices for example. 

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

To install all the dependencies, execute: `pip install -r requirements.txt`

# Description of files/folders in the repository 
- best_model : The best model event logs and checkpoint file
- predictions : The csv file with the predictions
- CustomDataset.py : To Create a dataset with the unseen testing data
- main.py : code to train and test ResNet architectures 
- config.yaml : contains the hyperparamters used for constructing and training a ResNet architecture 
- project1_model.py : ResNet architecture used (flexible to change/modify using config.yaml)
- load_model.py : For loading the model from the checkpoint
- test.py : For testing the model on the competition dataset and generating the csv
- validation.ipynb : Notebook for validating the trained model's output predictions and accuracy (including plots)

# Training
Training can be started with the following command
```
python3 main.py  --config config.yaml --resnet_architecture best_model
```
To modify and test with new ResNet architectures, you can create a new configuration experiment in project1_model.py directly.

# Testing
To test the unseen data run the following command after loading the checkpoint in the best_model folder:
```
python3 test.py 
```
The output will be saved as predicitions.csv in the predictions folder.

# Validation
Validation of the trained model's output predictions and accuracy can be seen in validation.ipynb.

# Visualizing the logs
TO VISUALIZE TFEVENTS LOGS:
1. copy events.out.tfevent..... file to local
2. pip install tensorflow tensorboard
3. from CLI run tensorboard --logdir=[directory that the tfevents file IS IN (not the directory to the file itself)]
4. go to http://localhost:6006/ in browser
