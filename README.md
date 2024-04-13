# Mini-Project 1: Residual Network Design 
*Authors:* Alison Reed, David Glaser, Naman Soni

**Summary:** We implemented a modified ResNet architecture which classifies CIFAR-10 images with high accuracy. Our implementation modifies the EfficientNet architecture, which makes use of compound dimension coefficients in order to effectively increase classification accuracy with convolutional network scaling. EfficientNet is also relatively lightweight, allowing us to efficiently classify images with under 5 million parameters.

This repository has been inspired by [Efficient Resnets](https://github.com/Nikunj-Gupta/Efficient_ResNets)

# Introduction 
ResNets (or Residual Networks) are one of the most commonly used models for image classification
5 tasks. In this project, you will design and train your own ResNet model for CIFAR-10 image
6 classification. In particular, your goal will be to maximize accuracy on the CIFAR-10 benchmark
7 while keeping the size of your ResNet model under budget. Model size, typically measured as the
8 number or trainable parameters, is important when models need to be stored on devices with limited
9 storage capacity, mobile devices for example. 

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

To install all the dependencies, execute: `pip install -r requirements.txt`

# Description of files/folders in the repository 
- best_model : The best model event logs and checkpoint file
- configs : configs for testing
- predictions : The csv file with the predictions
- CustomDataset.py : To Create a dataset with the unseen testing data
- models/resnet.py : PyTorch description of ResNet model architecture (flexible to change/modify using config.yaml) 
- main.py : code to train and test ResNet architectures 
- config.yaml : contains the hyperparamters used for constructing and training a ResNet architecture 
- project1_model.py : ResNet architecture used.
- load_model.py : For loading the model from the checkpoint
- testing.py : For testing the model on the competition dataset and generating the csv

# Training
Training can be started with the following command
```
python3 main.py  
```
To modify and test with new ResNet architectures, you can create a new configuration experiment in project_model.py directly.

# Reproduce the results 

#### Train our best modified ResNet Architecture with: 
We have set the above as our default inputs in `main.py` and hence the following will reproduce our results too:
```
python3 main.py 
```
# Testing
To test the unseen data run the following command after loading you checkpoint in the best_model folder:
```
python3 testing.py 
```
The output will be saved as predicitions.csv in the predictions folder.


# Visualizing the logs
TO VISUALIZE TFEVENTS LOGS:
1. copy events.out.tfevent..... file to local
2. pip install tensorflow tensorboard
3. from CLI run tensorboard --logdir=[directory that the tfevents file IS IN (not the directory to the file itself)]
4. go to http://localhost:6006/ in browser
