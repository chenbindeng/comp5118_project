# comp5118_project
This is the project for course COMP 5118: Trends in Big Data Management

# Objective
This project is to continue the investigation on the artical "Power of Deep Learning for Channel Estimation
and Signal Detection in OFDM Systems"

# Requirements
This project has been test with GPU support:
   - Python 3.7.9
   - Tensorflow 1.15.0
   - Keras 2.3.1

# Dataset
Please download this dataset from Google driver [download url](https://drive.google.com/drive/folders/1pwjEzmLZIybk3SWNAwo6hmzmUnd5Sgsf?usp=sharing) 
  - Training Dataset -- channel_train.npy (which has 1,000,000 samples for training)
  - Test Dataset     -- channel_test.npy  (which has 390,000 samples for teting)

# How to Run
After downloading the datasets, and put them to ./data folder, thenit is good to run them seperately:
  - mlp.py ----- to train and verify MLP model
  - cnn.py ----- to train and verify CNN model
  - rnn.py ----- to train and verify RNN model
