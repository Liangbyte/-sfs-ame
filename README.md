# sfs-ame
Code for Enhancing Deepfake Detection Generalization through Spatial-Frequency Synergy and Adaptive Mixture-of-Experts.The paper is currently being submitted, and we will publish the detailed structure after submission.This code is directly related to the manuscript currently submitted to The Visual Computer.
If you use this repository or find it helpful, please cite the corresponding paper once it becomes available.
# Project Name  
Enhancing Deepfake Detection Generalization through Spatial-Frequency Synergy and Adaptive Mixture-of-Experts.
# Introduction
We propose a novel detection framework that integrates Discrete Wavelet Transform (DWT) and Vision Transformer (ViT) to extract discriminative frequency-domain and spatial-domain features, thereby significantly improving the performance of facial tampering detection.
# Installation
pip install -r requirements.txt
# dateset
The datasets used can be obtained from DeepfakeBench.
# Training and Testing the Model
Ensure that you have correctly set the path to the dataset and the model parameters.
# Train and Evaluation
First, use train.py to train your model, then use eval.py to load the weights of the best model for testing.
