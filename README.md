# Steering Self-Supervised Feature Learning Beyond Local Pixel Statistics [[Project Page]](https://sjenni.github.io/LCI/) 

This repository contains demo code of our CVPR2020 [paper](https://arxiv.org/abs/2004.02331). 
It contains code for the training and evaluation on the STL-10 dataset. 

*Training and evaluation on ImageNet is coming soon!*

## Requirements
The code is based on Python 3.7 and tensorflow 1.15.

## How to use it

### 1. Setup

- Set the paths to the data and log directories in **constants.py**.
- Run **init_datasets.py** to download and convert the STL-10 dataset.

### 2. Training and evaluation 

- To train and evaluate a transformation classifier on STL-10 run **run_stl10.py**.

