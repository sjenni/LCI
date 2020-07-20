## Steering Self-Supervised Feature Learning Beyond Local Pixel Statistics [[Project Page]](https://sjenni.github.io/LCI/) 

[Simon Jenni](https://sjenni.github.io), [Hailin Jin](https://sites.google.com/view/hailinjin), and [Paolo Favaro](http://www.cvg.unibe.ch/people/favaro).  
In [CVPR](https://arxiv.org/abs/2004.02331), 2020.

![Model](https://sjenni.github.io/LCI/assets/model_LCI.png)


This repository contains code for self-supervised pre-training and supervised transfer learning on the STL-10 dataset.

***Training and evaluation on ImageNet is coming soon!***

## Requirements
The code is based on Python 3.7 and tensorflow 1.15. 

## How to use it

### 1. Setup

- Set the paths to the data and log directories in [constants.py](constants.py).
- Run [init_datasets.py](init_datasets.py) to download and convert the STL-10 dataset to the TFRecord format:
```
python init_datasets.py
```

### 2. Training and evaluation 

- To train and evaluate a transformation classifier on STL-10 execute [run_stl10.py](run_stl10.py). An example usage could look like this: 
```
python run_stl10.py --tag='test' --num_gpus=1
```

## Citation

If you find this repository useful for your research, please use the following.

```
@inproceedings{jenni2020steering,
  title={Steering Self-Supervised Feature Learning Beyond Local Pixel Statistics},
  author={Jenni, Simon and Jin, Hailin and Favaro, Paolo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6408--6417},
  year={2020}
}
```
