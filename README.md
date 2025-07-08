# Int*-Match: Balancing Intra-Class Compactness and Inter-Class Discrepancy for Semi-Supervised Speaker Recognition (AAAI 2025)
This repository contains the code for our paper: "Int*-Match: Balancing Intra-Class Compactness and Inter-Class Discrepancy for Semi-Supervised Speaker Recognition". We propose a novel method to control the variation of the inter-class confidence threshold, thereby improving the performance of speaker recognition.

# Data preparation

Please follow the instructions in the 'Data preparation' section of [this repository](https://github.com/TaoRuijie/ECAPA-TDNN) or the official resource [voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer) to prepare your VoxCeleb2 dataset using the official code

# Training
Before training, please change the data path and save path in the ```main.py```.

# Acknowledge

We would like to express our sincere gratitude to the following teams and individuals for their open-source contributions, which have greatly assisted our work:

1. [TorchSSL](https://github.com/TorchSSL/TorchSSL) and [USB](https://github.com/microsoft/Semi-supervised-learning) for helping us better understand semi-supervised learning algorithms.
2. [ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN) by Tao, which has been instrumental in building our speaker recognition system.
