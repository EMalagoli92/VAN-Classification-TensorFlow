# VAN-Classification-TensorFlow
TensorFlow 2.X reimplementation of [Visual Attention Network](https://arxiv.org/abs/2202.09741v5), Meng-Hao Guo, Cheng-Ze Lu, Zheng-Ning Liu, Ming-Ming Cheng, Shi-Min Hu.
- Exact TensorFlow reimplementation of official PyTorch repo, including `timm` modules used by authors, preserving models and layers structure.
- ImageNet pretrained weights ported from PyTorch official implementation.

## Table of contents
- [Abstract](#abstract)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgement](#acknowledgement)
- [Citations](#citations)
- [License](#license)

<div id="abstract"/>

## Abstract
*While originally designed for natural language processing (NLP) tasks, the self-attention mechanism has recently taken various computer vision areas by storm. However, the 2D nature of images brings three challenges for applying self-attention in computer vision. (1) Treating images as 1D sequences neglects their 2D structures. (2) The quadratic complexity is too expensive for high-resolution images. (3) It only captures spatial adaptability but ignores channel adaptability. In this paper, the authors propose a novel large kernel attention (LKA) module to enable self-adaptive and long-range correlations in self-attention while avoiding the above issues. The authors further introduce a novel neural network based on LKA, namely Visual Attention Network (VAN). While extremely simple and efficient, VAN outperforms the state-of-the-art vision transformers (ViTs) and convolutional neural networks (CNNs) with a large margin in extensive experiments, including image classification, object detection, semantic segmentation, instance segmentation, etc.*


![Alt text](https://github.com/EMalagoli92/VAN-Classification-TensorFlow/blob/main/assets/images/Comparsion.png?raw=true) 
<p align = "center"><sub>Compare with different vision backbones on ImageNet-1K validation set.</sub></p>


![Alt text](https://github.com/EMalagoli92/VAN-Classification-TensorFlow/blob/main/assets/images/decomposition.png?raw=true)
<p align = "center"><sub>Decomposition diagram of large-kernel convolution. A standard convolution can be decomposed into three parts: a depth-wise convolution (DW-Conv), a depth-wise dilation convolution (DW-D-Conv) and a 1×1 convolution (1×1 Conv).</sub></p>


![Alt text](https://github.com/EMalagoli92/VAN-Classification-TensorFlow/blob/main/assets/images/LKA.png?raw=true)
<p align = "center"><sub>The structure of different modules: (a) the proposed Large Kernel Attention (LKA); (b) non-attention module; (c) the self-attention module (d) a stage of our Visual Attention Network (VAN). CFF means convolutional feed-forward network. The difference between (a) and (b) is the element-wise multiply. It is worth noting that (c) is designed for 1D sequences.</sub></p>


<div id="results"/>

## Results
TensorFlow implementation and ImageNet ported weights have been compared to the official PyTorch implementation on [ImageNet-V2](https://www.tensorflow.org/datasets/catalog/imagenet_v2) test set.

### Models pre-trained on ImageNet-1K
| Configuration  | Resolution | Top-1 (Original) | Top-1 (Ported) | Top-5 (Original) | Top-5 (Ported) | #Params
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| VAN-B0 | 224x224 | 0.59 | 0.59 | 0.81 | 0.81 | 4.1M |
| VAN-B1 | 224x224 | 0.64 | 0.64 | 0.84 | 0.84 | 13.9M |
| VAN-B2 | 224x224 | 0.69 | 0.69 | 0.88 | 0.88 | 26.6M |
| VAN-B3 | 224x224 | 0.71 | 0.71 | 0.89 | 0.89 | 44.8M |

Metrics difference: `0`.


<div id="installation"/>

## Installation

<div id="usage"/>

## Usage

<div id="acknowledgement"/>

## Acknowledgement
- [VAN-Classification](https://github.com/Visual-Attention-Network/VAN-Classification) (Official PyTorch implementation)


<div id="citations"/>

## Citations
```bibtex
@article{guo2022visual,
  title={Visual Attention Network},
  author={Guo, Meng-Hao and Lu, Cheng-Ze and Liu, Zheng-Ning and Cheng, Ming-Ming and Hu, Shi-Min},
  journal={arXiv preprint arXiv:2202.09741},
  year={2022}
}
```


<div id="license"/>

## License
