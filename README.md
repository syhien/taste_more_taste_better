# Taste More Taste Better

## Requirements

- Python 3 (Latest version recommended)
- CUDA 11.8 or higher
- [VMamba](https://github.com/MzeroMiko/VMamba?tab=readme-ov-file#installation)

## Parameters

Detailed parameters can be found in the [paper](https://arxiv.org/html/2503.17984v1#S4). 

[train.py](train.py) has set the default parameters.

## Inpainter

We use [Fooocus](https://github.com/lllyasviel/Fooocus) as the inpainter. However, sadly, Fooocus is being deprecated and will not get any new features. We recommend using some other more modern inpainting models, and making your own implementation in the [inpainter.py](inpainter.py) file.
