# Model compression via Pruning and Quantization on NVIDIA Jetson Nano

[Paper](https://drive.google.com/file/d/1xVQjVuJvFmD0Yktru4ozkDjvm9NZkePI/view?usp=drive_link)

**This repository contains source code for model compression techniques on the NVIDIA Jetson Nano, focusing on pruning and quantization.**

## Dataset & Model
Given the resource constraints of the Jetson Nano, the following dataset and model were selected:
- Dataset: FashionMNIST
- Model  : LeNet-5


## Model Compression techniques
- Pruning 
- Quantization


## Code
### Install dependencies
```
python -m pip install -r requirements.txt
```

### Experimentation
All experimental code can be found in `experiment.ipynb`. This notebook contains step-by-step implementations of pruning and quantization techniques on the selected model and dataset.
