# Cream of the Crop: Distilling Prioritized Paths For One-Shot Neural Architecture Search

## Requirements
* python >= 3.6
* torch >= 1.2
* torchscope
* apex (not necessary, please make sure your nvcc CUDA version is the same with pytorch CUDA verision)

## Data Preparation 
You need to first download the [ImageNet-2012](http://www.image-net.org/) to the folder `./data/imagenet` and move the validation set to the subfolder `./data/imagenet/val`. To move the validation set, you cloud use the following script: <https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh> 

Put the imagenet data in ${Root}/data. It should be like following:
```buildoutcfg
${Root}/data/imagenet/train
${Root}/data/imagenet/val
...
```

## Environment Setup

To set up the enviroment you can easily run the following command:
```buildoutcfg
python setup.py
```

## Checkpoints
For quick test, we have stored the checkpoints of our models in [google drive](https://drive.google.com/drive/folders/1CQjyBryZ4F20Rutj7coF8HWFcedApUn2?usp=sharing).

Just download the checkpoints from [google drive](https://drive.google.com/drive/folders/1CQjyBryZ4F20Rutj7coF8HWFcedApUn2?usp=sharing) and put the checkpoints in `${ROOT}/experiments/ckps/`.

## Quick Start

### I. Search
To search for an architecture, you need to configure the parameters `flops_minimum` and `flops_maximum` to specify the desired model flops, such as [0,600]MB flops. You can specify the flops interval by changing these two parameters in `./experiments/scripts/supernet.sh`.
```buildoutcfg
--flops_minimum 0 # Minimum Flops of Architecture
--flops_maximum 600 # Maximum Flops of Architecture
```

After you specify the flops of the architectures you would like to search, you can search an architecture now by running:
```buildoutcfg
sh ./experiments/scripts/supernet.sh
```

### II. Retrain 
We also give the architecture we searched. To train those architectures, you need to configure the parameter `model_selection` to specify the model flops. To specify which model to train, you should add `--model_seclection` in `./experiments/scripts/train.sh`. You can select one from [14,42,112,285,470,600], which stands for different flops(MB).
```buildoutcfg
--model_selection 42 # Retrain 42m model
--model_selection 470 # Retrain 470m model
......
```

After specifying the flops, you need to choose the config settings in `./experiments/scripts/train.sh`. The config files are in `./lib/configs`
```buildoutcfg
--config ./lib/configs/42.yaml
--config ./lib/configs/470.yaml
......
```

After adding `model_selection` in `train.sh`, you need to use the following command to train the model.
```buildoutcfg
sh ./experiments/scripts/train.sh
```

The trained model and log file will be saved in `./retrain`. You can configure the `--ouput` in `./experiments/scripts/train.sh` to specify a path for saving the model and log file.

### III. Test
To test our trained of models, you need to use `model_selection` in `./experiments/scripts/test.sh` to specify which model to test.
```buildoutcfg
--model_selection 42 # test 42m model
--model_selection 470 # test 470m model
......
```

After specifying the flops of the model, you need to write the path to the resume model in `./experiments/scripts/test.sh`.
```buildoutcfg
--resume './experiments/ckps/42.pth.tar'
--resume './experiments/ckps/470.pth.tar'
......
```

After adding `--model_selection` and `--resume` in './experiments/scripts/test.sh', you need to use the following command to test the model.
```buildoutcfg
sh ./experiments/scripts/test.sh
```

The test result will be saved in `./retrain`. You can configure the `--ouput` in `./experiments/scripts/test.sh` to specify a path for it.

##### Test Rank Correlation 

To perform a correlation analysis, we randomly sample 30 subnetworks from the hypernetwork and calculate the rank correlation between the weight sharing performance and the true performance of training from scratch. Unfortunately, training these subnetworks on ImageNet is very computational expensive, we thus construct a subImageNet dataset, which only consists of 100 classes randomly sampled from ImageNet. Each class has 250 training images and 50 validation images. We can generate the imagelist by running the following script: 

```buildoutcfg
python ./experiments/scripts/generate_subImageNet.py
```
Thus we get the subImageNet in `./data/subImagnet`. The class list is provided in `./data/subclass_list.txt`. The images list is provided in `./data/subimages_list.txt`

