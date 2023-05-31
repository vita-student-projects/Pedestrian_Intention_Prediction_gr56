# Pedestrian Intention Prediction (based on [MotionBert](https://github.com/Walter0807/MotionBERT))

<p align="center">
    <img src="images/example2.gif" width="70%" height="50%"/>
</p>

## Table of Contents :

1. [Introduction](#1-introduction)
2. [Installation](#2-installation)
3. [Training Dataset Generation](#3-training-dataset-generation)
4. [Training](#4-training)
5. [Evaluation](#5-evaluation)
6. [Inference](#6-inference)

## 1. Introduction

This project propose a model to predict the intention of pedestrians in a video. The model is based on the [MotionBert](https://github.com/Walter0807/MotionBERT) model and is trained and evaluated on the [JAAD dataset](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/). For more information about the model and its performances, please refer to the [report.md](https://github.com/Yseoo/Pedestrian-Intention-Predicition/tree/main/report.md) documentation.

This project was made in the context of the [CIVIL-459:Deep Learning for Autonomous Vehicles](https://edu.epfl.ch/coursebook/en/deep-learning-for-autonomous-vehicles-CIVIL-459) course at EPFL. The goal was to solve a task related to autonomous vehicles using deep learning that bring a contribution to the state of the art in order in the end to reproduce the autopilot of an Autonomous Vehicle.

## 2. Installation

To install the project and be able to run it, you need to follow the following steps:

1. Clone the repository:
2. Install the requirements:

```bash
conda create -n mbpip python=3.7 anaconda
conda activate mbpip
# Please install PyTorch according to your CUDA version.
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```
3. Run the script init.sh :
   
```bash
chmod +x init.sh
./init.sh
```

4. If you want to use a already trained model, download the checkpoint from [here](https://drive.google.com/drive/folders/1fOxR13Tp8Jm9EeOku-FGu5fD3UfqgWOp?usp=sharing) and put it in the `checkpoints` folder.

## 3. Training Dataset Generation

The already generated dataset can be downloaded from [here](https://drive.google.com/drive/folders/1fOxR13Tp8Jm9EeOku-FGu5fD3UfqgWOp?usp=sharing). To generate the dataset yourself, execute the following command:

```
python3 dataset.py --data_path=<folder_path> --compute_kps --regen
```
- `--data_path` helps to specify the folder path if different from the current one
- `--compute_kps` flag to compute keypoints with bounding boxes. Only the boundingbox will be included in the output pickle file if the flag is omitted.
- `--regen` flag to regenerate the database

The output of the code is a pickle file *jaad_database.pkl* containing a dictionary with the following structure :
``` 
'annotations': 
    'vid_id'(str): 
        'num_frames':          int
        'width':               int
        'height':              int
        'ped_annotations':     list (dict)
            'ped_id'(str):         list (dict)
                'old_id':              str
                'frames':              list (int)
                'occlusion':           list (int)
                'bbox':                list ([x1 (float), y1 (float), x2 (float), y2 (float)])
                '2dkp':                list (array(array))
                'cross':               int
'split': 
    'train_ID':     list (str)
    'test_ID':      list (str)
'ckpt': str
'seq_per_vid': list (int)
```

## 4. Training

In this part we will explain how to train the model on the JAAD dataset.

1. Follow the procedure in the Training Dataset Generation part above.
2. Create a config file in the `config` folder. You can use the `config/JAAD_train.yaml` file as a template.
3. Fill the config file with the correct paths to the dataset and the correct wanted parameters.
4. Run the following command:

#### From scracth :
```bash
python train.py --config config/<your_config_file>.yaml -f <print_frequency>
```
#### From a checkpoint :
```bash
python train.py --config config/<your_config_file>.yaml -f <print_frequency> -c
```

### Visualize logs with Tensorboard

The code will automatically create a folder named `logs` in the root directory of the project. You can visualize the logs with Tensorboard by running the following command:

```bash
tensorboard --logdir=logs/
```

## 5. Evaluation

In this part we will explain how to evaluate the model on the JAAD dataset.

1. Follow the procedure of the part [Training Dataset Creation](#training-dataset-creation), or download the already generated database.
2. Create a config file in the `config` folder. You can use the `config/JAAD_eval.yaml` file as a template.
3. Fill the config file with the correct paths to the dataset and the correct wanted parameters (make sure that it is the same config file as the one used for training the model you want to evaluate).
4. Run the following command:

```bash
python train.py --config config/<your_config_file>.yaml -f <print_frequency> -e
```

## 6. Inference

In this part we will explain how to use the model to predict the intention of pedestrians on a video.

#### From a video, using OpenPifPaf for 2D keypoints extraction

1. Download the video you want to use for the inference in the `datagen/infer_DB/infer_clips' folder.
2. Create a config file in the `config` folder. You can use the `config/inference.yaml` file as a template.
3. Fill the config file with the correct paths to the model you want to use and the correct corresponding parameters.
4. Run the following command:

```bash
python inference.py --config config/<your_config_file>.yaml --data_path datagen/infer_DB/infer_clips/ --filename <your_video_name>
```

#### From a pickle file (with appropriate format, see bellow):

1. Make sure you have a proper pickle file with the right format (see the Inference data format bellow).
2. Create a config file in the `config` folder. You can use the `config/inference.yaml` file as a template.
3. Fill the config file with the correct paths to the model you want to use and the correct corresponding parameters.
4. Run the following command:

```bash
python inference_wo_gen.py --config config/<your_config_file>.yaml --data_path <your_pickle_file_path>
```
##### Inference data format :

```
'vid_id':           str
'num_seq':          int
'forecast_step':    int
'nbr_frame_seq':    int
'total_frame_vid':  int
'width':            int
'height':           int
'per_seq_ped':      list (list (int))
'ped_annotations':  list (dict) 
    'frames':           list (int)
    'occlusion':        list (int)
    'bbox':             list ([x1 (float), y1 (float), x2 (float), y2 (float)])
    '2dkp':             list (array(array))
```


## Acknowledgment

We acknowledge and appreciate the contributions of Alexandre Alahi and Saeed Saadatnejad to our Pedestrian Intention Prediction project. Their expertise and guidance have significantly impacted our research, and we are grateful for their valuable support.
