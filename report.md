# Project - Pedestrian Intention Predicition 

*<p style="text-align: center;">This project was made in the context of the [CIVIL-459:Deep Learning for Autonomous Vehicles](https://edu.epfl.ch/coursebook/en/deep-learning-for-autonomous-vehicles-CIVIL-459) course at EPFL. The goal was to solve a task related to autonomous vehicles using deep learning that bring a contribution to the state of the art in order in the end to reproduce the autopilot of an Autonomous Vehicle.</p>*

## Contents

1. [Project Motivation & Objectives](#1---project-motivation--objectives)
2. [Contribution Overview](#2---contribution-overview)
    1. [Dataset Creation](#21---dataset-creation)
    2. [Model Modification and Adaptation for Training & Testing](#22---model-modification-and-adaptation-for-training--testing)
    3. [Inference Data Creation & Model Prediction](#23---inference-data-creation--model-prediction)
3. [Description of the Data](#3---description-of-the-data)
    1. [JAAD Dataset](#31---jaad-dataset)
    2. [Inference Data](#32---inference-data)
4. [Experimental Setup and Results](#4---experimental-setup-and-results)
    1. [Experimental Setup](#41---experiments-setup)
        1. [Metrics](#411---metrics)
        2. [Overfitting](#412---overfitting)
        3. [1. Bounding box only](#1-bounding-box-only)
        4. [2. Regularization + Less complex model](#2-regularization--less-complex-model)
        5. [3. Keypoints](#3-keypoints)
        6. [4. Bounding box, occlusion + keypoints + 1s prediction, 1s seqence](#4-bounding-box-occlusion--keypoints--1s-prediction-1s-seqence)
        7. [5. Bounding box, occlusion + keypoints + 1s prediction, 1s seqence + dynamic detection to identify the pedestrian](#5-bounding-box-occlusion--keypoints--1s-prediction-1s-seqence--dynamic-detection-to-identify-the-pedestrian)
    2. [Results](#results)
        1. [Overfitting](#overfitting)
        2. [1. Bounding box only](#1-bounding-box-only-1)
        3. [2. Regularization + Less complex model](#2-regularization--less-complex-model-1)
        4. [3. keypoints](#3-keypoints-1)
        5. [4. Bounding box, occlusion + keypoints + 1s prediction, 1s seqence](#4-bounding-box-occlusion--keypoints--1s-prediction-1s-seqence-1)
        6. [5. Bounding box, occlusion + keypoints + 1s prediction, 1s seqence + dynamic detection to identify the pedestrian](#5-bounding-box-occlusion--keypoints--1s-prediction-1s-seqence--dynamic-detection-to-identify-the-pedestrian-1)
        7. [Best Checkpoint Result Comparison](#best-checkpoint-result-comparison)
5. [Conclusion](#conclusion)



## 1 - Project Motivation & Objectives

&emsp; In this project was implemented a model predicting the crossing intention of pedestrians in the context of autonomous driving. Intention predicting models in autonomous driving are important for both safety and efficiency. It helps autonomous vehicles anticipate pedestrian behavior, adjust their actions accordingly, and reduce the risk of accidents. It can also improves traffic flow by allowing vehicles to plan maneuvers in advance, avoiding sudden stops or delays.

&emsp;This report describes the problem, the project, and the experimental results in details.
## 2 - Contribution Overview

<p align="center">
    <img src="./images/Network_Architecture.jpg" width="80%" height="80%"/>
</p>

&emsp; To solve the problem we're assigned, we choose to base our model on the [MotionBert](https://github.com/Walter0807/MotionBERT), a current state of the art model for 3D pose estimation and action recognition. While some of those tasks are not directly related to our problem, we believe that the part of the model implemented for action recognition could be adapted to our problem. In this project, our contribution is to adatpt the already existing MotionBert model to take as input 2D keyposes and bounding boxes, and to predict Crossing/Not-Crossing.

### 2.1 - Dataset Creation

<p align="center">
    <img src="./images/Dataset_Creation.jpg" width="30%" height="30%"/>
</p>


&emsp; Our Dataset is based on the JAAD Database, a video Database with annotations often used to train Deep Learning models in the context of autonomous driving. The motivations for choosing such a Database will be further developped in the next section. To get started, a first step was to understand the structure of the Database, and to get familiar with the annotations. As our model needs sequences of specific data only from relevant pedestrians, a new Dataset was created to best fit our needs. Most of the annotations provided by the Database were discarded only to keep a few such as bounding boxes and occlusion. The Dataset was then organized in sequences of 1 seconds, overlapping each other by 0.5 second to best capture the motion of the pedestrian. Since the JAAD model provides a label Crossing/Not-Crossing for each frame, a new label per sequence was created and assigned to the sequence if at least one frame of the sequence was labeled as Crossing. 

&emsp; After training and testing the model only with the bounding boxes, it was realized that we would need much more data to get better, and more accurate results. To this end, keypoints of the pedestrians generated by OpenPifPaf for each JAAD video were added to the Dataset. In order to do so, and since the kypoints generation was a long process, a checkpoint file system was implemented to save the new keypoints after each video analyzed. In order to correspond the keypoints to the right JAAD bounding box, a cross check had to be implemented for each frame analized to make sure that the keypoints corresponded to the right bounding box.

&emsp; To better analyse the newly created Dataset, and the results of our model, specific functions were created to visualize the Dataset, and to get statistics on it. The Dataset was then split into a training set and a testing set, while making sure that all the sequences corresponding to the same video were in the same set to avoid prediction in the testing set of training data which would bias ou estimate of the accuracy.

### 2.2 - Model Modification and Adaptation for Training & Testing

<p align="center">
    <img src="./images/Training.jpg" width="50%" height="50%"/>
</p>
 
&emsp; To begin this part, time was spent to get acquainted with MotionBert and run main functions created for this specific model such as the training or the testing. This step was important to get a more precise idea the current structure and work needed, of the functions to implement to adapt to our needs, and the ones to get rid of. 

&emsp; To process and extract data from the Dataset pickle file previously created and intended for our model, new Dataset subclasses (for bounding boxes only, and bounding boxes & 2D keypoints) returning the data in the right format for the training and testing DataLoader was created.

&emsp; A new training file was created implementing the important training, validating and evaluating function. This training file was modified to take into account the new Dataset subclasses, and a new specific configuration file *JAAD_train.yaml* previously created to store the network specifications and choosen training parameters. The training function is implementing a checkpoint system to save the best model, the last epoch, and to record the currently implemented metrics such as Loss, Accuracy, and F1. 

### 2.3 - Inference Data Creation & Model Prediction

<p align="center">
    <img src="./images/Inference.jpg" width="50%" height="50%"/>
</p>

&emsp; In order to run inference, the created model needs as input 2D keypoints, bounding boxes and occlusion. Since this is not provided in the video for inference, it had to be created. To generate this data, a new function was created to run OpenPifPaf on the video, and to store the generated keypoints and bounding boxes in a pickle file, and the keypoints are then processed in another function to best adapt to the model's input.

&emsp; A new inference file (containing the above functions) was created to make predictions on any video, and the to save them in a json file. This file was also modified to support a new Dataset subclasses *KPInfDataset* loading the pickle file storing the inference data extracted from the inference video. This file allows to run the model on the processed inference data, and to save the predictions with their correspondant pedestrian bounding box in a json file.

&emsp; To visualize and better evaluate the results of the inference, a new function was created to draw the bounding boxes on the video with the predicted labels, and to save it as a new video. To do, this function takes as input the inference video, and the json file containing the predictions.

## 3 - Description of the Data

### 3.1 - JAAD Dataset

The pickle datafile contaning the JAAD dataset processed for our model has the following dictionnary structure:
``` 
'annotations': 
    'vid_id'(str): 
        'num_frames':               int
        'width':                    int
        'height':                   int
        'ped_annotations':          list (dict)
            'ped_id'(str):              list (dict)
                'old_id':                   str
                'frames':                   list (int)
                'occlusion':                list (int)
                'bbox':                     list ([x1 (float), y1 (float), x2 (float), y2 (float)])
                '2dkp':                     list (array(array))
                'cross':                    int
'split': 
    'train_ID':     list (str)
    'test_ID':      list (str)
'ckpt': str
'seq_per_vid': list (int)
```

Dictionnary keys :
- `'annotations'` - list of video dictionnaries used for training and testing the model
- `'vid_id' (str)` - dictionnary of the properties for the video `'vid_id'`
- `'width'` - width of the video `'vid_id'`
- `'height'` - height of the video `'vid_id'`
- `'ped_annotations' (str)` - list of pedestrians in the video `'vid_id'`
- `'ped_id (str)'` - list of sequences annotations dictionnaries for pedestrian `'ped_id (str)'`
- `'old_id'` - default `'old_id'` from the JAAD Database
- `'frames'` - list of frame number relative to the video `'vid_id'` contained in the current sequence
- `'occlusion'` - list of occlusion for all the frames contained in the current sequence containing the frames `'frames'`
- `'bbox'` - list of bounding boxes for all the frames contained in the current sequence containing the frames `'frames'`
- `'2dkp'` - list of arrays of 2D keypoints for all the frames contained in the current sequence containing the frames `'frames'`
- `'cross'` - label cross (0 or 1) for the current sequence containing the frames `'frames'`
- `'split'` - dictionnary with the `'train_ID'` and `'test_ID'`
- `'train_ID'` - list of `'vid_id'` contained in the training set (80% of the data)
- `'test_ID'` - list of `'vid_id'` contained in the testing set (20% of the data)
- `'ckpt'` - last `'vid_id'` that was being processed
- `'seq_per_vid'` - list of number of sequences per video for all videos


### 3.2 - Inference Data

Dictionary structure:
```
'vid_id':           str
'num_seq':          int
'forecast_step':    int
'nbr_frame_seq':    int
'total_frame_vid':  int
'width':            int
'height':           int
'per_seq_ped':      list (list (int))
'ped_annotations':      list (dict) 
    'frames':               list (int)
    'occlusion':            list (int)
    'bbox':                 list ([x1 (float), y1 (float), x2 (float), y2 (float)])
    '2dkp':                 list (array(array))
```

Dictionnary keys :
- `'vid_id'` - filename of the inference video without extension
- `'num_seq'` - number of sequences created
- `'forecast_step'` - overlap step between sequences
- `'nbr_frame_seq'` - number of frames per sequences
- `'total_frame_vid'` - total number of frames in the video
- `'width'` - width of the inference video
- `'height'` - height of the inference video
- `'per_seq_ped'` - list of list of index of pedestrian for each sequence
- `'ped_annotations'` - list of pedestrians in the inference video
- `'frames'` - list of frame number relative to the inference video contained in the current sequence of the current pedestrian
- `'occlusion'` - list of occlusion for the current pedestrian for all the frames contained in the current sequence containing the frames `'frames'`
- `'bbox'` - list of bounding boxes for the current pedestrian for all the frames contained in the current sequence containing the frames `'frames'`
- `'2dkp'` - list of arrays of 2D keypoints for the current pedestrian for all the frames contained in the current sequence containing the frames `'frames'`


## 4 - Experimental Setup and Results

&emsp; During the conception of this project, we had to make some experiments to find the best model for our problem. This section will describe the experiments we made and the results we obtained.

### 4.1 - Experiments Setup

#### 4.1.1 - Metrics

&emsp; To evaluate the performances of our model, **three main metrics** were used: the ***loss***, the ***accuracy*** and the ***F1 score***.

 The loss is computed, and optimised by the model during both the training and the testing. We compute a **CrossEntropyLoss** for our model. The accuracy is the **percentage of correct prediction** made by the model both on the training set and the testing set. To make the prediction we choose the class (crossing or not crossing) that get the highest softmax probability. This way of computing the loss and accuracy may create what we observe in the result : the fact that for the training the loss increase but the accuracy stays pretty much constant. The F1 score is the **harmonic mean** of the **precision** and **recall**. The precision is the percentage of correct positive prediction made by the model. The recall is the percentage of positive prediction made by the model that are correct. We compute the F1 measure also on the training and testing set.

#### 4.1.2 - Overfitting 

&emsp; To make sure our system could model the problem, we first made sure that it was able to overfit a small portion of the training set. A small training set of 10 sequences was used and the model was trained for 20 epochs. It was observed that the model was able to overfit the training set, and it was thus concluded that data could be modeled by our system.

#### 1. Bounding box only
&emsp; The first idea we had was to trying to give to the model only the **bounding boxes** of the pedestrian. It contains the position of the pedestrian in the image, thus we thought that it could be enough for the model to predict the crossing intention (most of the time the car has a camera from the point of view of the driver then the relative position of a pedestrian in the image could be good enough to predict his intention).
We used scitas and google collab to generate the Dataset and train the model. We also used the same hypeparameters as the one used in the original paper and we juste changed the dimension of the input and output of the network.

#### 2. Regularization + Less complex model
&emsp; The result of the first experiment was satisfying for a first experiment. However it was quickly overfitting thus we thought that the model was maybe too complex for our problem. We decided to use a **less complex model** and to add some **regularization** to the model.

&emsp; To do so, we looked into the structure of the model in the original paper and using tensorboard visualization and we noticed that the model was composed of 5 blocks of spatial and temporale encoding layers in the original architecture. This seems to be too much taking into account that with only the bounding box as input, the model has less information to process. We decided to reduce the number of blocks to 3.

&emsp; We also added some regularization to the model by increasing the weight decay and the dropout rate.

#### 3. Keypoints
&emsp; As the second experiment was also not giving conclusive results, we decided to give more information to the model and we decided to use **2D pose keypoints** to the input of the model. As we don't have keypoints in JAAD, we used **openpifpaf** to estimate them for all the videos. 

We thought that it could help the model to better understand the position of the pedestrian, more specificaly the model would be then able to understand aspect of the pedestrian like when it looks or notice the car, the way his body is oriented, etc. Moreover, MotionBert was originally trained to use keypoints as input, so we thought that it could be a good idea to use them.

#### 4. Bounding box, occlusion + keypoints + 1s prediction, 1s seqence:
&emsp; The third experiment didn't show any particular improvement neither. We decided to try to predict the crossing intention of the pedestrian using **1 second instead of 2 second** from the video as input sequence length. We thought that it could help the model to better generalize. 

We also tried to increase the prediction fame to 1 second in the future instead of 0.5 second. We thought that it could help the model to better understand the intention of the pedestrian. 

Finally we added information about the **ground truth bounding box** and the **occlusion score** in order for the model to be able to have access to the position information even when openpifpaf doesn't detect keypoints.

#### 5. Bounding box, occlusion + keypoints + 1s prediction, 1s seqence + dynamic detection to identify the pedestrian:

&emsp; As a last experiment, we decided to try to use a **dynamic detection** to identify the pedestrian in the video in order to reduce the potential error we could have in the identification of the pedestrian. Before that we we're using a constant threshold to identify the correspondancy between the bouding box given by JAAD and the keypoints given by openpifpaf and we thought that this may cause error in the identification process. To solve that, we use in this experiment dynamic distances between keypoints and bounding boxes to identify the pedestrian.

### Results

#### Overfitting :

![overfitting](./images/overfitting.png)

As we can see on the plot, the model is able to overfit on a little training set. We achieved an accuracy of 100%, F1 of 1 and a loss of 0. This is a good sign that the model is able to learn.

#### 1. Bounding box only

- <u>during training :</u>
<image>

- <u>discussion :</u> 

The result are pretty satisfying for a first experiment. We're able to achieve an accuracy of 70 % in average only with the bounding box. However, we can see that the model is overfitting very quickly. We think that it's because the model is too complex for the low complexity input. 

For the next experiment we decided to try to reduce the complexity of the model and to add some regularization in order to try solving the overfitting problem.

#### 2. Regularization + Less complex model

- <u>during training :</u>
<image>

- <u>discussion :</u> 

#### 3. keypoints

- <u>during training :</u>
<image>

- <u>discussion :</u> 

#### 4. Bounding box, occlusion + keypoints + 1s prediction, 1s seqence

- <u>during training :</u>
![training plot 4](./images/exp4.png)

- <u>discussion :</u> 

As we can see in the result, this experiment didn't bring improvement, it even perform less good than with just the keypoints. We think that this can be caused by the fact that by bringing the bounding box we thought that the model will be able to have information even when openepifpaf doesn't detect keypoints. However if the detection gives wrong keypoints, it can cause errors in the identification of the pedestrian or just comprehension erro for the model that can't learn to use the bounding box to estimate the position when he doesn't have the keypoints.
For this reason we decided to try to use a dynamic detection to identify the pedestrian.

#### 5. Bounding box, occlusion + keypoints + 1s prediction, 1s seqence + dynamic detection to identify the pedestrian

- <u>during training :</u>

![training plot 5](./images/exp5.png)

- <u>discussion :</u> 

As we can see on the result, we didn't get any particular improvement on the loss. However the best checkpoint we have and the mean accuracy and f1 measure seems to be a bit impoved (0.7442 vs 0.7044 for the accuracy and 0.5949 vs 0.5850 for the f1 measure). We can see that the model is able to generalize a bit better. Thus we can conclude that the dynamic detection is a good idea to improve the result of the model.

#### Best Checkpoint Result Comparison :

| Experiment | Loss | Accuracy | F1 score |
|------------|------|----------|----------|
|     5      | 0.6488 | 0.7442 | 0.5949   |
|     4      | 0.5084 | 0.7044 | 0.5850   |
|     3      | 0.6931 | 0.5000 | 0.6667   |
|     2      | 0.6931 | 0.5000 | 0.6667   |
|     1      | 0.6931 | 0.5000 | 0.6667   |

## Conclusion

&emsp; To conlude, the task/contribution we choosed was to create a deep learning application using MotionBert to predict the crossing intention of pedestrians in the context of autonomous driving. We tried and experimented different ideas to solve the problem and we managed to create a model that is able, even if it's not with an estonishing accuracy, to predict the crossing intention of pedestrians and that can make an inference on raw video data. 

&emsp; However, the model is not able to generalize and tends to still overfit quickly the training set. We think that the problem comes from the fact that the Dataset is too small and we don't have ground truth keypoints. Indeed for a complex model like MotionBert, we have a very little amount of samples, even after cutting JAAD into sequences of 1 second. Moreover, we don't have ground truth keypoints, so we have to use openpifpaf to detect the keypoints and this can cause some errors in the identification of the pedestrian. We think that if we had a bigger Dataset with ground truth keypoints, we could have a better model.

#### Further improvements :

&emsp; We still think that our application have a great potential and that it could be improved in many ways. Here are some ideas that we think could improve the performances of our model :

- Use a **bigger Dataset** like the PIE Dataset that contains a lot more samples and video time than JAAD or use a Dataset that contains ground truth keypoint.
- Use **semi-supervised learning** to use the unlabeled data of the JAAD Dataset (or PIE Dataset) to train the model, allowing it to generalize better and to create a pre-knowledge on what is a pedestrian and how it behaves (we could try for example to first predict his displacement and then his crossing intention).
- Use **semi-supervised learning** on a another Dataset to pretrain the model and then fine-tune it on JAAD or PIE. If we take the example of MotionBert we could train first our model to predict the 3D pose of a person from a 2D pose, this way it would get an deeper understanding on the what is a pose, and then fine-tune it to predict the crossing intention of a pedestrian.
- Try to **improve the detection step** of the pose with openpifpaf on the video. Indeed, we noticed that the keypoints given by openpifpaf are not always correct and that it can cause some errors in the identification of the pedestrian. We could try to solve this problem by finding ways to ensure the best detection possible.
