# Report

## Introduction

Our project was to implement a model that predicts the crossing intention of pedestrians in the context of autonomous driving.

This report will describe the project and result in details.
## Contribution Overview

To solve the problem we're assigned, we choose to base our model on the [MotionBert](https://github.com/Walter0807/MotionBERT) model, the current state of the art model for 3D pose estimation and action recognition. While none of those tasks are directly related to our problem, we believe that the model used for the action recognition can be adapted to our problem. This is our contribution, adapt the MotionBert model allowing it to predict the crossing intention of pedestrians.

#### Changes made:

<span style="color:red">**DIMITRI PARTIE MODIFICATION DATASET**</span>

Because we're using an other dataset, with another formatting, we have to adapt the dataset class to our needs. 

## Description of the dataset

## Experimental Setup and Results

During the conception of this project, we had to make some experiments to find the best model for our problem. This section will describe the experiments we made and the results we obtained.

### Experiments

#### 1. Bounding box only
The first idea we had was to give to the model only the bounding box of the pedestrian. As it contains the position of the pedestrian in the image, we thought that it could be enough for the model to predict the crossing intention.
We used scitas and google collab to generate the dataset and train the model. We also used the same hypeparameters as the one used in the original paper and we juste changed the dimension of the input and output of the network.

#### 2. Regularization + Less complex model
As the first experiment didn't give good results, and that it was quickly overfitting we thought that the model was too complex for our problem. We decided to use a less complex model and to add some regularization to the model.

To do so, we looked into the structure of the model in the original paper and using tensorboard visualization (see result section) and we noticed that the model was composed of 5 blocks of spatial and temporale encoding layers in the original architecture. This seems to be too much taking into account that with only the bounding box as input, the model has less information to process. We decided to reduce the number of blocks to 3.

We also added some regularization to the model by increasing the weight decay and the dropout rate.

#### 3. Bounding box, occlusion + keypoints
As the second experiment was not giving conclusive results, we decided to add the keypoints to the input of the model. We thought that it could help the model to better understand the position of the pedestrian, more specificaly the model would be then able to understand aspect of the pedestrian like when it look or notice the car, the way his body is oriented, etc. We also add to the model the occlusion score given by JAAD. Therefore, we thought that it would able to better predict the crossing intention. Moreover, MotionBert was originally trained to use keypoints as input, so we thought that it could be a good idea to use them.

#### 4. Bounding box, occlusion + keypoints + 1s prediction, 1s seqence:
The third experiment didn't show any particular improvement. We decided to try to predict the crossing intention of the pedestrian using one second instead of 2 second frome the video as input sequence length. We thought that it could help the model to better generalize. We also tried to increase the prediction fame to 1 second in the future instead of 0.5 second. We thought that it could help the model to better understand the intention of the pedestrian.

#### 5. Bounding box, occlusion + keypoints + 1s prediction, 1s seqence + dynamic detection to identify the pedestrian:

As the fourth experiment didn't show any particular improvement neither, we decided to try to use a dynamic detection to identify the pedestrian in the video. Before that we we're using a constant threshold to identify the output the correspondancy between the bouding box given by JAAD and the keypoints given by openpifpaf and we thought that this may cause error in the identification process .
## Conclusion
