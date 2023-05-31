# Report

## Introduction

&emsp; Our project was to implement a model that predicts the crossing intention of pedestrians in the context of autonomous driving. This report will describe the project and result in details.
## Contribution Overview

&emsp; To solve the problem we're assigned, we choose to base our model on the [MotionBert](https://github.com/Walter0807/MotionBERT) model, the current state of the art model for 3D pose estimation and action recognition. While none of those tasks are directly related to our problem, we believe that the model used for the action recognition can be adapted to our problem. This is our contribution, adapt the MotionBert model allowing it to predict the crossing intention of pedestrians.

#### Changes made:

<span style="color:red">**DIMITRI PARTIE MODIFICATION DATASET**</span>

Because we're using an other dataset, with another formatting, we have to adapt the dataset class to our needs. 

## Description of the dataset

## Experimental Setup and Results

&emsp; During the conception of this project, we had to make some experiments to find the best model for our problem. This section will describe the experiments we made and the results we obtained.

### Experiments Setup

#### Metrics

&emsp; To evaluate the performances of our model, we use **three main metrics** : the ***loss***, the ***accuracy*** and the ***F1 score***.

 The loss is the loss computed by the model during both the training and the testing. We compute a **CrossEntropyLoss** for our model. The accuracy is the **percentage of correct prediction** made by the model both on the training set and the testing set. To make the prediction we choose the class (crossing or not crossing) that get the highest softmax probability. This way of computing the loss and accuracy may create what we observe in the result : the fact that for the training the loss increase but the accuracy stays pretty much constant. The F1 score is the **harmonic mean** of the **precision** and **recall**. The precision is the percentage of correct positive prediction made by the model. The recall is the percentage of positive prediction made by the model that are correct. We compute the F1 measure also on the training and testing set.

#### Overfitting :
To be noted that we made sure that our model was able to learn by checking that it was able to overfit a little training set.

#### 1. Bounding box only
&emsp; The first idea we had was to give to the model only the **bounding box** of the pedestrian. As it contains the position of the pedestrian in the image, we thought that it could be enough for the model to predict the crossing intention.
We used scitas and google collab to generate the dataset and train the model. We also used the same hypeparameters as the one used in the original paper and we juste changed the dimension of the input and output of the network.

#### 2. Regularization + Less complex model
&emsp; As the first experiment didn't give good results, and that it was quickly overfitting we thought that the model was maybe too complex for our problem. We decided to use a **less complex model** and to add some **regularization** to the model.

&emsp; To do so, we looked into the structure of the model in the original paper and using tensorboard visualization and we noticed that the model was composed of 5 blocks of spatial and temporale encoding layers in the original architecture. This seems to be too much taking into account that with only the bounding box as input, the model has less information to process. We decided to reduce the number of blocks to 3.

&emsp; We also added some regularization to the model by increasing the weight decay and the dropout rate.

#### 3. Bounding box, occlusion + keypoints
&emsp; As the second experiment was also not giving conclusive results, we decided to add the **keypoints** to the input of the model. As we don't have keypoints in JAAD, we used **openpifpaf** to estimate them for all the videos. 

We thought that it could help the model to better understand the position of the pedestrian, more specificaly the model would be then able to understand aspect of the pedestrian like when it looks or notice the car, the way his body is oriented, etc. We also add to the model the occlusion score given by JAAD. Moreover, MotionBert was originally trained to use keypoints as input, so we thought that it could be a good idea to use them.

#### 4. Bounding box, occlusion + keypoints + 1s prediction, 1s seqence:
&emsp; The third experiment didn't show any particular improvement neither. We decided to try to predict the crossing intention of the pedestrian using **1 second instead of 2 second** frome the video as input sequence length. We thought that it could help the model to better generalize. We also tried to increase the prediction fame to 1 second in the future instead of 0.5 second. We thought that it could help the model to better understand the intention of the pedestrian.

#### 5. Bounding box, occlusion + keypoints + 1s prediction, 1s seqence + dynamic detection to identify the pedestrian:

&emsp; As a last experiment, we decided to try to use a **dynamic detection** to identify the pedestrian in the video. Before that we we're using a constant threshold to identify the correspondancy between the bouding box given by JAAD and the keypoints given by openpifpaf and we thought that this may cause error in the identification process. In order to solve that, we use in this experiment dynamic distances between keypoints and bounding boxes to identify the pedestrian.

### Results

#### Overfitting :

![overfitting](./images/overfitting.png)

As we can see on the plot, the model is able to overfit a little training set. We achieve an accuracy of 100%, F1 of 1 and a loss of 0. This is a good sign that the model is able to learn.

#### 1. Bounding box only

- <u>during training :</u>
<image>

- <u>final result (best epoch) :</u>

| Loss | Accuracy | F1 score |
|------|----------|----------|
| 0.6931 | 0.5000 | 0.6667 |

- <u>discussion :</u>  

#### 2. Regularization + Less complex model

- <u>during training :</u>
<image>

- <u>final result (best epoch) :</u>

| Loss | Accuracy | F1 score |
|------|----------|----------|
| 0.6931 | 0.5000 | 0.6667 |

- <u>discussion :</u> 

#### 3. Bounding box, occlusion + keypoints

- <u>during training :</u>
<image>

- <u>final result (best epoch) :</u>

| Loss | Accuracy | F1 score |
|------|----------|----------|
| 0.6931 | 0.5000 | 0.6667 |

- <u>discussion :</u> 

#### 4. Bounding box, occlusion + keypoints + 1s prediction, 1s seqence

- <u>during training :</u>
<image>

- <u>final result (best epoch) :</u>

| Loss | Accuracy | F1 score |
|------|----------|----------|
| 0.6931 | 0.5000 | 0.6667 |

- <u>discussion :</u> 

#### 5. Bounding box, occlusion + keypoints + 1s prediction, 1s seqence + dynamic detection to identify the pedestrian

- <u>during training :</u>

![training plot](./images/exp5.png)

- <u>final result (best epoch) :</u>

| Loss | Accuracy | F1 score |
|------|----------|----------|
| 0.6488 | 0.7442 | 0.5949  |

- <u>discussion :</u> 

As we can see on the result, we didn't get any particular improvement on the final accuracy. However, we can definitly see that during the traning the accuracy seems more stable and that's an improvement compare to the last experiment. 

## Conclusion

&emsp; To conlude, the task/contribution we choosed was to create a deep learning application using MotionBert to predict the crossing intention of pedestrians in the context of autonomous driving. We tried and experimented different ideas to solve the problem and we managed to create a model that is able, even if it's not with an estonishing accuracy, to predict the crossing intention of pedestrians and that can make an inference on raw video data. 

&emsp; However, the model is not able to generalize and tends to still overfit quickly the training set. We think that the problem comes from the fact that the dataset is too small and we don't have ground truth keypoints. Indeed for a complex model like MotionBert, we have a very little amount of samples, even after cutting JAAD into sequences of 1 second. Moreover, we don't have ground truth keypoints, so we have to use openpifpaf to detect the keypoints and this can cause some errors in the identification of the pedestrian. We think that if we had a bigger dataset with ground truth keypoints, we could have a better model.

#### Further improvements :

&emsp; We still think that our application have a great potential and that it could be improved in many ways. Here are some ideas that we think could improve the performances of our model :

- Use a **bigger dataset** like the PIE dataset that contains way more samples and video time than JAAD or use a dataset that contains ground truth keypoint.
- Use **semi-supervised learning** to use the unlabeled data of the JAAD dataset (or PIE dataset) to train the model, allowing it to generalize better and to create a pre-knowledge on what is a pedestrian and how it behaves (we could try for example to first predict his displacement and then his crossing intention).
- Use **semi-supervised learning** on a another dataset to pretrain the model and then fine-tune it on JAAD or PIE. If we take the example of MotionBert we could train first our model to predict the 3D pose of a person from a 2D pose, this way it would get an deeper understanding on the what is a pose, and then fine-tune it to predict the crossing intention of a pedestrian.
- Try to **improve the detection step** of the pose with openpifpaf on the video. Indeed, we noticed that the keypoints given by openpifpaf are not always correct and that it can cause some errors in the identification of the pedestrian. We could try to solve this problem by finding ways to ensure the best detection possible.
