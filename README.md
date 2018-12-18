# Toy-Model-Checking





<br/><br/>


## 1. Analyze


We have 17 min training video, with 20 frames per second. It's 20400 frames, where we have speed on each frame. From the initial description, we can start with some basics on it a) Features taken from Images b) Features elapsed over time. We will include all work we had, and other possible thoughts.

Start with checking training video, we will see autobahn for about 70% of the video, and some street roads (30 %). We do have some noises, like low distance between cars, and few turns with very smoothed rotation. Some examples of the frames, you can see below. Nothing special at all. 

<br/> <br/> <br/>
<br/><br/>

<img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/0/car-normal-ex1-0.gif" width="280" height="170" /> <img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/0/car-normal-ex2-0.gif" width="280" height="170" /> <img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/0/car-normal-ex3-0.gif" width="280" height="170" /> 

1.1.1. Examples of Working Areas in the train set.
<br/> <br/> <br/> 
<br/><br/>



However, we have existing video for testing. Test video have 9 min video with the same framerate, and will be used for the next model evaluation. As you will see, it's not just usual task, where you need to keep in mind a regularization objective. But think about alternatives for the feature extraction. 

So lets describe test set, and what we have.Around 30 % of the driving it's autobahn, and other part it's streen roads. We have a lot of noises, which completely new comparing to the training set. Few of them are - Sharp turns - Car stop - Road steep descent and ascent.

<br/> <br/> <br/>
<br/><br/>

<img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/0/car-turning-ex1-0.gif" width="280" height="170" /> <img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/0/car-stop-ex1-0.gif" width="280" height="170" />  <img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/0/car-turning-ex2-0.gif" width="280" height="170" /> 

1.1.2. Examples of Working Areas in the test set. LTR Car turning Ex.1 - Car stop - Car turning Ex.2
<br/> <br/> <br/>
<br/><br/>



We will start with check training data and it's changes over time. Most useful information, it's changes speed over time. You can see, where speed changes very swift, and we can use it for tracking anomaly detection for the next validation phase (when ever your predictions meet near to unexpected result). 


<br/>

<img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/0/speed-changes-with-frame.png" width="900" height="250" /> 

1.1.3. Speed value at Frame index from initial training Video. 
<br/> <br/>

<br/> 

<img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/0/speed-changes-with-previous.png" width="900" height="250" /> 

1.1.4. Delta Speed changes over Previous frame. 
<br/> <br/> 


Above Image explains how Speed may change from the previous. Even one Video is not enough to relay on this value, we can assume this parameter is good for validation and checking anomalies in a prediction. One more sample, of avarage rate of changing Speed, over previous 20 frames (1 second in time). 

<br/>

<img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/0/speed-changes-delta-20-frame.png" width="900" height="250" /> 

1.1.5. Delta Speed changes over Previous 20 frames.
<br/> <br/>











<br/> <br/>

## 2. Preprocessing


### 2.1 Working Area
From each frame, we can decide working area, which includes most of the features, depending on our task. And reduce number of the Inputs. Even we did some feature area investigation, it's easy to manually retrive most useful part.

<br/>

<img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/1.1/source-image.jpg" width="400" height="300" /> <img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/1.1/source-feature-area.png" width="400" height="300" /> 

2.1.1. Source Image and Source Image Areas. Red Box represents Working Area, for the next steps. 
<br/> <br/>

But we have another interesting question here. As you can see, we have Working Area, which consist from some Sub Areas, Close and Far. First one, has more features about road changes (has more changes over time, due of camera angle). And Second one with more noise like side objects, other cars and less road changes. 

<br/>

<img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/1.1/car-angle-variants.jpg" width="800" height="200" /> 

2.1.2. Frame changes updates in different way, based on Angle of a Camera. 
<br/> <br/>

At this point, we have few variants, for next preprocessing steps. We will use this options in the next phases. - Using Far Sub Area. - Using Mid Sub Area. - Using Close Sub Area, and so on.

*Note. It's very important to reduce number of Inputs, especialy in such cases, where we working with Video, and features elapsed over time. One wrong step will cause your model to have the Curse of Dimensionality. That is why we suppose to avoid last variant with using complete Working Area.*


Below animation, how we can estimate it, with changes over time. Note, this sample only for visualizing and deciding about next steps. because correct Area we can choose, only after using some Model and testing feature extraction on each frame. 

<br/>

<img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/1.1/image-mov-top-0.gif" width="280" height="170" /> <img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/1.1/image-mov-mid-0.gif" width="280" height="170" /> <img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/1.1/image-mov-bot-0.gif" width="280" height="170" /> 

2.1.3. Sub Area features changing over time. LTR Far - Mid - Close Areas.
<br/> <br/>

As a result, we come with few options for next work **a) Check Model velocity using several Sub Areas Types. b) Check Model velocity over all possible Sub Areas, with momentum over some shifting.**



<br/>

### 2.2 Scaling and Normalization

Scaling takes some part in preprocessing, even we can simplify and avoid it's usage. Scale working in two directions, and for some cases might reduce a Dimensionality in two different ways. We will not talk about simple scaling down Image, to reduce it's size, but in some resources, you can find opossite usage of scaling, by increasing Image size.

<br/>

<img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/1.2/scale-source.gif" width="280" height="170" /> <img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/1.2/scale-image.gif" width="280" height="170" /> <img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/1.2/scale-compressed.gif" width="280" height="170" />

2.2.1. Feature changing over time. LTR Source - Scaled - Compressed Images.
<br/><br/>

For some cases, very simple process of scaling up small Image helps to make some kind of preprocessing and retaint most important features and their changes over time. We can compare this process, to some N Compressing, but requires substantially less resources to compute. 

For the normalization, we don't have something new. In case we working with Images, we can just normilize features, over maximum value of colors. Mean of RBG values at pixel divided by Max possible color value (255) (or even simpler for an Image, loaded in the Grey filter). As a another point, we can think about normalization with `Scikit-Learn` Scalers, however first variant is good enough. 

At this stage, we will play with Scaling in two different ways. After choosing some Sub Area from Working, we will **a) Apply some Scalling up and down based on Working Area size.*


<br/>

### 2.3 Frames Timeline 

As we mentioned before, we will work on features elapsed over time. And for the single training sample, we should have few frames before the focus one. This Timeline of single training sample, might be configured in very different way. And not only previous frame, before the focused.

<br/>

<img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/1.3/frames-elapsed-over-time.jpg" width="800" height="160" /> 

2.3.1. Single training sample consist from several frames
<br/><br/>

For testing purpose, we created Preprocessor objects, which can retrive Timelines in different format, and not only previous frame. We can represent it, as indexes '(0, 1, 2)', where '(0)' it's current frame, where we now the right Y value (and looking back to the previous '(1)', and next to it '(2)' frames). 

In out testing, we might check dependencies between frames '(0, 1)' or '(0, 1, 2, 3, 4)' and some variations in this range.   And even more complicated behavior, Timeline with different steps (ex. with step 2, we will have '(0, 2, 4)'), where frame changes will be more visible. So action items for this mapping is **a) Check different Frame Timelines.**


<br/>

### 2.4 Frames Augmentation

TBD



<br/>

### 2.4 Regularization and Errors

TDB



<br/>

### 2.5 Сonclusions


In general it's very simple process, where we just shared all thoughts during Preprocessing. For some model, we should take Frames Timeline on Working Area and Normilize inputs. However we came up with several different options: Preprocessor Combinations, which we should investigate (combination of all possible parameters, also marked above). 

Well known other algorithms might be used for feature extraction on Images. Some of them like `SIFT`, `ORB`, `HOG` and other, might work very well, but doesn't suite for our problem. Main reason covers in performance of our model, and such actions takes huge time for computing.







<br/> <br/>


## 3. Model.

Same with preprocessing, we will describe just all investigation and work for choosing Model. We will start from simple implementations. In first phases, we suppose to keep everything as simple, as possible. Because you can increase complexity in any time and every part of the Model. 

Below you will find some graphics and Model structures. No Models (or it's structure) has the goal of being correct and used in dirty examples for general evaluation. Check sub branches, for the detailed implementation for each of them. They might have mistakes.  


<br/>

### 3.1 MLP and 2D-CNN

Even from the initial sentence of the task (with processing video), which gives us hint to move in RNN directions, we had few examples of MLP. In our examples, we might represent features as delta changes between frames, where tried to achive some kind of Compressing during Timeline.

After some minor updates in Preprocessing behavior, we used different types of MLP, with different structure. However, this work, didn't effect model learning, and it was hard to resolve pattern in delta changes over time. Even with no Regularization, model didn't fit training samples. No Graphics here. No Samples.

Why 2D-CNN also included here, because we supposed to have such model, with Convolution over Delta changes. Comparing to previous MLP where we might have few Frames on Timeline (which Flatten into row of features), for this case, we had only one delta Frame over previous. Again in our testing it seems real to find correct Windows for Convolution in delta changes over time. But not so good after validation.


<br/>

### 3.2 RNN and LSTM

Back to initial thoughts, RNN should have enough velocity on data over time. One more pros here, it's number of the inputs. Comparing to the previous methods, where we Flatten few frames (based on Timeline) into one blob features, here we can consume frames one by one. And finally we can see some progress in Model validation. It was good for next steps, but we didn't enough for working solution. 



<img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/2.1-2.3/lstm-sample-over-iteration.png" width="600" height="300" /> 

3.2.1. Velocity of RNN model on some Preprocessor Combinations.
<br/><br/>



<br/>

### 3.3 2D-CNN with LSTM

Combination of 2D-CNN windows, and tracking frames changes over time should work very well. And for validation, we received even worse result, comparing of RNN. Back to model strucutre, we continue some updates, slowly increasing model complexity, but tests result still not enough to work in this direction. 


<img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/2.1-2.3/lstm-cnn-sample-over-iteration.png" width="600" height="300" /> 

3.3.1. Velocity of 2D-CNN with LSTM Model on some Preprocessor Combinations.
<br/><br/>


<br/>

### 3.4 3D-CNN

Obviously 3D-CNN takes places in this list. Even this is most recommended types for the features elapsed over time, we came to this Model in last examples. And this Model works noticeably well on validation data. Starting from very simple Model we can see very good results, and continue work on Model complexity and Preprocessor Combinations. You can see how well this Model fit, comparing to other types. 


TBD

3.4.1. Velocity of 3D-CNN Model.
<br/><br/>

