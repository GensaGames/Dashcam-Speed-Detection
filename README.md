# Toy-Model-Checking

<br/>


## 1. Preprocessing

This section will include all work we did, and other possible thoughts. We have 17 min training video, with 20 frames per second. It's 20400 frames, where we have speed on each frame. From the initial description, we can start with some basics on it. 1. Features taken from Images 2. Features elapsed over time. 



<br/>

### 1.1 Working Area
From each frame, we can decide working area, which includes most of the features, depending on our task. And reduce number of the Inputs. Even we did some feature area investigation, it's easy to manually retrive most useful part.

<br/>

<img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/source-image.jpg" width="400" height="300" /> <img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/source-feature-area.png" width="400" height="300" /> 

1.1.1. Source Image and Source Image Areas. Red Box represents Working Area, for the next steps. 
<br/> <br/>

But we have another interesting question here. As you can see, we have Working Area, which consist from `Close` and `Far` subareas. First one, has more features about road changes (has more changes over time, due of camera angle). And Second one with more noise like side objects, other cars and less road changes. 

<br/>

<img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/car-angle-variants.jpg" width="800" height="200" /> 

1.1.2. Frame changes updates in different way, based on Angle of a Camera. 
<br/> <br/>

At this point, we have few variants, for next preprocessing steps. We will use this options in the next phases. - Using Far Sub Area. - Using Mid Sub Area. - Using Close Sub Area. - Using complete Working Area. 

*Note. It's very important to reduce number of Inputs, especialy in such cases, where we working with Video, and features elapsed over time. One wrong step will cause your model to have the Curse of Dimensionality. That is why we suppose to avoid last variant with using complete Working Area.*


Below animation, how we can estimate it, with changes over time. Note, this sample only for visualizing and deciding about next steps. because correct Area we can choose, only after using some Model and testing feature extraction on each frame. 

<br/>

<img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/image-mov-top-0.gif" width="280" height="170" /> <img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/image-mov-mid-0.gif" width="280" height="170" /> <img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/image-mov-bot-0.gif" width="280" height="170" /> 

1.1.3. Sub Area features changing over time. LTR Far - Mid - Close Areas.
<br/> <br/>

As a result, we come with few options for next work **a) Check Model velocity using several Sub Areas Types. b) Check Model velocity over all possible Sub Areas, with momentum over some shifting.**



<br/>

### 1.2 Scaling and Normalization

Scaling takes some part in preprocessing, even we can simplify and avoid it's usage. Scale working in two directions, and for some cases might reduce a Dimensionality in two different ways. We will not talk about simple scaling down Image, to reduce it's size (ex. 500x500 scale down to 250x250), but in some resources, you can find opossite usage of scaling, by increasing Image size (ex. 125x125 scale up to 250x250).

<br/>

<img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/1.2/scale-source.gif" width="280" height="170" /> <img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/1.2/scale-image.gif" width="280" height="170" /> <img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/1.2/scale-compressed.gif" width="280" height="170" />

1.2.1. Feature changing over time. LTR Source - Scaled - Compressed Images.
<br/><br/>

For some cases, very simple process of scaling up small Image helps to make some kind of preprocessing and retaint most important features and their changes over time. We can compare this process, to some N Compressing, but requires substantially less resources to compute. 

For the normalization, we don't have something new. In case we working with Images, we can just normilize features, over maximum value of colors. Mean of RBG values at pixel divided by Max possible color value (255) (or even simpler for an Image, loaded in the Grey filter). As a another point, we can think about normalization with `Scikit-Learn` Scalers, however first variant is good enough. 

At this stage, we will play with Scaling in two different ways. After choosing some Sub Area from Working, we will **a) Apply few Scalling Down types. b) Apply some Scalling Up types.**


<br/>

### 1.3 Frames Timeline 

As we mentioned before, we will work on features elapsed over time. And for the single training sample, we should have few frames before the focus one. This Timeline of single training sample, might be configured in very different way. And not only previous frame, before the focused.

<br/>

<img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/1.3/frames-elapsed-over-time.jpg" width="800" height="160" /> 

1.3.1. Single training sample consist from several frames
<br/><br/>

For testing purpose, we created `Preprocessor` objects, which can retrive Timelines in different format, and not only previous frame. We can represent it, as indexes `(0, 1, 2)`, where `(0)` it's current frame, where we now the right Y value (and looking back to the previous `(1)`, and next to it `(2)` frames). 

In out testing, we might check dependencies between frames `(0, 1)` or `(0, 1, 2, 3, 4)` and some variations in this range.   And even more complicated behavior, Timeline with different steps (ex. with step 2, we will have `(0, 2, 4)`), where frame changes will be more visible. So action items for this mapping is **a) Check different Frame Timelines.**

<br/>

### 1.4 Сonclusions


In general it's very simple process, where we just shared all thoughts during Preprocessing. For some model, we should take Frames Timeline on Working Area and Normilize inputs. However we came up with several different options, which we should investigate (just combination of possible parameters, also marked above). 

<br/>

### Bonus. Error Resolving.

TBD

### Bonus. Frames Augmentation

TBD



