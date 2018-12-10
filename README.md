# Toy-Model-Checking

## 1. Preprocessing

This section will include all work we did, and other possible thoughts. We have 17 min training video, with 20 frames per second. It's 20400 frames, where we have speed on each frame. From the initial description, we can start with some basics on it. 1. Features taken from Images 2. Features elapsed over time. 



### 1.1 Working Area
From each frame, we can decide working area, which includes most of the features, depending on our task. And reduce number of the Inputs. Even we did some feature area investigation, it's easy to manually retrive most useful part.


<img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/source-image.jpg" width="400" height="300" /> <img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/source-feature-area.png" width="400" height="300" /> 

1.1.1. Source Image and Source Image Areas. Red Box represents Working Area, for the next steps. 
<br/> <br/>

But we have another interesting question here. As you can see, we have Working Area, which consist from `Close` and `Far` subareas. First one, has more features about road changes (has more changes over time, due of camera angle). And Second one with more noise like side objects, other cars and less road changes. 


<img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/car-angle-variants.jpg" width="800" height="400" /> 

1.1.2. Frame changes updates in different way, based on Angle of a Camera. 
<br/> <br/>

At this point, we have few variants, for next preprocessing steps. We will use this options in the next phases. a) Using Far Sub Area. b) Using Mid Sub Area. c) Using Close Sub Area. d) Using complete Working Area. 

*Note. It's very important to reduce number of Inputs, especialy in such cases, where we working with Video, and features elapsed over time. One wrong step will cause your model to have the Curse of Dimensionality. That is why we suppose to avoid last variant with using complete Working Area.*

<img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/image-mov-top-0.jpg" width="280" height="170" /> <img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/image-mov-mid-0.jpg" width="280" height="170" /> <img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/image-mov-bot-0.jpg" width="280" height="170" /> 

1.1.3. Examples of Sub Area subtract from complete Working Area. LTR Far Area. Mid. Close.
<br/> <br/>


Below animation, how we can estimate it, with changes over time. Note, this sample only for visualizing and deciding about next steps. because correct Area we can choose, only after using some Model and testing feature extraction on each frame. 

<img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/image-mov-top-0.gif" width="280" height="170" /> <img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/image-mov-mid-0.gif" width="280" height="170" /> <img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/image-mov-bot-0.gif" width="280" height="170" /> 

1.1.3. Examples of Sub Area features changing over time. LTR Far Area. Mid. Close.
<br/> <br/>

As a result, we come with few options for next work. a) Check Model velocity using several Sub Areas Types. b) Check Model velocity over all possible Sub Areas, with momentum over some shifting.  



### 1.2 Scaling 

Scaling takes some part in preprocessing, even we can simplify and avoid it's usage. Scale working in two directions, and for some cases might reduce a Dimensionality in two different ways. The simplest example here it's scaling down image from source to lower one (ex. 500x500 scale down to 250x250). But in some resources, you can find opossite usage of scaling, by increasing image size (ex. 125x125 scale up to 250x250).

<img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/1.2/scale-source.gif" width="280" height="170" /> <img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/1.2/scale-image.gif" width="280" height="170" /> <img src="https://raw.githubusercontent.com/GensaGames/Toy-Model-Checking/master/files/1.2/scale-compressed.gif" width="280" height="170" />

1.2.1. Feature changing over time, with Scaling. LTR Source Image. Scaled. Compressed.
<br/> <br/>
