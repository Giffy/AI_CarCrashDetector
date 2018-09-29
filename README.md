# CarCrashDetector
This repository includes a Car crash detector from dashcam video using Convolutional neural network.

Every saturday from April to June 2018, we meet at AI Saturdays Barcelona to learn about Machine Learning and Deep Learning.
This code is the implementation of the techniques and theory that we have learned.

Team members:<br>
        <li>Alicia Escontrela
        <li>Christian Tutivén
        <li>Joan Melchor
        <li>Jordi Guix

We implemented the code in Python 3, with 'colab jupyter notebook' in Google Colaboratory.

# 1 Building a Dataset

To build a Dataset is the first challenge, sometimes there are huge datasets ready to be downloaded ( www.kaggle.com , in public organizations http://governobert.gencat.cat/en/dades_obertes/ , etc). Unfortunately, there was not any dataset available.
Where could we get thousands of videos recorded with dashboard camera? Right, in www.youtube.com

Youtube is the largest repository of videos with many examples of both car crashes (lots of compilations) and non-crashes. 
First challenge is to have consistent data. 
Main rules in the dataset creation were:<br>
    - location of the camera: should be a dashboard camera or recorded from similar location<br>
    - crashes between cars or car and truck (no motorbikes, trains, ...)<br>
    - light conditions: records during the day<br>
    - video quality at least 640x480 or above<br>
    - removed any cover with titles<br>
    - car crash accidents type<br>
    
First task was to download the candidate videos in order to process the images and homogenize the data.<br>
Then run the module '1_Building_a_Dataset' in google colaboratory.
We used OpenCV library to extract the frames, and scikit-image to modify and resize them. Frames are converted to 640 pixels width and from colors to grayscale.

# 2 Dataset preparation

Once images are processed, starts the tough task which is to visualize and classify the images.
Images should be stored manually in two folders accidents and no-accidents.

When images are allocated in the right folder, it is the time to run the next module '2_Dataset_preparation' to:

Split the supervised images in two groups, one for train and another for valid which each contains images of accidents and no-accidents. The module splits them randomly and does data augmentation of accident images (which are fewer than no-accident). 

It splits in train (80% of supervised image set) and valid (20% of supervised image set) i.e. to test it.

# 3 Model train

We followed fastai course during the AI Saturdays. To train the model, we apply the knowledge adquired and fastai libraries.
To create the model we selected ResNet-34 architecture. 
ResNet-34 is a pre-trained Model for PyTorch to work with the images. By using a pre-trained model we are saving time. Someone else has already spent the time and computed resources to learn a lot of features and our model will likely benefit from it.


# 4 Prediction

To analyze if a video have an accident, with current coding, it is required to upload a video to a folder.
Then video is split to frames and processed to homogenize the images to grayscale and downscale the width to 640 pixels.  
Then we load the trained model and process the images to generate a preliminar prediction.
We normalize the preliminar prediction to remove false positives due to model accuracy, difficult light conditions in image or low quality, ...
Once the preliminar prediction is normalized, we analyze it to determine if there is any accident in the video.

# 5 All the best and get started.
