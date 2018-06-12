# CarCrashDetector
This repository includes a Car crash detector from dashcam video using Convolutional neural network.

Every saturday from April to June 2018, we meet at AI Saturdays Barcelona to learn about Machine Learning and Deep Learning.
This code is the implementation of the techniques and theory that we have learned.

Team members:     Alicia Escontrela
                  Christian Tutivén
                  Joan Melchor
                  Jordi Guix

We implemented the code in Python 3, with 'colab jupyter notebook' in Google Colaboratory.

# Building a Dataset

To build a Dataset is the first challenge, sometimes there are huge datasets ready to be downloaded ( www.kaggle.com , in public organizations http://governobert.gencat.cat/en/dades_obertes/ , etc). Unfortunatelly, there was not any dataset available.
Where could we get thousands of videos recorded with dshboard camera? Right, in www.youtube.com

Youtube is the largest repository of videos with many examples of both car crashes (lots of compilations) and non-crashes. 
Fist challenge is to have consistent data. 
Main rules in the dataset creation were:<br>
    - location of the camera: should be a dashboard camera or recorded from similar location
    - crashes between cars or car and truck (no motorbikes, trains, ...)
    - light conditions: records during the day 
    - video quality at least 640x480 or above
    - removed any cover with titles
    - car crash accidents 
    
First task was to download the candidate videos in order to process the images and homogenize the data.
We used OpenCV library to extract the frames and scikit-image to resize to 640 width and convert colors to grayscale.

