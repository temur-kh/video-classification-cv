# Video Classification

The repository builds a two approaches for video classification (or action recognition) using [UCF101](http://crcv.ucf.edu/data/UCF101.php) with PyTorch. To propagate a video through a model, we randomly select a specific number of frames from it. Below are two neural nets models:


## Dataset

![alt text](./fig/kayaking.gif)

[UCF101](http://crcv.ucf.edu/data/UCF101.php) has total 13,320 videos from 101 actions. Videos have various time lengths (frames) and different 2d image size; the shortest is 28 frames. 

You may find a Google Colab notebook with some exploratory data analysis following this link:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fzQNT85RXqCrQHICnMJZz3csiUAv02N_?usp=sharing)

## Models 

### 1. Avg CNN Model

Videos are viewed as 3d images randomly chosen from the time length and passed through a pretrained CNN model (for instance, ResNet18 pretrained on ImageNet). For each video, the logit outputs of images are averaged to return a single logit and classify the video. 

### 2. CNN to RNN Model

The model is a pair of CNN encoder and RNN decoder (see figure below):

  - **[encoder]** A [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network) function encodes (meaning compressing dimension) every 2D image **x(t)** into a 1D vector **z(t)** by <img src="./fig/f_CNN.png" width="140">

  - **[decoder]** A [RNN](https://en.wikipedia.org/wiki/Recurrent_neural_network) receives a sequence input vectors **z(t)** from the CNN encoder and outputs another 1D sequence **h(t)**. A final fully-connected neural net is concatenated at the end for categorical predictions. Additionally, when the model is being trained, there is a dropout layer in between the RNN and a fully-connected layer.
  - For a CNN model, we use the existing models pretrained on ImageNet.

## Training $ testing
- Training dataset contains 9788 videos.
- Validation dataset contains 202 videos.
- Testing dataset contains 3330 videos.
- A single epoch of training and validation takes approximately 1.5-2 hours on Google Colab with the default parameters of the application.

## Usage

### 0. Prerequisites
You may install all the required libraries with pip:
```
pip install -r requirements.txt
```

### 1. Download a UCF101 dataset

You may directly download the dataset from the original website and decompress it in the root directory of the project with the following code:
```
!wget --no-check-certificate "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
!unrar e UCF101.rar /content/data/videos/
```

### 2. Train a model
You may run the model with the default parameters without passing any arguments. It will train the model and generate predictions:
```
python train.py
```

For more information about arguments to pass. Refer to the `help` message:
```
usage: Script for baseline training. Model is ResNet18 (pretrained on ImageNet). Training takes ~ 15 mins (@ GTX 1080Ti).
       [-h] [--name NAME] [--data DATA] [--batch-size BATCH_SIZE]
       [--frames-cnt FRAMES_CNT] [--model-type MODEL_TYPE] [--epochs EPOCHS]
       [--learning-rate LEARNING_RATE] [--gpu] [--predict]

optional arguments:
  -h, --help            show this help message and exit
  --name NAME, -n NAME  Experiment name (for saving best models and prediction
                        results).
  --data DATA, -d DATA  Path to dir with videos folder and train/test files.
  --batch-size BATCH_SIZE, -b BATCH_SIZE
                        Batch size.
  --frames-cnt FRAMES_CNT, -f FRAMES_CNT
                        Number of video frames for random selection.
  --model-type MODEL_TYPE, -m MODEL_TYPE
                        Model to run. Two options: 'cnn-avg' or 'cnn-rnn'.
  --epochs EPOCHS, -e EPOCHS
                        Number of training epochs.
  --learning-rate LEARNING_RATE, -lr LEARNING_RATE
                        Learning rate for the optimizer.
  --gpu                 Whether to run using GPU or not.
  --predict             Whether to only make predictions or to train a model,
                        too.
```



