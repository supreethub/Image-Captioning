# Image-Captioning
## Description
This project is a combination of Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) knowledge, to build a deep learning model that produces captions given an input image.

Image captioning requires that you create a complex deep learning model with two components: a CNN that transforms an input image into a set of features, and an RNN that turns those features into rich, descriptive language.

## Files
Dataset.ipynb: Explore MS COCO dataset using COCO API
Preliminaries.ipynb: Load and pre-process data from the MS COCO dataset and design the CNN-RNN model for automatically generating image captions
Training.ipynb: Training phase of the CNN-RNN model
Inference.ipynb: Using the previously trained model to generate captions for images in the test dataset.
data_loader.py : Custom data loader for PyTorch combining the dataset and the sampler
vocabulary.py : Vocabulary constructor built from the captions in the training dataset
vocab.pkl : Vocabulary file stored to load it immediately from the data loader

## CNN Encoder
The encoder is based on a Convolutional neural network that encodes an image into a compact representation.

The CNN-Encoder is a ResNet (Residual Network). These kind of network help regarding to the vanishing and exploding gradient type of problems. The main idea relies on the use of skip connections which allows to take the activations from one layer and suddenly feed it to another layer, even much deeper in the neural network and using that, we can build ResNets which enables to train very deep networks. In this project I used the ResNet-152 pre-trained model, which among those available from PyTorch : https://pytorch.org/docs/master/torchvision/models.html , is the one that is performing best on the ImageNet dataset.

## RNN Decoder
The CNN encoder is followed by a recurrent neural network that generates a corresponding sentence.

The RNN-Decoder consists in a single LSTM layer followed by one fully-connected (linear) layer, this architecture was presented from the paper Show and Tell: A Neural Image Caption Generator (2014) https://arxiv.org/pdf/1411.4555.pdf (figure 3.1)

## CNN-RNN model

By merging the CNN encoder and the RNN decoder, a model is generated that can find patterns in images and then use that information to help generate a description of those images. The input image will be processed by a CNN and we will connect the output of the CNN to the input of the RNN which will allow us to generate descriptive text.
