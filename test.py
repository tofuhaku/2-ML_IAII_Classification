import numpy as np
import cv2
import matplotlib.pyplot as plt
import sklearn
from collections import Counter
import glob
import pickle
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from mpl_toolkits.axes_grid1 import AxesGrid
from sklearn.metrics import confusion_matrix
from keras.optimizers import SGD, Adam
from random import shuffle
import os

import torch

WEIGHT_PATH = 'simpsons_cnn_weights.h5'
TRAIN_DIR = 'D:/Programs/ML/ML_IAII/2-ML_IAII_Classification/dataset/train'
TEST_DIR = 'D:/Programs/ML/ML_IAII/2-ML_IAII_Classification/dataset/test-renamed_images'

IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")