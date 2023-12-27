#!/usr/bin/env python
# coding: utf-8

# #Import Dataset

# ##Import Needed Packages
# 

# In[1]:


# import the necessary packages
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, RMSprop
from tensorflow.keras.initializers import RandomNormal, RandomUniform
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalHinge
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K


# In[2]:


from keras.preprocessing.image import ImageDataGenerator


# In[3]:


from os import listdir


# ##Load Dataset

# Download dataset from Kaggle

# In[4]:


import os

ORIG_INPUT_DATASET = "Dataset"
BASE_PATH = "DatasetNew"

# Derive the training, validation, and testing directories
TRAIN_PATH = os.path.join(BASE_PATH, "training")
VAL_PATH = os.path.join(BASE_PATH, "validation")
TEST_PATH = os.path.join(BASE_PATH, "testing")

# Define the amount of data that will be used for training
TRAIN_SPLIT = 0.8

# The amount of validation data will be a percentage of the
# training data
VAL_SPLIT = 0.1


# In[5]:


from imutils import paths
import random
import shutil
import os
# grab the paths to all input images in the original input directory
# and shuffle them
imagePaths = list(paths.list_images(ORIG_INPUT_DATASET))
random.seed(42)
random.shuffle(imagePaths)
# compute the training and testing split
i = int(len(imagePaths) * TRAIN_SPLIT)
trainPaths = imagePaths[:i]
testPaths = imagePaths[i:]
# we'll be using part of the training data for validation
i = int(len(trainPaths) * VAL_SPLIT)
valPaths = trainPaths[:i]
trainPaths = trainPaths[i:]
# define the datasets that we'll be building
datasets = [
	("training", trainPaths, TRAIN_PATH),
	("validation", valPaths, VAL_PATH),
	("testing", testPaths, TEST_PATH)
]


# In[6]:


# loop over the datasets
for (dType, imagePaths, baseOutput) in datasets:
	# show which data split we are creating
	print("[INFO] building '{}' split".format(dType))
	# if the output base output directory does not exist, create it
	if not os.path.exists(baseOutput):
		print("[INFO] 'creating {}' directory".format(baseOutput))
		os.makedirs(baseOutput)
	# loop over the input image paths
	for inputPath in imagePaths:
		# extract the filename of the input image and extract the
		# class label ("0" for "negative" and "1" for "positive")
		filename = inputPath.split(os.path.sep)[-1]
		label = filename[-5:-4]
		# build the path to the label directory
		labelPath = os.path.sep.join([baseOutput, label])
		# if the label output directory does not exist, create it
		if not os.path.exists(labelPath):
			print("[INFO] 'creating {}' directory".format(labelPath))
			os.makedirs(labelPath)
		# construct the path to the destination image and then copy
		# the image itself
		p = os.path.sep.join([labelPath, filename])
		shutil.copy2(inputPath, p)# Define Model
 


# In[7]:


class CancerNet:
  @staticmethod
  def build(width, height, depth, classes):
    height, width, depth = 50, 50, 3  # Assuming 3 channels for RGB images
    inputShape = (height, width, depth)
    chanDim = -1
    if K.image_data_format() == "channels_first":
      inputShape = (depth, height, width)
      chanDim = 1
    model = Sequential()
    # CONV => RELU => POOL
    model.add(SeparableConv2D(32, (3, 3), padding="same", input_shape = inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # (CONV => RELU => POOL) * 2
    model.add(SeparableConv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(SeparableConv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # (CONV => RELU => POOL) * 3
    model.add(SeparableConv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(SeparableConv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(SeparableConv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))
    # return the constructed network architecture
    return model


# In[4]:


# ! pip install kaggle
# # ! mkdir ~/.kaggle
# ! /content/kaggle.json
# # ! chmod 600 ~/.kaggle/kaggle.json
# ! kaggle datasets download paultimothymooney/breast-histopathology-images


# In[5]:


# ! unzip /content/Dataset/breast-histopathology-images.zip


# In[10]:


# path = ("Dataset")

# data = ImageDataGenerator()
# dataset = data.flow_from_directory(path, target_size = (50, 50), batch_size = 32, class_mode = 'categorical')



# In[11]:


# # Find the total length of data/Find out how many patients are there

# files = listdir(path)
# print("Total Number of Patients: "+ str(len(files)))


# # Put PNGs to array (0 or 1)

# In[12]:


# dataset = []

# for i in range(len(files)):
#     patient_id = files[i]
#     for c in [0,1]:
#         patient_path = path + '/' +  patient_id
#         class_path = patient_path + '/' + str(c) + '/'
#         if (patient_id[0] != '.'):
#           subfiles = listdir(class_path)
#           for pic in subfiles:
#               image_path = class_path + pic
#               dataset.append([image_path,c])


# In[ ]:


# dataset[0]
# dataset[1]


# # Divide Dataset to Train,  Validation, Testing

# Split Dataset into Training, Validation, Testing

# In[ ]:


# from sklearn.model_selection import train_test_split

# print("Dataset size:", len(dataset))

# # Extract images and labels from the dataset
# images = [item[0] for item in dataset]
# labels = [item[1] for item in dataset]

# # Split into train, test, and validation sets
# # First, split into train and temp (which combines validation and test)
# train_images, temp_images, train_labels, temp_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# # Split temp into validation and test
# validation_images, test_images, validation_labels, test_labels = train_test_split(temp_images, temp_labels, test_size=0.5, random_state=42)

# # Print the sizes of each set
# print("Train set size:", len(train_images))
# print("Validation set size:", len(validation_images))
# print("Test set size:", len(test_images))



# # Define Model

#  Aet the input shape and channels dimension for building a neural network model (presumably using Keras) based on the provided image dimensions and the backend's data format (either "channels_last" or "channels_first").

# In[ ]:


# class CancerNet:
# 	@staticmethod
# 	def build(width, height, depth, classes):
# 		height, width, depth = 50, 50, 3  # Assuming 3 channels for RGB images
# 		inputShape = (height, width, depth)
# 		chanDim = -1

# 		if K.image_data_format() == "channels_first":
# 			inputShape = (depth, height, width)
# 			chanDim = 1


# Define the Convolutional Neural Network

# In[ ]:


# class CancerNet:
#   @staticmethod
#   def build(width, height, depth, classes):
#     height, width, depth = 50, 50, 3  # Assuming 3 channels for RGB images
#     inputShape = (height, width, depth)
#     chanDim = -1
#     if K.image_data_format() == "channels_first":
#       inputShape = (depth, height, width)
#       chanDim = 1
#     model = Sequential()
#     # CONV => RELU => POOL
#     model.add(SeparableConv2D(32, (3, 3), padding="same", input_shape = inputShape))
#     model.add(Activation("relu"))
#     model.add(BatchNormalization(axis=chanDim))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#     # (CONV => RELU => POOL) * 2
#     model.add(SeparableConv2D(64, (3, 3), padding="same"))
#     model.add(Activation("relu"))
#     model.add(BatchNormalization(axis=chanDim))
#     model.add(SeparableConv2D(64, (3, 3), padding="same"))
#     model.add(Activation("relu"))
#     model.add(BatchNormalization(axis=chanDim))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#     # (CONV => RELU => POOL) * 3
#     model.add(SeparableConv2D(128, (3, 3), padding="same"))
#     model.add(Activation("relu"))
#     model.add(BatchNormalization(axis=chanDim))
#     model.add(SeparableConv2D(128, (3, 3), padding="same"))
#     model.add(Activation("relu"))
#     model.add(BatchNormalization(axis=chanDim))
#     model.add(SeparableConv2D(128, (3, 3), padding="same"))
#     model.add(Activation("relu"))
#     model.add(BatchNormalization(axis=chanDim))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#     # first (and only) set of FC => RELU layers
#     model.add(Flatten())
#     model.add(Dense(256))
#     model.add(Activation("relu"))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.5))
#     # softmax classifier
#     model.add(Dense(classes))
#     model.add(Activation("softmax"))
#     # return the constructed network architecture
#     return model


# <!-- #Addtional Imports -->

# In[8]:


import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


# ## To be continued

# In[55]:


# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--plot", type=str, default="plot.png",
# 	help="path to output loss/accuracy plot")
# args = vars(ap.parse_args())


# - Number of epochs, learning rate
# - Calculate Weights

# In[24]:


# initialize our number of epochs, initial learning rate, and batch
# size
NUM_EPOCHS = 150
INIT_LR = 1e-2
BS = 32
# determine the total number of image paths in training, validation,
# and testing directories
trainPaths = list(paths.list_images(TRAIN_PATH))
totalTrain = len(trainPaths)
totalVal = len(list(paths.list_images(VAL_PATH)))
totalTest = len(list(paths.list_images(TEST_PATH)))
# calculate the total number of training images in each class and
# initialize a dictionary to store the class weights
trainLabels = [int(p.split(os.path.sep)[-2]) for p in trainPaths]
trainLabels = to_categorical(trainLabels)
classTotals = trainLabels.sum(axis=0)
classWeight = dict()
# loop over all classes and calculate the class weight
for i in range(0, len(classTotals)):
	classWeight[i] = classTotals.max() / classTotals[i]



# - initialize training data
# - initialize validation

# In[26]:


# initialize the training data augmentation object
trainAug = ImageDataGenerator(
	rescale=1 / 255.0,
	rotation_range=20,
	zoom_range=0.05,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.05,
	horizontal_flip=True,
	vertical_flip=True,
	fill_mode="nearest")
# initialize the validation (and testing) data augmentation object
valAug = ImageDataGenerator(rescale=1 / 255.0)


# - training generator
# - validation generator
# - testing generator

# In[27]:


# initialize the training generator
trainGen = trainAug.flow_from_directory(
	TRAIN_PATH,
	class_mode="categorical",
	target_size=(48, 48),
	color_mode="rgb",
	shuffle=True,
	batch_size=BS)
# initialize the validation generator
valGen = valAug.flow_from_directory(
	VAL_PATH,
	class_mode="categorical",
	target_size=(48, 48),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)
# initialize the testing generator
testGen = valAug.flow_from_directory(
	TEST_PATH,
	class_mode="categorical",
	target_size=(48, 48),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)


# - compile
# - fit model

# In[28]:


from scipy import ndimage


# In[29]:


# initialize our CancerNet model and compile it

model = CancerNet.build(width=48, height=48, depth=3,
	classes=2)
opt = Adagrad(learning_rate=INIT_LR, weight_decay=INIT_LR / NUM_EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
# fit the model
H = model.fit(
	x=trainGen,
	steps_per_epoch=totalTrain // BS,
	validation_data=valGen,
	validation_steps=totalVal // BS,
	class_weight=classWeight,
	epochs=NUM_EPOCHS)


# In[30]:


# reset the testing generator and then use our trained model to
# make predictions on the data
print("[INFO] evaluating network...")
testGen.reset()
predIdxs = model.predict(x=testGen, steps=(totalTest // BS) + 1)
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)
# show a nicely formatted classification report
print(classification_report(testGen.classes, predIdxs,
	target_names=testGen.class_indices.keys()))


# # Print Plot and Measures

# Compute Accuracy, Sensitivity, Specificity

# In[31]:


# compute the confusion matrix and and use it to derive the raw
# accuracy, sensitivity, and specificity
cm = confusion_matrix(testGen.classes, predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
# show the confusion matrix, accuracy, sensitivity, and specificity
print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))


# Plot

# In[32]:


get_ipython().run_line_magic('matplotlib', 'inline')
# plot the training loss and accuracy
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()
#plt.savefig(args["plot"])


# In[ ]:




