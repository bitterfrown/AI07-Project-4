# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 12:22:56 2022

Project 4: Image Segmentation
    
@author: mrob
"""
# Import necessary packages

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,losses,optimizers,callbacks
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
#%%

#1. Load the data for preparation

# Since the data has been kept in 2 different folders (train and test), create 2 different path for each folders.

train_data_path= r"C:\Users\captc\Desktop\AI_07\TensorFlow\Datasets_Online_Download\data-science-bowl-2018\data-science-bowl-2018-2\train"
test_data_path= r"C:\Users\captc\Desktop\AI_07\TensorFlow\Datasets_Online_Download\data-science-bowl-2018\data-science-bowl-2018-2\test"
#%%
# Load images and masks using OpenCV
#First, prepare an empty list for images and define a function to load the images.
train_images = []
train_masks = []
test_images = []
test_masks = []

train_image_dir = os.path.join(train_data_path,'inputs')
for image_file in sorted(os.listdir(train_image_dir)):
    train_img = cv2.imread(os.path.join(train_image_dir,image_file))
    train_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB)
    train_img = cv2.resize(train_img,(128,128))
    train_images.append(train_img)
    
test_image_dir = os.path.join(test_data_path,'inputs')
for image_file in sorted(os.listdir(test_image_dir)):
    test_img = cv2.imread(os.path.join(test_image_dir,image_file))
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    test_img = cv2.resize(test_img,(128,128))
    test_images.append(test_img)
    
train_mask_dir = os.path.join(train_data_path,'masks')
for mask_file in sorted(os.listdir(train_mask_dir)):
    train_mask = cv2.imread(os.path.join(train_mask_dir,mask_file),cv2.IMREAD_GRAYSCALE)
    train_mask = cv2.resize(train_mask,(128,128))
    train_masks.append(train_mask)
    
test_mask_dir = os.path.join(test_data_path,'masks')
for mask_file in sorted(os.listdir(test_mask_dir)):
    test_mask = cv2.imread(os.path.join(test_mask_dir,mask_file),cv2.IMREAD_GRAYSCALE)
    test_mask = cv2.resize(test_mask,(128,128))
    test_masks.append(test_mask)
#%%
#Convert the list of images and masks to numpy array
np_train_images= np.array(train_images)
np_train_masks= np.array(train_masks)
np_test_images= np.array(test_images)
np_test_masks= np.array(test_masks)
#%%
# Check the images

plt.figure(figsize=(10,10))
for i in range(1,5):
    plt.subplot(1,5,i)
    plt.imshow(np_train_images[i])
    plt.axis('off')
    
plt.show()

plt.figure(figsize=(10,10))
for i in range(1,5):
    plt.subplot(1,5,i)
    plt.imshow(np_test_masks[i])
    plt.axis('off')
    
plt.show()
#%%
#2. Data preprocessing

#2.1  Expand masks dimension
np_train_masks_exp= np.expand_dims(np_train_masks, axis=-1)
np_test_masks_exp= np.expand_dims(np_test_masks, axis=-1)

#Check the expanded mask output
print(np.unique(np_train_masks_exp[0]))
#%%
# Convert mask values into class labels

#2.2 Change the masks value (Encode into numeric coding 0 to 1)
converted_train_masks= 1- (np.round(np_train_masks/255).astype(np.int64))
converted_test_masks= 1- (np.round(np_test_masks/255).astype(np.int64))

#Check the converted mask output
print(np.unique(converted_train_masks[0]))
#%%
#2.3 Normalize the train and test images
converted_train_images= np_train_images/255.0
converted_test_images= np_test_images/255.0

#Check the converted train image
print(np.unique(converted_train_images[0]))
#%%
#2.4 Convert the numpy arrays into tensor slices

X_train_tensor= tf.data.Dataset.from_tensor_slices(converted_train_images)
X_test_tensor= tf.data.Dataset.from_tensor_slices(converted_test_images)
y_train_tensor= tf.data.Dataset.from_tensor_slices(converted_train_masks)
y_test_tensor= tf.data.Dataset.from_tensor_slices(converted_test_masks)

#2.5 Combine the images and masks using zip
train_dataset= tf.data.Dataset.zip((X_train_tensor,y_train_tensor))
test_dataset = tf.data.Dataset.zip((X_test_tensor,y_test_tensor))
#%%
# 2.7 Create a subclass layer for data augmentation

class Augment(layers.Layer):
    def __init__(self,seed=42):
        super().__init__()
        self.augment_inputs = layers.RandomFlip(mode='horizontal',seed=seed)
        self.augment_labels = layers.RandomFlip(mode='horizontal',seed=seed)
        
    def call(self,inputs,labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs,labels
#%%
#2.8 Convert into prefetch datasets

BATCH_SIZE= 16
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 500
TRAIN_SIZE= len(train_dataset)
STEPS_PER_EPOCH = TRAIN_SIZE//BATCH_SIZE

train_batches= (
    train_dataset
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    #.map(Augment())
    .prefetch(buffer_size= tf.data.AUTOTUNE)
    )

test_batches= (
    test_dataset 
    .cache()
    .batch(BATCH_SIZE)
    .repeat()
    #.map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
#%%
# 2.9 Visualize some examples

def display(display_list):
    plt.figure(figsize=(15,15))
    title = ['Input Image','True Mask','Predicted Mask']
    
    for i in range(len(display_list)):
        plt.subplot(1,len(display_list),i+1)
        plt.title(title[i])
        plt.imshow(keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
        
    plt.show()
    
for images, masks in train_batches.take(2):
    sample_image,sample_mask = images[0],masks[0]
    display([sample_image,sample_mask])
#%%
#3.Prepare the model

#3.1 Import pretrained model to be used as the feature extraction layers
base_model = keras.applications.MobileNetV2(input_shape=[128,128,3],include_top=False)

#3.2 List down some activation layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
    ]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

#Define the extraction layer
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
down_stack.trainable = False

#Define the upsampling path
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels:int):
    inputs = layers.Input(shape=[128,128,3])
    #Apply functional API to construct U-Net
    #Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])
    
    #Upsampling and establishing the skip connections(concatenation)
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = layers.Concatenate()
        x = concat([x,skip])
        
    #This is the last layer of the model (output layer)
    last = layers.Conv2DTranspose(
        filters=output_channels,kernel_size=3,strides=2,padding='same') #64x64 --> 128x128
    
    x = last(x)
    
    return keras.Model(inputs=inputs, outputs=x)
#%%
#3.3 Make of use of the function to construct the entire U-Net
OUTPUT_CLASSES = 2

model = unet_model(output_channels=OUTPUT_CLASSES)

#%%
# 3.4 Compile the model

loss = losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(keras.optimizers.Adam(learning_rate=0.0001),loss=loss,metrics=['accuracy'])
keras.utils.plot_model(model, show_shapes=True)

model.summary()

#%%
#3.6 Create functions to show predictions

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask,axis=-1)
    pred_mask = pred_mask[...,tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None,num=1):
    if dataset:
        for image,mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0],mask[0],create_mask(pred_mask)])
            
    else:
        display([sample_image,sample_mask,create_mask(model.predict(sample_image[tf.newaxis,...]))])

#%%
#3.7 Create a callback to help display results during model training

class DisplayCallbacks(callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch+1))
        
# 3.8 Define EarlyStopping callbacks
from tensorflow.keras.callbacks import EarlyStopping
es= tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,verbose=2)

#%%
#4. Furhter training the model

EPOCHS = 15
VAL_SUBSPLITS = 5
VALIDATION_STEPS = len(test_dataset)//BATCH_SIZE//VAL_SUBSPLITS

history = model.fit(train_batches,validation_data=test_batches,epochs=EPOCHS,steps_per_epoch=STEPS_PER_EPOCH,validation_steps=VALIDATION_STEPS,callbacks=[DisplayCallbacks(),es])

#%%
training_loss = history.history['loss']
val_loss = history.history['val_loss']
training_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs_x_axis = history.epoch

plt.plot(epochs_x_axis,training_loss,label='Training Loss')
plt.plot(epochs_x_axis,val_loss,label='Validation Loss')
plt.title("Training vs Validation Loss")
plt.legend()
plt.figure()

plt.plot(epochs_x_axis,training_acc,label='Training Accuracy')
plt.plot(epochs_x_axis,val_acc,label='Validation Accuracy')
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.figure()

plt.show()   

#%%
#5. Deploy the model























    
    