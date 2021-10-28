# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 16:03:46 2021

@author: jbt5jf
"""


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import *
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np

from Dataset import Dataset

from IPython.display import clear_output
import datetime
import cv2



#UNET 1 -----------------------------------------------------------------------
def UNET(img_size, num_classes):
    inputs = keras.Input(shape=img_size)

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = Reshape([-1, 128, 128, 1])(inputs)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same') (x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model

#UNET2 -----------------------------------------------------------------------

def UNET2(img_size, num_classes):
    
    
    inputs = Input(img_size)
    s = inputs
    
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)
    
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)
    
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)
    
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)
    
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)
    
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)
    
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)
    
    #o =  Reshape(img_size[-1,...])(c9)
    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid') (c9)
    
    model = keras.Model(inputs=[inputs], outputs=[outputs])
    
    return model


#------------------------------------------------------------------------------
#CUSTOM CALLBACK 

trainstack = np.zeros((50,128,256))

COUNT = 0

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]



def show_predictions(dataset=None, num=1):
    """
    pred_mask = model.predict(image)
    display([image[0], mask[0], create_mask(pred_mask)])
    """
    pred = model.predict(train_x[0][tf.newaxis, ...])
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle('Prediction')
    ax1.imshow(train_x[0].reshape(128,128))
    ax2.imshow(train_y[0].reshape(128,128))
    ax3.imshow(pred[0,:,:,:].reshape(128,128))
    plt.show()
    global COUNT
    trainstack[COUNT] = np.concatenate([train_x[0].reshape(128,128)*255, pred[0,:,:,:].reshape(128,128)*255], axis=1)
    
    
    COUNT = COUNT + 1
    print("traincounter", COUNT)
    #plt.imshow(pred[0,:,:,:].reshape(256,256))


class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

#-----------------------------------------------------------------------------

def quickReshapeNetwork(img,  shape = [-1, 128, 128, 1]):
    return img.reshape(shape)

def reshapeOutput(img, shape=[-1,128,128]):
    return img.reshape(shape)

#-----------------------------------------------------------------------------
#MAIN SECTION
#if __name__ == "__main__":
# GLOBAL VARS 
   
img_size = np.array([128, 128, 1])
num_classes = 1
batch_size = 16
EPOCHS = 50
save = True
output_path = '.'


#Dataset -----------------------------------------------------------------
data = Dataset('.')

#Reshape images and labels b/c tf expects matrix with size (num_samples, width, height, channels)
train_x = quickReshapeNetwork(data.images)
train_y = quickReshapeNetwork(data.labels)


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()


# Build model-------------------------------------------------------------
model = UNET2(img_size, num_classes)
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.BinaryCrossentropy(),
    # List of metrics to monitor
    metrics=['accuracy']
)


#Train model--------------------------------------------------------------
model.fit(train_x, train_y, epochs=EPOCHS, callbacks=[DisplayCallback()], batch_size=20)


#Test Model---------------------------------------------------------------
pred = model.predict(train_x[0][tf.newaxis, ...])

# Output Prediction shape is (-1 128 128 1) -> reshape to 128 128
x = reshapeOutput(pred)[0]
y = data.images[0]

plt.imshow(x*y)


#Save Model
model_file = str(datetime.datetime.now()).split('.')[0].replace(' ', '_').replace(':','-') 
if save:
    print("saving model in", model_file)
    model.save(f"{model_file}model.h5")
    out = cv2.VideoWriter("training.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (256, 128), False)
    for i in range(50):
        out.write(trainstack[i].astype('uint8'))
    out.release()
        















