# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 10:09:24 2021

@author: jbt5jf

TESTING SCRIPT for the neural network

"""

import matplotlib.pyplot as plt
import numpy as np
import imageio
from skimage.transform import resize
import tqdm
import cv2

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import *
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np

from videos import sepVideos
import os
from Dataset import Dataset


#CHECK IF model is actaully saving correctly
def testModel(model, path = "./mouse_heart/"):
    input_folder = path
    test = Dataset('.')
    
    videos = [f for f in os.listdir(input_folder) if os.path.isfile(input_folder+f) and f[-3:]=='avi']
    
    #for video in videos:
    video = videos[3]
    print(video)
    if not os.path.exists(input_folder+video): os.makedirs(input_folder+video)
    print('Spliting', video, '...')
    x = sepVideos(video, save=False, resize=(128,128))
    
    print(x.shape)
    segnet = tf.keras.models.load_model('2021-10-25_17-02-21model'+'.h5')
    
    for i in range(test.shape[0]):
        img, mask = test[i]

        pred = segnet.predict(img.reshape(128,128,1)[tf.newaxis,...])

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(img)
        ax2.imshow(pred.reshape(128,128))
        plt.show()
        break


#Try the network on the video
def testVideo(model, path = "./mouse_heart/"):
    input_folder = path
    test = Dataset('.')
    
    videos = [f for f in os.listdir(input_folder) if os.path.isfile(input_folder+f) and f[-3:]=='avi']
    
    for video in videos:
        print(video)
        if not os.path.exists(input_folder+video): os.makedirs(input_folder+video)
        print('Spliting', video, '...')
        x = sepVideos(video, save=False, resize=(128,128))
        
        #print(x.shape)
        
        segnet = tf.keras.models.load_model(model)
        pred = segnet.predict(x.reshape(-1,128,128,1)).reshape(-1,128,128)
        
        """ DEBUG STUFF
        pred = segnet.predict(img.reshape(128,128,1)[tf.newaxis,...])
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(img)
        ax2.imshow(pred.reshape(128,128))
        """
        size = 128, 128*2 # Height, Width
        
        fps = 10
        print(pred.shape)
        
        out = cv2.VideoWriter(f"{video.split('.')[0]}_segmented.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
        for i in range(pred.shape[0]):
            test = np.concatenate([x[i], pred[i]*255], axis=1).astype('uint8')
            out.write(test)
        out.release()
        break
        
        
        
if __name__ == "__main__":
    keras.backend.clear_session()
    model = '2021-10-26_12-18-12model'+'.h5'
    testModel(model)
    testVideo(model)





