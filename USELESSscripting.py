# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 10:09:24 2021

@author: jbt5jf
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
def main():
    input_folder = "./mouse_heart/"
    test = Dataset('.')
    
    videos = [f for f in os.listdir(input_folder) if os.path.isfile(input_folder+f) and f[-3:]=='avi']
    
    #for video in videos:
    video = videos[3]
    print(video)
    if not os.path.exists(input_folder+video): os.makedirs(input_folder+video)
    print('Spliting', video, '...')
    x = sepVideos(video, save=False, resize=(128,128))
    
    print(x.shape)
    segnet = tf.keras.models.load_model('525.h5')
    img, mask = test[7]
    plt.imshow(img)
    print(img.shape)
    pred = segnet.predict(img.reshape(128,128,1)[tf.newaxis,...])
    #print(pred.shape)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    ax2.imshow(pred.reshape(128,128))
    
    size = 128, 128
    
    fps = 10
    """
    #out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
    for i in range(pred.shape[0]):
        #out.write(pred[i,:,:]*255)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(pred[i])
        ax2.imshow(x[i].reshape(128,128))
    #out.release()
    """

#Try the network on the video
def main2():
    input_folder = "./mouse_heart/"
    test = Dataset('.')
    
    videos = [f for f in os.listdir(input_folder) if os.path.isfile(input_folder+f) and f[-3:]=='avi']
    
    for video in videos:
        #video = videos[3]
        print(video)
        if not os.path.exists(input_folder+video): os.makedirs(input_folder+video)
        print('Spliting', video, '...')
        x = sepVideos(video, save=False, resize=(128,128))
        
        print(x.shape)
        img = x[5]
        segnet = tf.keras.models.load_model('525.h5')
        
        
        pred = segnet.predict(x.reshape(-1,128,128,1))
        """
        pred = segnet.predict(img.reshape(128,128,1)[tf.newaxis,...])
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(img)
        ax2.imshow(pred.reshape(128,128))
        """
        size = 128, 128
    
        fps = 10
        out = cv2.VideoWriter(f'{video}_segmented.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
        for i in range(pred.shape[0]):
            out.write(pred[i,:,:]*255)
            #fig, (ax1, ax2) = plt.subplots(1, 2)
            #ax1.imshow(pred[i].reshape(128,128))
            #ax2.imshow(x[i].reshape(128,128))
        out.release()
        
if __name__ == "__main__":
    #pass
    keras.backend.clear_session()
    main()
    main2()
    
    
""" Create a video """




