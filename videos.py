# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 13:59:00 2021

@author: jbt5jf

This File will read and separate a video into frames
"""

import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
import imageio
from skimage.transform import resize
import tqdm
import cv2

input_folder = "./mouse_heart/"


def get_Files(mypath:str = "mouse_heart", fileType:str='avi'):
    return [f for f in listdir(mypath) if isfile(join(mypath, f)) and f[-3:]==fileType]


def sepVideos(video:str, crop=[[107,492],[227,638]], resize=(256,256), save=True)->np.ndarray:
    """"
    Function will take a video file and returns np array of video
    Inputs: 
        file: input file
        crop: [[x1,y1], [x2,y2]] defaults to [[107,492],[227,638]]
        resize: tuple (width, height) defaults to (256,256)
        save: bool defaults to true
    
    """
    
    output = video.split(".")[0]
    videostack = None
    if not os.path.exists(output): 
        os.makedirs(output)
	
    print('Segmenting', video, '...')
    images = np.array(imageio.mimread(input_folder+video, memtest=False))
    if len(images.shape) == 4:
        images = images[:,:,:,0] # convert RGB to grayscale
    reader = imageio.get_reader(input_folder+video, 'ffmpeg')
    fps = reader.get_meta_data()['fps']
	
    for i,image in enumerate(images):
        print(i+1,'/', images.shape[0], 'frames')
        img = cv2.resize(image[crop[0][0]:crop[0][1],crop[1][0]:crop[1][1]], resize)
        
        if save: 
            cv2.imwrite('./{0}/frame{1:04d}.png'.format(output,i), img)
        
        if type(videostack)!=np.ndarray:
            videostack = img.reshape([1,img.shape[0],img.shape[1]])
        else:
            videostack = np.concatenate([videostack,img.reshape([1,img.shape[0],img.shape[1]])])                 
    return videostack
                                        
                                                                                  
if __name__ == "__main__":    
    videos = [f for f in os.listdir(input_folder) if os.path.isfile(input_folder+f) and f[-3:]=='avi']
    
    for video in videos:
        if not os.path.exists(input_folder+video): os.makedirs(input_folder+video)
        print('Spliting', video, '...')
        x = sepVideos(video, save=False)#, crop = [[0, -1],[0,-1]])
        
    print('Separation complete.')