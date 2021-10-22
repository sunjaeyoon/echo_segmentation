# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 10:52:45 2021

@author: jbt5jf
"""

import os
from os import listdir
from os.path import isfile, join, isdir
import matplotlib.pyplot as plt
import numpy as np
import imageio
from skimage.transform import resize
import tqdm
import cv2

from mask import makeMaskJSON


def get_Files(mypath:str = "mouse_heart", fileType:str='avi'):
    return [f for f in listdir(mypath) if isfile(join(mypath, f)) and f[-len(fileType):]==fileType]


def get_Folders(path = '.', keyword = None):
    if keyword == None:
        return [name for name in os.listdir(path) if isdir(name)]
    return [name for name in os.listdir(path) if isdir(name) and name.startswith(keyword)]

class Dataset(object):
    def __init__(self, path):
        
        self.datapath = path 
        self.images, self.labels = self.getData()
        self.shape = self.images.shape
    
    def getData(self, resize=(128,128)):
        """
        getData obtains all files that are pngs of the data created from 
        videos.py     
        """
        folders = get_Folders(self.datapath, '2021')
        imgstack = None
        maskstack = None
        
        print(folders)
        for folder in folders:
            jsonFiles = get_Files(folder,'json')
            #print(jsonFiles)
            
            if len(jsonFiles) != 0:
                
                for f in jsonFiles:
                    imgpath = os.path.join(self.datapath,folder,f.split('.')[0]+'.png')
                    img = cv2.resize(plt.imread(imgpath).astype(np.float), resize, interpolation = cv2.INTER_AREA)
                    maskpath = os.path.join(self.datapath,folder,f)
                    mask = cv2.resize(makeMaskJSON(maskpath).astype(np.float), resize, interpolation = cv2.INTER_AREA)
                    
                
                    if type(imgstack)!=np.ndarray:
                        imgstack = img.reshape([1,img.shape[0],img.shape[1]])
                        maskstack = mask.reshape([1,img.shape[0],img.shape[1]])
                        #print(imgstack.size)
                    else:
                        imgstack = np.concatenate([imgstack,img.reshape([1,img.shape[0],img.shape[1]])])
                        maskstack = np.concatenate([maskstack, mask.reshape([1,img.shape[0],img.shape[1]])])
                        #print(imgstack.shape)
                    
        #print(type(imgstack))
                
        return [imgstack, maskstack]
    
    def __getitem__(self, i):
        return self.images[i,:,:], self.labels[i,:,:]
    
    def segmentVideo(self, file, model):
        pass
        
 
    
 
    
 
    
def main():
    t = Dataset('.')
    
    for j in range(t.shape[0]):
        plt.figure(figsize=(15, 15))
        
        title = ['Input Image', 'True Mask']
        
        for i in range(len(title)):
            plt.subplot(1, len(title), i+1)
            plt.title(title[i])
            plt.imshow(t[j][i])
            plt.axis('off')
        plt.show()     
    
# TEST CODE
if __name__ == '__main__':
    main()