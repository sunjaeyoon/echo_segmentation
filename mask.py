# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 13:59:01 2021

@author: jbt5jf
"""

from labelme.utils import shape_to_mask
import json

import numpy as np
import matplotlib.pyplot as plt
from videos import get_Files

def makeWallMaskJSON(file:str):
    with open(file, "r",encoding="utf-8") as f:
        dj = json.load(f)
    # dj['shapes'][0]Is for one label this time.
    mask_in =   shape_to_mask((dj['imageHeight'],dj['imageWidth']), dj['shapes'][0]['points'], shape_type=None,line_width=1, point_size=1)
    mask_out = shape_to_mask((dj['imageHeight'],dj['imageWidth']), dj['shapes'][1]['points'], shape_type=None,line_width=1, point_size=1)
    mask_img1 = mask_in.astype(np.int)#boolean to 0,Convert to 1
    mask_img2 = mask_out.astype(np.int)
    plt.imshow(mask_img2-mask_img1)
    
def makeMaskJSON(file:str):
    with open(file, "r",encoding="utf-8") as f:
        dj = json.load(f)
    # dj['shapes'][0]Is for one label this time.
    mask_in =   shape_to_mask((dj['imageHeight'],dj['imageWidth']), dj['shapes'][0]['points'], shape_type=None,line_width=1, point_size=1)
    mask_img1 = mask_in.astype(np.int)#boolean to 0,Convert to 1
    #plt.imshow(mask_img1)
    return mask_img1

"""   
def createCSV(folder):
    files = get_Files(folder, 'json')
    
    with open('tracings.csv', 'w+') as write_f:
        for file in files:
            with open(path, "r",encoding="utf-8") as read_f:
                dj = json.load(f)
                # Format: 'FILENAME x1 y1 x2 y2 Frame'
    pass
"""

if __name__ == "__main__":
    path = "C:\\Users\\jbt5jf\\Documents\\EchoSeg\\2021-10-13-13-46-37_2-2011-05-01-13-16-23_1\\frame0000.json"
    mask = makeMaskJSON(path)