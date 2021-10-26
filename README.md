# echo_segmentation

This repositiory attempts image segmentation of echocardiograms of mice experiments at the University of VIrginia Department of Biomedical Engineering.

## Background 
Myocardial infract induces scarring in the tissue wall that prevents proper electral signaling from occuring. The improper signaling is reflected on the overall physiology of the heart as uncoordinated contractions of the heart cannot pumped blood out of the heart. Diagnosis of this issue is a challenge because of the contraints of modern imaging modalities in addition to the challenge of developing real time solutions. This repositiory is the first steps to using ultrasound data for epicardial abnoramlities.

### LabelMe
LabelMe is a GUI tool used to create segmentation masks for the dataset. To get started, look at https://github.com/wkentaro/labelme. TLDR: use 
```
conda install labelme -c conda-forge 
```
and in the command line type 
```
labelme
```
Limitations:
  this tool is that it cannot annotate videos -> I created videos.py which contains a function that saves every individual frame into a new folder.
  The json file create expects the format with two custom label names 'inner' and 'outter'
  
### File structure
Data is expected to be within a folder called "mouse_heart"
  
# Python Code

Files
- videos.py
- masks.py
- Dataset.py
- UNET_working.py (training)
