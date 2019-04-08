#!/usr/bin/env python
# coding: utf-8

# ## MPM Microscopy Data Processing ##
# 
# Use three images of each region of interest to create a processed version
#  - Take the average of the two TPEF images, use as red channel
#  - Use SHG image as blue channel
#  - Create a greyscale version from this
#  - Save both multi-channel and greyscale images
# 

# Converted from Jupyter notebook to run on server


import numpy as np
import matplotlib.pyplot as plt #requires Pillow to read tiffs
from os import walk #to get directory listing

from skimage.color import rgb2gray
from scipy import ndimage


# In[8]:


data_dir = '/data/jo/MPM Skin Deep Learning Project/'
subdirs = ['Dysplastic Tissue', 'Malignant Tissue', 'Healthy Tissue']
path = data_dir + '%s/%s/med/' #Substitute in subdirectory and slide ID to get full path

out_file = '/data/jo/MPM Skin Deep Learning Project/processed/%s_%d.tif' #substitute in slide ID and region ID
out_file_greyscale = '/data/jo/MPM Skin Deep Learning Project/processed/%s_%d_gs.tif' #substitute in slide ID and region ID


# In[22]:


#For each of the data subdirectories get a list of slides (each is a subdirectory of that)
#Then iterate the slides and process each region.
for sd in subdirs:
    (_, slidedirs, _) = next(walk(data_dir+sd))
    
    for slide in slidedirs:
        print('Processing slide', slide, 'in', sd)
        image_folder = path % (sd, slide) + '/'
        (_, _, images) = next(walk(image_folder))
        
        #Sort alphabetically to get the files for one region together. Then iterate through
        #regions and create the processed images
        images = sorted(images)
        numRegions = len(images)//3
        assert(numRegions*3 == len(images))
        
        print('Found %d regions'%(numRegions,))
        for roi in range(numRegions):
            shg = plt.imread(image_folder+images[3*roi])/255.
            blue_ch = np.reshape(shg, (*shg.shape,1))
            
            tpef1 = plt.imread(image_folder+images[3*roi + 1])
            tpef2 = plt.imread(image_folder+images[3*roi + 2])
            red_ch = ((tpef1 + tpef2) / 2)/255.
            red_ch = np.reshape(red_ch, (*red_ch.shape,1))

            processed = np.append(red_ch, np.zeros(red_ch.shape, dtype=int), axis=-1)
            processed = np.append(processed, blue_ch, axis=-1)

            #MPM images are rotated by 90 degrees counter clockwise compared to brightfield. Rotate back
            processed = ndimage.rotate(processed, -90)

            plt.imsave(out_file%(slide,roi), processed)
            plt.imsave(out_file_greyscale%(slide,roi), rgb2gray(processed), cmap='gray', vmin=0, vmax=1)




