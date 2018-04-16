# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 21:42:42 2017

@author: FurryMonster Yang
"""

import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import os, cv2
import glob
from PIL import Image
ia.seed(10)
img_path = r'C:\Users\FurryMonster Yang\Documents\Python Scripts\trainingdata\Normal'
path = os.path.join(img_path,'*tif')
files = glob.glob(path)
# Example batch of images.
# The array has shape (32, 64, 64, 3) and dtype uint8.

seq = iaa.Sequential([
iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
iaa.Fliplr(0.1), 
iaa.ContrastNormalization((0.9, 1.1)),
iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        rotate=(-10, 10),
        shear=(-8, 8)
    )# horizontally flip 50% of the images
    #iaa.GaussianBlur(sigma=(0, 5.0)) # blur images with a sigma of 0 to 3.0
], random_order=True)

i = 0

for file in files:
    # 'images' should be either a 4D numpy array of shape (N, height, width, channels)
    # or a list of 3D numpy arrays, each having shape (height, width, channels).
    # Grayscale images must have shape (height, width, 1) each.
    # All images must have numpy's dtype uint8. Values are expected to be in
    # range 0-255.
   # if 'n0'  or 'iv0' or 'is0' or 'b0' in file:
        image = cv2.imread(file)
        images_aug = seq.augment_image(image)
        new_file_name = 'aug_Normal' + str(i) + '.tif'
        img = Image.fromarray(images_aug, 'RGB')
        full_dir = img_path+'/' + new_file_name
        img.save(full_dir,'TIFF')
        #images_aug.save(new_file_name,img_path)
        i += 1
        
    
#seq = iaa.Sequential([
#iaa.Crop(px=(0, 128)), # crop images from each side by 0 to 16px (randomly chosen)
#iaa.Fliplr(1), 
#iaa.ContrastNormalization((0.75, 1.5)),
#iaa.Affine(
#        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
#        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
#        rotate=(-25, 25),
#        shear=(-8, 8)
#    )# horizontally flip 50% of the images
#    #iaa.GaussianBlur(sigma=(0, 5.0)) # blur images with a sigma of 0 to 3.0
#], random_order=True)

#
#for file in files:
#    # 'images' should be either a 4D numpy array of shape (N, height, width, channels)
#    # or a list of 3D numpy arrays, each having shape (height, width, channels).
#    # Grayscale images must have shape (height, width, 1) each.
#    # All images must have numpy's dtype uint8. Values are expected to be in
#    # range 0-255.
#    image = cv2.imread(file)
#    images_aug = seq.augment_image(image)
#    new_file_name = 'aug_Benign_2' + str(i) + '.tif'
#    img = Image.fromarray(images_aug, 'RGB')
#    img.save(new_file_name)
#    #images_aug.save(new_file_name,img_path)
#    i += 1