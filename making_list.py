#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 14:21:20 2020

@author: muhammadmubeen
"""

import numpy as np
import argparse
import cv2
from pylab import * 
import os,sys
from os.path import join, isfile


celeb_path='/home/muhammadmubeen/google-download/celeb_classifier_data'
celeb_names=os.listdir(celeb_path)
image_list=[]
for celeb_name in celeb_names:
    all_file_path=os.listdir(join(celeb_path,celeb_name))
    os.chdir(join(celeb_path,celeb_name))
    for img_name in all_file_path:
        if(img_name.endswith('.txt')):
            c=1
        else:
            image_list.append(join(celeb_path,celeb_name,img_name))


a=open('/home/muhammadmubeen/google-download/train_face.list','w')
a.writelines(["%s\n" % item  for item in image_list])
a.close()