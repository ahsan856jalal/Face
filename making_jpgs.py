#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:42:23 2020

@author: ahsanjalal
"""

import numpy as np
import argparse
import cv2
from pylab import * 
import os,sys
from os.path import join, isfile
import shutil

main_dir='/media/muhammadmubeen/data/datasets/celeb_10k/10000 snaps Celeb'
save_dir='/media/muhammadmubeen/data/datasets/celeb_10k_modified'
fols=os.listdir(main_dir)
for fol in fols:
    if not os.path.exists(join(save_dir,fol)):
        os.makedirs(join(save_dir,fol))
    names=os.listdir(join(main_dir,fol))
    print('{} is in proicess'.format(fol))
    counter=0
    for img_name in names:
        print(counter)
        counter+=1
        img=cv2.imread(join(main_dir,fol,img_name))
        if img is not None:
            cv2.imwrite(os.path.splitext(join(save_dir,fol,img_name))[0]+'.jpg',img)