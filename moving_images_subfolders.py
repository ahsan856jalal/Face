#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 12:55:24 2020

@author: muhammadmubeen
"""

import numpy as np
import argparse
import cv2
from pylab import * 
import os,sys
from os.path import join, isfile
import shutil
main_dir='/home/muhammadmubeen/google-download/downloads_10k'
celebs=os.listdir(main_dir)
counter=1
for name in celebs:
    os.chdir(join(main_dir,name))
    sub_fol=os.listdir(join(main_dir,name))
    for sub in sub_fol:
        filenames=os.listdir(join(main_dir,name,sub))
        for files in filenames:
            if os.path.exists(join(main_dir,name,files)):
                files1=os.path.splitext(files)[0]+'_{}'.format(counter)+os.path.splitext(files)[1]
                counter+=1
            shutil.move(join(main_dir,name,sub,files),join(main_dir,name,files))
    