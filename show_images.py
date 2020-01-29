#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 13:00:27 2020

@author: muhammadmubeen
"""
import numpy as np
import argparse
import cv2
from pylab import * 
import os,sys
from os.path import join, isfile
import glob
font = cv2.FONT_HERSHEY_SIMPLEX
main_dir='/home/muhammadmubeen/google-images-download/celeb_classifier_data'
a=open('/home/muhammadmubeen/face_dnn/CAFFE_DNN/celeb_names.txt','r')
celebrities=a.readlines()
celebrities=[bb.rstrip() for bb in celebrities]
celeb='Aamir Hussain Liaquat'
image_files=glob.glob(main_dir+'/'+celeb+'/'+'*.jpg')
image_files=(image_files)
for img_name in image_files:
    img=cv2.imread(img_name)
    height,width,ch=shape(img)
    a=open(os.path.splitext(img_name)[0]+'.txt')
    text=a.readlines()
    for line in text:
        line = line.rstrip()
        coords=line.split(' ')
        face_id=int(coords[0])
        w=float(coords[3])*width
        h=float(coords[4])*height
        x=float(coords[1])*width
        y=float(coords[2])*height
        x=int(x)
        y=int(y)
        h=int(h)
        w=int(w)
        xmin = x - w/2
        ymin = y - h/2
        xmax = x + w/2
        ymax = y + h/2
        img=cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255,12,0),2)
        cv2.putText(img,'amir',(x+2+w/2,y+h/2), font, 0.5,(255,0,0),1,cv2.LINE_AA)
    
    cv2.imshow('asa',img)
    k = cv2.waitKey(200) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()