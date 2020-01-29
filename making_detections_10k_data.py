#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 15:40:09 2020

@author: muhammadmubeen
"""


import sys,os,glob
from os.path import join, isfile
import numpy as np
from pylab import *
import cv2
# import dlib
from scipy.misc import imresize
import numpy
#from rgb2gray import rgb2gray
import scipy.misc
from natsort import natsorted, ns
from shutil import copytree
import matplotlib.pyplot as plt
import glob
import os
import PIL
from ctypes import *
import math
import random
bkg_count=0
###################### incorporating darknet.py ######################################################

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("/home/muhammadmubeen/darknet_alexay_latest/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.2, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res
a=open('/home/muhammadmubeen/face_dnn/CAFFE_DNN/celeb_names.txt','r')
celebrities=a.readlines()
celebrities=[bb.rstrip() for bb in celebrities]
specie_list=celebrities
net = load_net("/home/muhammadmubeen/darknet_alexay_latest/cfg/yolov3_pak_celeb.cfg", "/home/muhammadmubeen/darknet_alexay_latest/backup_pak_celeb/yolov3_pak_celeb_99000.weights", 0)
meta = load_meta("/home/muhammadmubeen/darknet_alexay_latest/cfg/pak_celeb.data")

main_dir='/media/muhammadmubeen/data/datasets/celeb_10k_modified'
celebs=os.listdir(main_dir)
save_dir='/media/muhammadmubeen/data/datasets/celeb_10k_with_boxes'
for name in celebs:
    counter=1
    if not os.path.exists(join(save_dir,name)):
        os.makedirs(join(save_dir,name))
    os.chdir(join(main_dir,name))
#    filenames=glob.glob(join(main_dir,name)+'/*.jpg')
#    print('images in this folder are {}'.format(len(filenames)))
    filenames=os.listdir(join(main_dir,name))
    for img_name in filenames:
        print(counter)
        counter+=1
        obj_arr = []
        img=cv2.imread(img_name)
        if img is not None:
            [gt_height,gt_width,ch]=shape(img)
            
            r = detect(net, meta, img_name)
            f = open(os.path.splitext(join(save_dir,name,img_name))[0]+'.txt', "w")
            #f.write(xml_content)
            f.close()

            if r:
                    
                if r[0][1]>0.4 or r[0][0]=='Unknown':
                    for celeb_info in r:
                        celeb_name_det=celeb_info[0]
                        celeb_lab_det=specie_list.index(celeb_name_det)
                        x=(celeb_info[2][0])
                        y=(celeb_info[2][1])
                        w=(celeb_info[2][2])
                        h=(celeb_info[2][3])
                        xmin_det = int(x - w/2)
                        ymin_det = int(y - h/2)
                        xmax_det = int(x + w/2)
                        ymax_det = int(y + h/2)
                        y_text = ymin_det - 10 if ymin_det - 10 > 10 else ymin_det + 10
                #cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0, 0, 255), 2)
                #cv2.putText(image, text, (xmin,y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                        img=cv2.rectangle(img,(xmin_det,ymin_det),(xmax_det,ymax_det),(255,12,0),2)
                        cv2.putText(img,celeb_name_det,(xmin_det,y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,0),1,cv2.LINE_AA)
                        x1 = x / gt_width
                        y1 = y / gt_height
                        w1 = float(w) / gt_width
                        h1 = float(h) / gt_height
                        tmp = [celeb_lab_det, x1, y1, w1, h1]
                        obj_arr.append(tmp)
                    xml_content = ""
                    for obj in obj_arr:
                        xml_content += "%d %f %f %f %f\n" % (obj[0], obj[1], obj[2], obj[3], obj[4])
                    f = open(os.path.splitext(join(save_dir,name,img_name))[0]+'.txt', "w")
                    f.write(xml_content)
                    f.close()
                    cv2.imwrite(join(save_dir,name,img_name),img)

        else:
            os.remove(img_name)
                                    
       




