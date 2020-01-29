# import the necessary packages
import numpy as np
import argparse
import cv2
from pylab import * 
import os,sys
from os.path import join, isfile
os.chdir('/home/muhammadmubeen/face_dnn/CAFFE_DNN')
confidence_th=0.5
prototxt='/home/muhammadmubeen/face_dnn/CAFFE_DNN/deploy.prototxt.txt'
model='/home/muhammadmubeen/face_dnn/CAFFE_DNN/res10_300x300_ssd_iter_140000.caffemodel'
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

a=open('celeb_names.txt','r')
celebrities=a.readlines()
celebrities=[bb.rstrip() for bb in celebrities]
# now looping over all celeb data
celeb_path='/home/muhammadmubeen/google-images-download/celeb_classifier_data'
celeb_names=os.listdir(celeb_path)
for celeb_name in celeb_names:
    all_img_path=os.listdir(join(celeb_path,celeb_name))
    os.chdir(join(celeb_path,celeb_name))
    print('total images in {} folder are {}'.format(celeb_name,len(all_img_path)))
    for img_name in all_img_path:
        text_filename=os.path.splitext(img_name)[0]
        image = cv2.imread(img_name)
        (h, w) = image.shape[:2]
        f = open(join(celeb_path,celeb_name,text_filename)+'.txt', "w")
        f.close()
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        # pass the blob through the network and obtain the detections and
        # predictions
        #print("[INFO] computing object detections...")
        net.setInput(blob)
        detections = net.forward()
        count = 0
        
        # loop over the detections
        obj_arr=[]
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            #print(i,confidence)
            if confidence > confidence_th:
                count += 1
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (xmin, ymin, xmax, ymax) = box.astype("int")
                #text = "{:.2f}%".format(confidence * 100) + 'Count ' + str(count)
                #y_text = ymin - 10 if ymin - 10 > 10 else ymin + 10
                #cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0, 0, 255), 2)
                #cv2.putText(image, text, (xmin,y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                b_w=xmax-xmin
                b_h=ymax-ymin
                x=(xmin+b_w/2.0)/w
                y=(ymin+b_h/2.0)/h
                w_new=float(b_w)/w
                h_new=float(b_h)/h
                obj_label=int(celebrities.index(celeb_name))
                tmp=[obj_label,x,y,w_new,h_new]
                obj_arr.append(tmp)
                
        # now saving into text files
        xml_content = ""
        for obj in obj_arr:
            xml_content += "%d %f %f %f %f\n" % (obj[0], obj[1], obj[2], obj[3], obj[4])
        #    if not os.path.exists(optical_save_dir):
        #        os.makedirs(optical_save_dir)        
            f = open(join(celeb_path,celeb_name,text_filename)+'.txt', "w")
            f.write(xml_content)
            f.close()
#            cv2.imwrite('ahsan_new.jpg',image)
    


#print('Count ', count)
## show the output image
#cv2.imshow("Output", image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
