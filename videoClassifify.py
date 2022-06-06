import numpy as np
import cv2

from InductiveColorRange import svm

rendering_resolution=0.667  # >=0.08

SVM=svm()
img=cv2.imread("test_1.avi_000025208.png")
mask=cv2.imread("sumMask.bmp",0)

#優化訓練資料
h,w,d=img.shape
imgroi=img[int(h*0.07):,int(0.2*w):int(0.886*w),:]
maskroi=mask[int(h*0.07):,int(0.2*w):int(0.886*w)]

bestmodel=SVM.Model_optimize(imgroi,maskroi,3)
SVM.setModel(bestmodel)

cap=cv2.VideoCapture("video/test_1.avi")
i=0
while(cap.isOpened()):
    ret,frame = cap.read()
    
    h,w,d=img.shape
    #範圍外，不視為雞
    frameroi=frame[int(h*0.07):,int(0.2*w):int(0.886*w),:]
    
    result=SVM.classifier(frameroi,rendering_resolution,False)
    frame=SVM.Image_Fusion(frameroi,rendering_resolution,result,frame)
    cv2.imshow('frame',frame)#cv2.resize(frame,(1920,1080)))
    c=cv2.waitKey(1)
    if ( c==27 ):  #ESC
        break
    i+=1
    cv2.imwrite("VideoOutput/"+str(i)+".jpg",frame)
    
cap.release()
cv2.destroyAllWindows()