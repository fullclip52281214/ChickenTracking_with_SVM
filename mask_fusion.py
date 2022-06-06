import cv2,os
import numpy as np

for (root,dirs,files) in os.walk('masks', topdown=True): 
    pass
    #print(files) 

sumMask=np.zeros((1080,1920),dtype=np.uint8)

for file in files:
    stdimg=cv2.imread("masks/"+file,0)

    sumMask=cv2.add(sumMask,stdimg)
    
t,sumMask=cv2.threshold(sumMask,200,255,cv2.THRESH_BINARY)
cv2.imwrite("sumMask.png",sumMask)