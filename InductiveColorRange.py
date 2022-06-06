import cv2 
import numpy as np
import random

class svm:
    def __init__(self):
        self.svm=cv2.ml.SVM_create()
        
    def training(self,img,scale,background_ratio,mask,show):
        #整理數據
        w,h,d=img.shape
        h=int(h*scale)
        w=int(w*scale)
        img=cv2.resize(img,(h,w))
        HSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        black=np.zeros(HSV.shape,dtype=np.uint8)
        mask=cv2.resize(mask,(h,w))
        
        
        #分離前後景
        chicken=cv2.add(black,HSV,mask=mask)
        mask=255-mask
        background=cv2.add(black,HSV,mask=mask)
        
        
        if show : print("-----preprossing data------")
        
        TwoDimChicken=np.reshape(chicken[:,:,0:2],(-1,2)) #將彩色照片拉成直線
        newTwoDimChicken=[]
        for i in TwoDimChicken: #去除黑色區域，以免影響SVM訓練
            if(i[1]>0):
                newTwoDimChicken.append(i)
        TwoDimChicken=np.array(newTwoDimChicken)
        
        TwoDimBackground=np.reshape(background[:,:,0:2],(-1,2))#將彩色照片拉成直線，只使用色像跟飽和度
        newTwoDimBackground=[]
        for i in TwoDimBackground:#去除黑色區域，以免影響SVM訓練
            if(i[1]>0):
                newTwoDimBackground.append(i)
        TwoDimBackground=np.array(newTwoDimBackground)
        
        chickenPixels=len(TwoDimChicken)  #取得非黑色像素總量
        if show : print("chickenPixels=",chickenPixels)
        alable=np.ones((chickenPixels,1))
        
        backPixels=len(TwoDimBackground)
        if show : print("backPixels=",backPixels)
        
        randIndexList=random.sample(range(0,backPixels),int(chickenPixels*background_ratio))
        newTwoDimBackground=[]                  #使前景跟背景訓練用的資料量相同
        for i in randIndexList:                 #刪減背景訓練資料
            newTwoDimBackground.append(TwoDimBackground[i]) 
        TwoDimBackground=newTwoDimBackground 
        backPixels=len(TwoDimBackground)
        if show : print("new backPixels=",backPixels)
        blable=np.zeros((backPixels,1))
        
        lable=np.vstack((alable,blable))
        lable=np.array(lable,dtype='int32')
        
        TwoDimBackground=np.array(newTwoDimBackground) #List2Array
        
        data=np.vstack((TwoDimChicken,TwoDimBackground)) #合併陣列
        
        data=np.array(data,dtype='float32')
        
        if show : print("-----training start------")
        #訓練SVM
        self.svm.train(data,cv2.ml.ROW_SAMPLE,lable)  
        if show : print("-----training complete------")
        
        #if show : self.svm.save("Model/svmModel.xml")
        if not show : return self.svm
        
    def classifier(self,img,scale,show): #基於SVM的顏色分類器
        if show : print("-----predict------")
        #整理輸入數據
        img=cv2.resize(img,None,fx=scale,fy=scale)
        w,h,d=img.shape
        HSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        
        TwoDimHSV=np.reshape(HSV[:,:,:2],(-1,2))#圖形直線化 ，只使用色像跟飽和度
        pt_data = np.array(TwoDimHSV) #設定資料型態
        pt_data = np.array(pt_data, dtype = 'float32')
        #使用SVM進行預測
        predict = self.svm.predict(pt_data) 
        
        predictImg=np.reshape(predict[1],(w,h)) #將預測結果轉回二維
        def post_processing(img): #將預測結果進行後處理，使之更容易看懂
            scale_value=w/360
            kernal_diameter=9 #平滑濾波
            kernal=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (kernal_diameter,kernal_diameter))
            img=cv2.filter2D(img,cv2.CV_8U,kernal) #轉回uint8形式
            #設定值處理
            retval,img=cv2.threshold(img,40,255,cv2.THRESH_BINARY)
            #侵蝕
            kernal_diameter=int(5*scale_value) 
            kernal=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (kernal_diameter,kernal_diameter))
            img=cv2.morphologyEx(img,cv2.MORPH_ERODE,kernal)
            #膨脹
            kernal_diameter=int(9*scale_value)
            kernal=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (kernal_diameter,kernal_diameter))
            img=cv2.morphologyEx(img,cv2.MORPH_DILATE,kernal)
            #再侵蝕
            kernal_diameter=int(7*scale_value)
            kernal=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (kernal_diameter,kernal_diameter))
            img=cv2.morphologyEx(img,cv2.MORPH_ERODE,kernal)
            return img
        predictImg=post_processing(predictImg)
        

        
        
        if show : print("-----predict complete------")
        if show : cv2.imwrite("result.png",predictImg)
        return predictImg
    
    def Image_Fusion(self,img,scale,predictImg,oriimg): #將預測結果合併至原圖，產生彩色圖像
        img=cv2.resize(img,None,fx=scale,fy=scale)
        h,w,d=img.shape
        
        #標示記號顏色設定
        blue=np.ones((h,w),dtype=np.uint8)*0
        green=np.ones((h,w),dtype=np.uint8)*255
        red=np.ones((h,w),dtype=np.uint8)*0
        mark=cv2.merge([blue,green,red])

        
        mark=cv2.add(mark,mark,mask=predictImg)
        result=cv2.addWeighted(mark,0.23,img,1,gamma=0)
        #貼回原圖
        rh,rw,rd=result.shape
    
        oriimg=cv2.resize(oriimg,None,fx=scale,fy=scale)
        h,w,d=oriimg.shape
        
        oriimg[h-rh:h,int(0.2*w):int(0.2*w)+rw,:]=result
        return oriimg

    def Model_optimize(self,img,mask,times): #進行多次訓練取最優結果
        
        def errorCal(predictImg,mask): #誤差計算
            predictImg=np.array(predictImg,dtype=np.float32)
            mask=np.array(mask,dtype=np.float32) #以浮點數形式記錄負數
            predictImg=cv2.subtract(predictImg,mask)
            predictImg=np.absolute(predictImg)#絕對值
            return np.sum(predictImg) #像素值加總
        
        bestmodel=None
        besterror=1e+100
        mask=cv2.resize(mask,None,fx=0.25,fy=0.25)
        newmodel=cv2.ml.SVM_create()
        for t in range(times):
            newmodel=self.training(img,0.1,random.uniform(1,2.5),mask,False)
            predictImg=self.classifier(img,0.25,False) #模型透過self.svm傳遞，須注意
            newerror=errorCal(predictImg,mask)
            if( newerror<besterror ): #取代
                besterror=newerror
                bestmodel=newmodel
            print(newerror)
        return bestmodel
    
    def setModel(self,model):
        self.svm=model

if __name__ == "__main__":
    
    #設定值
    scale=0.1 #訓練時輸入圖片尺寸
    background_ratio=1#背景資料量增強
    watchScale=0.5 #渲染尺寸
    
    img=cv2.imread("test_1.avi_000025208.png") #訓練集
    mask=cv2.imread("sumMask.bmp",0)
    h,w,d=img.shape
    imgroi=img[int(h*0.07):,int(0.2*w):int(0.886*w),:]
    maskroi=mask[int(h*0.07):,int(0.2*w):int(0.886*w)]
    
    SVM=svm()
    SVM.training(imgroi,scale,background_ratio,maskroi,show=True)

    #bestmodel=SVM.Model_optimize(imgroi,maskroi,3)
    #SVM.setModel(bestmodel)
    
    result=SVM.classifier(imgroi,watchScale,show=True)  
    
    result=SVM.Image_Fusion(imgroi,watchScale,result,img)

    cv2.imshow("initial image",result)
    
    '''
    imgtest=cv2.imread("images/20200327125450.jpg") #測試集
    result=SVM.classifier(imgtest,watchScale,show=True)
    result=SVM.Image_Fusion(imgtest,watchScale,result)
    cv2.imshow("test image1",result)
    '''
    
    cv2.waitKey()
    cv2.destroyAllWindows()
