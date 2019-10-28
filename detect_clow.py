import cv2  
import numpy as np  
from matplotlib import pyplot as plt  
from PIL import Image
  
def return_cow(img):
    img = img
#    img = address_regoin
    GrayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    ret,thresh1=cv2.threshold(GrayImage,100,255,cv2.THRESH_BINARY)
    (h,w)=thresh1.shape #返回高和宽
     
    a = [0 for z in range(0, h)] 
#    print(a) 
     
    for j in range(0,h):  
        for i in range(0,w):  
            if  thresh1[j,i]==0:
                a[j]+=1 
                thresh1[j,i]=255
             
    for j  in range(0,h):  
        for i in range(0,a[j]):
            thresh1[j,i]=0
#    plt.imshow(img,cmap=plt.gray())
#    plt.imshow(thresh1,cmap=plt.gray())
#    plt.show()
    total = 0
    if sum(a[0:int(h/3)]) >50:
        total  = total+1
    if sum(a[int(h/3):int(h/3*2)]) >5:
        total =  total+1
    if sum(a[int(h/3*2):int(h)]) >5:
        total = total +1
    return(total)
