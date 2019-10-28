#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 09:18:19 2019

@author: lx
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from imutils.perspective import four_point_transform
import time
import math
import torch
import uuid
from PIL import Image
from torchvision import transforms
device = torch.device('cuda')
net=torch.load('detect.pth')
net=net.to(device)
transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
                            ])
#读入图片
def rotate_bound(image,angle):
    #获取图像的尺寸
    #旋转中心
    (h,w) = image.shape[:2]
    (cx,cy) = (w/2,h/2)
    
    #设置旋转矩阵
    M = cv2.getRotationMatrix2D((cx,cy),-angle,1.0)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])
    
    # 计算图像旋转后的新边界
    nW = int((h*sin)+(w*cos))
    nH = int((h*cos)+(w*sin))
    
    # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
    M[0,2] += (nW/2) - cx
    M[1,2] += (nH/2) - cy
    return cv2.warpAffine(image,M,(nW,nH),borderValue=(255,255,255))
def line_detection(image_crop):
    h ,w = image_crop.shape[:-1]
    image_crop = image_crop[int(7/8*h):h,:,:]
#    plt.imshow(image_crop )
    gray = cv2.cvtColor(image_crop, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(thresh, 30, 70)
#    edges = cv2.Canny(gray, 50, 160, apertureSize=3)  #apertureSize参数默认其实就是3
#    plt.imshow(edges)
#    plt.imshow(thresh )
    lines = cv2.HoughLines(edges , 1, np.pi/180, 80)
    jiaodu = []
    if lines is None:
        return -1
    for line in lines:
        rho, theta = line[0]  #line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的。
        jiaodu.append(math.degrees(theta))

    return jiaodu 
def aliged(file):
#    T = time.time()
#    file = "back6.png"
    if type(file) == str:
        image = cv2.imread(file)
    else:
        image = file
    #转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #高斯滤波
#    plt.imshow(gray,cmap=plt.gray())
    
    h, w = gray.shape
    gray = gray[10:h-10,10:w-10]
        
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    #自适应二值化方法
    blurred = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,51,2)
    '''
    adaptiveThreshold函数：第一个参数src指原图像，原图像应该是灰度图。
        第二个参数x指当像素值高于（有时是小于）阈值时应该被赋予的新的像素值
        第三个参数adaptive_method 指： CV_ADAPTIVE_THRESH_MEAN_C 或 CV_ADAPTIVE_THRESH_GAUSSIAN_C
        第四个参数threshold_type  指取阈值类型：必须是下者之一  
                                     •  CV_THRESH_BINARY,
                            • CV_THRESH_BINARY_INV
         第五个参数 block_size 指用来计算阈值的象素邻域大小: 3, 5, 7, ...
        第六个参数param1    指与方法有关的参数。对方法CV_ADAPTIVE_THRESH_MEAN_C 和 CV_ADAPTIVE_THRESH_GAUSSIAN_C， 它是一个从均值或加权均值提取的常数, 尽管它可以是负数。
    '''
    #这一步可有可无，主要是增加一圈白框，以免刚好卷子边框压线后期边缘检测无果。好的样本图就不用考虑这种问题
    blurred = cv2.copyMakeBorder(blurred,5,5,5,5,cv2.BORDER_CONSTANT,value=(255,255,255))
    
    #canny边缘检测
    edged = cv2.Canny(blurred, 10, 100)
    # 从边缘图中寻找轮廓，然后初始化答题卡对应的轮廓
    '''
    findContours
    image -- 要查找轮廓的原图像
    mode -- 轮廓的检索模式，它有四种模式：
         cv2.RETR_EXTERNAL  表示只检测外轮廓                                  
         cv2.RETR_LIST 检测的轮廓不建立等级关系
         cv2.RETR_CCOMP 建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，
                  这个物体的边界也在顶层。
         cv2.RETR_TREE 建立一个等级树结构的轮廓。
    method --  轮廓的近似办法：
         cv2.CHAIN_APPROX_NONE 存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max （abs (x1 - x2), abs(y2 - y1) == 1
         cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需
                           4个点来保存轮廓信息
          cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh-Chinl chain 近似算法
    '''
    thresh1 = edged.copy()
    (h,w)=thresh1.shape #返回高和宽
    # print(h,w)#s输出高和宽
    #print(a) #a = [0,0,0,0,0,0,0,0,0,0,...,0,0]初始化一个长度为w的数组，用于记录每一列的黑点个数  
     
    #记录每一列的波峰
    thresh1[thresh1 == 255 ] = 1
    a = np.sum(thresh1,0)

    x = np.where(a>0)
    x_p = [x[0][0],x[0][-1]]
    #此时的thresh1便是一张图像向垂直方向上投影的直方图

    thresh1 = edged.copy()
#    thresh1[thresh1 == 255 ] = 1
    b = np.sum(thresh1,1)
    y = np.where(b>0)
    y_p = [y[0][0],y[0][-1]]
    
    if y_p[0] > 10 and  (h - y_p[1]) > 10:
        N = 10
    else:
        N = 0
    if x_p[0] > 10 and  (w - x_p[1]) > 10:
        M = 10
    else:
        M = 0
    image_crop = image[y_p[0]-N:y_p[1] + N,x_p[0]-M:x_p[1]+M,:]
#    plt.imshow(image_crop)
#    print(time.time() - T )
#    D = line_detection(image_crop)
#    D = np.mean(90 - np.array( D))
#    src_align = rotate_bound(image_crop,D)
#    cv2.imwrite('1234.png',src_align)
#    src_align = match( src_align)
    return image_crop
#    cv2.imwrite('1234.png',src_align)

# img2 =Image.open('0002.jpg')
# img2.convert('L')
# img_1 = np.array(img2)
#plt.imshow(thresh1,cmap=plt.gray())
#plt.show()

def prediect(image_crop):
#    net=torch.load('detect.pth')
#    net=net.to(device)
    torch.no_grad()
    img=Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
    img=transform(img).unsqueeze(0)
    img_ = img.to(device)
    outputs = net(img_)
    _, predicted = torch.max(outputs, 1)
    # print(predicted)
    if predicted[0].cpu().numpy() > 0.8:
        return True
    else:
        return False
#    print('this picture maybe :',classes[predicted[0].cpu()])
def match(src_align,flag):
    total = []
    add = []
    img_gray = cv2.cvtColor(src_align, cv2.COLOR_BGR2GRAY)
    for i,f in enumerate (flag):
        add = []
        
        template = cv2.imread('template/'+f,0)
        w, h = template.shape[::-1]
        
        res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
        pt = maxLoc
        pt, (pt[0] + w, pt[1] + h)
        add.append(str(i))
        add.append(str(pt[0]))
        add.append(str(pt[1]))
        add.append(str(pt[0] + w))
        add.append(str(pt[1] + h))
        total.append(add)
    return total


def MASK(src_align,ff):
    img_gray = cv2.cvtColor(src_align, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('template/'+'mask.png',0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
    pt = maxLoc
    mask = np.ones(src_align.shape[:-1],dtype = 'uint8') * 255
    mask[pt[1]:pt[1] + h,pt[0]:pt[0] + w] = template
#    plt.imshow( img_gray)
#    mask[mask == 255 ] = 0
    if pt[0] > 135 :
        pass
    else:
        mean_img = np.mean(img_gray[50:100,300:400])
        mask[mask < 170 ] = 0
        mask[mask >= 170 ] = 255
        mask = 255 - mask
        new = mask + img_gray
        masked = cv2.add(img_gray, np.zeros(np.shape(img_gray), dtype=np.uint8), mask=mask)
        mean_mask = np.mean(masked[masked>0])
    #    mask = ~ mask
#    cv2.imwrite('MASK/'+ ff[:-4] + '.jpg'  ,src_align)
#    cv2.imwrite('MASK/'+ ff[:-4] + '.png'  ,mask)
#    imgroi = cv2.bitwise_and(mask,img_gray) 
#    aver1 = np.sum(imgroi) / (np.sum(mask)/255)
    
#    aver = np.mean(img_gray)
##    dst = cv2.inpaint(src_align,mask,3,cv2.INPAINT_TELEA)
        r = ( mean_img - mean_mask) /255
#    print(r)
#    r = 0.2
        id_image = img_gray + 0.15* mask
        
        
#def MASK(src_align,ff):
#    img_gray = cv2.cvtColor(src_align, cv2.COLOR_BGR2GRAY)
#    template = cv2.imread('template/'+'mask.png',0)
#    w, h = template.shape[::-1]
#    mean = np.mean(img_gray[50:150,])
#    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
#    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
#    pt = maxLoc
#    mask = np.ones(src_align.shape[:-1],dtype = 'uint8') * 255
#    mask[pt[1]:pt[1] + h,pt[0]:pt[0] + w] = template
##    plt.imshow( img_gray)
##    mask[mask == 255 ] = 0
#    if pt[0] > 135 :
#        pass
#    else:
#        mean_img = np.mean(img_gray[50:100,300:400])
#        mask[mask < 170 ] = 0
#        mask[mask >= 170 ] = 255
#        mask = 255 - mask
#        new = mask + img_gray
#        masked = cv2.add(img_gray, np.zeros(np.shape(img_gray), dtype=np.uint8), mask=mask)
#        mean_mask = np.mean(masked[masked>0])
#    #    mask = ~ mask
##    cv2.imwrite('MASK/'+ ff[:-4] + '.jpg'  ,src_align)
##    cv2.imwrite('MASK/'+ ff[:-4] + '.png'  ,mask)
##    imgroi = cv2.bitwise_and(mask,img_gray) 
##    aver1 = np.sum(imgroi) / (np.sum(mask)/255)
#    
##    aver = np.mean(img_gray)
###    dst = cv2.inpaint(src_align,mask,3,cv2.INPAINT_TELEA)
#        r = ( mean_img - mean_mask) /255
##    print(r)
##    r = 0.2
#        id_image = img_gray + 0.15* mask
#        id_image = 255 - img_gray 
    
#    threshold = 0.3
#    loc = np.where( res >= threshold)
#    for pt in zip(*loc[::-1]):
#        cv2.rectangle(src_align, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
#    cv2.rectangle(src_align, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
#    plt.imshow(src_align)
#    plt.imshow(  id_image )
    

#def detect_jiaodu(src_align):
#    
#    img_gray = cv2.cvtColor(src_align, cv2.COLOR_BGR2GRAY)
#    template = cv2.imread('template/BDCI.png', 0)
#    h, w = template.shape[:2]
#    
#    # 2.标准相关模板匹配
#    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
#    threshold = 0.7
#    
#    # 3.这边是Python/Numpy的知识，后面解释
#    loc = np.where(res >= threshold)  # 匹配程度大于%80的坐标y,x
##    for pt in zip(*loc[::-1]):  # *号表示可选参数
##        right_bottom = (pt[0] + w, pt[1] + h)
##        cv2.rectangle(img_rgb, pt, right_bottom, (0, 0, 255), 2)
#    a = 0
#    for l in loc:
#        if l.any():
#            a = a + 1
#    a = a / len(loc)
#    if a < 0.2:
#        return True
#    else:
#        return False
def first(file,f):
    face = ['name.png','sex.png','race.png','birth.png','address.png','number.png']
    back = ['jiguan.png','riqi.png']
#    file = img_face
#    ff = file
#    img_gray,id_image = aliged(file)
    image_crop = aliged(file)
    direction = prediect(image_crop)
    if direction :
        image_crop = rotate_bound(image_crop,180)
    D = line_detection(image_crop)
    D = np.mean(90 - np.array(D))
    src_align = rotate_bound(image_crop,D)
#    src_align = file
#    cv2.imwrite('result/'+ ff[:-4] + '.png'  ,mask)
#    MASK(src_align,ff)
#        plt.imshow(file)
#    plt.imshow(src_align)
#    plt.imshow(image_crop)
#        plt.imshow(src_align)
#    try :
#        D = line_detection(image_crop)
#        s = np.std(np.array(D), ddof=1)
#        print(np.std(np.array(D), ddof=1),'2')
#        if np.isnan(s):
#            image_crop = rotate_bound(image_crop,180)
#            D = line_detection(image_crop)
#            s = 0
#    except:
#        image_crop = rotate_bound(image_crop,180)
#        D = line_detection(image_crop)
#        print(np.std(D, ddof=1),'2')
#    D = np.mean(90 - np.array( D))
#    src_align = rotate_bound(image_crop,D)
#    plt.imshow(src_align)
    if f:
        flag = face
#        cv2.imwrite('result/'+ ff[:-4] + '-face.jpg'  ,src_align)
    else:
        flag = back
#        cv2.imwrite('result/'+ ff[:-4] + '-back.jpg'  ,src_align)
    address = match( src_align,flag)
#    id_image = id_image.astype(np.uint8)
##    id_image = cv2.cvtColor(id_image, cv2.COLOR_GRAY2BGR)
#    target =  cv2.cvtColor(id_image, cv2.COLOR_BGR2RGB)
    return src_align,address
#if __name__ == '__main__':
#    
##    cv2.imwrite('res.png',img)
#    plt.imshow(img_gray,cmap='gray') # 显示灰度图片
##    plt.subplot(212)
#    plt.imshow(id_image,cmap='gray') 
#    plt.show()
#    cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE) 
#    cv2.imshow('1',img_orig)
#    cv2.imshow('2',cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
#    cv2.waitKey(0)
##    plt.imshow( img)
#    cv2.destroyAllWindows()
    
#for 
#id_image = img_gray + 0.2 * mask
#plt.imshow( id_image)
#    import os
#    for file in os.listdir('result'):
#        file_name = os.path.join('result',file)
#        for f in os.listdir(file_name):
#            fold = os.path.join(file_name,f)
#            aliged(fold)  
################################################
#img = cv2.imread('crop.png')
#house = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 获取灰度图
#edges = cv2.Canny(house, 150, 200)
#lines = cv2.HoughLines(edges, 1, np.pi/180, 260)  # 霍夫变换返回的就是极坐标系中的两个参数  rho和theta
#print(np.shape(lines))
#lines = lines[:, 0, :]  # 将数据转换到二维
#for rho, theta in lines:
#    a = np.cos(theta)
#    b = np.sin(theta)
#    # 从图b中可以看出x0 = rho x cos(theta)
#    #               y0 = rho x sin(theta)
#    x0 = a*rho
#    y0 = b*rho
#    # 由参数空间向实际坐标点转换
#    x1 = int(x0 + 1000*(-b))
#    y1 = int(y0 + 1000*a)
#    x2 = int(x0 - 1000*(-b))
#    y2 = int(y0 - 1000*a)
#    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
#cv2.imshow('img', img)
#cv2.imshow('edges', edges)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
