# -*- coding:utf-8 -*- 
import cv2
import copy
import math
import numpy as np
from matplotlib import pyplot as plt

img=cv2.imread(r'C:\Users\lenovo\Desktop\33.png')
#第一种反色变换
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#图像转换为灰度图
rows=img.shape[0]
cols=img.shape[1]
cover=copy.deepcopy(gray)
for i in range(rows):
    for j in range(cols):
        cover[i][j]=255-cover[i][j]
cv2.namedWindow('Image')
cv2.imshow('Image',cover)#展示灰度图
cv2.waitKey(0)
cv2.destroyAllWindows()
#分段函数
def SLT(img,x1,x2,y1,y2):
	lut=np.zeros(256)
	for i in range(256):
		if i<x1:
			lut[i]=y1/x1*i
		elif i<x2:
			lut[i]=(y2-y1)/(x2-x1)*(i-x1)+y1
		else:
			lut[i]=((y2-255.0)/(x2-255.0))*(i-255.0)+255.0
	img_output=cv2.LUT(img,lut)
	img_output=np.uint8(img_output+0.5)
	return img_output
img_x1 = 100
img_x2 = 160
img_y1 = 30
img_y2 = 230
#SLT_plot(img_x1, img_x2, img_y1, img_y2)
output_img=SLT(img,img_x1,img_x2,img_y1,img_y2)
cv2.imshow('output',output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
	
#对数变换
logc = copy.deepcopy(gray)
for i in range(rows):
    for j in range(cols):
        logc[i][j] =3*math.log(1+logc[i][j])
cv2.namedWindow('Image')
cv2.imshow('Image',logc)#展示灰度图
cv2.waitKey(0)
cv2.destroyAllWindows()
#二值化
a,gray2=cv2.threshold(img,50,255,cv2.THRESH_BINARY)
cv2.namedWindow('Image')
cv2.imshow("Image",gray2)
cv2.waitKey(0)
cv2.destroyAllWindows()
#伽马变换1
image=np.power(gray /255.0,2.2)#伽马变换，直接法，直接幂次方
cv2.namedWindow('Image')
cv2.imshow("Image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#伽马变换2
def gamma_trans(img,gamma):#伽马变换，复杂法
	#具体做法先归一化到1，然后gamma作为指数值求出新的像素值再还原
	gamma_table = [np.power(x/255.0,gamma)*255.0 for x in range(256)]
	gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
	#实现映射用的是Opencv的查表函数
	return cv2.LUT(img,gamma_table)
image3=gamma_trans(gray,2.2)
cv2.namedWindow('Image')
cv2.imshow("Image",image3)
cv2.waitKey(0)
cv2.destroyAllWindows()
#灰度级分层
def GrayLayer(img):
    lut = np.zeros(256,dtype=np.uint8)
    layer1=30
    layer2=60
    value1=10
    value2=250
    for i in range(256):
        if i >= layer2:
            lut[i] = value1
        elif i >= layer1:
            lut[i]=value2
        else:
            lut[i]=value1
    ans = cv2.LUT(img,lut)
    return ans
img_output = GrayLayer(img)
cv2.imshow('output', img_output)
cv2.waitKey(0)
cv2.destroyAllWindows()


