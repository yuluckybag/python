# -*- coding:utf-8 -*- 
import cv2 
import numpy as np
from matplotlib import pyplot as plt

i=0
img=cv2.imread(r'C:\Users\lenovo\Desktop\33.png')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#图像转换为灰度图   
color=('g','b','r')
#img = cv2.GaussianBlur(img,(3,3),0)#高斯滤波

while i<3:#循环绘出RBG的直方图 
	hist=cv2.calcHist([img],[i],None,[256],[0,256])#统计
	#hists=plt.hist(img.ravel(),256,[0,256])
	plt.plot(hist,color[i])#绘图
	plt.xlim(0,256)#控制x轴大小
	i+=1
plt.show()#展示
cv2.namedWindow('Image')
cv2.imshow('Image',gray)#展示灰度图
cv2.waitKey(0)
cv2.destroyAllWindows()
hist2=cv2.calcHist([gray],[0],None,[256],[0,255])#灰度图转直方
plt.plot(hist2)
plt.xlim(0,256)
plt.show()


