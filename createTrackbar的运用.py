# -*- coding:utf-8 -*- 
import cv2
import copy
import math
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread(r'C:\Users\lenovo\Desktop\33.png')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#图像转换为灰度图
name="二值化"
def turn(count):#回调函数
	a,gray2=cv2.threshold(img,count,255,cv2.THRESH_BINARY)
	cv2.imshow(name,gray2)#展示灰度图
if __name__ == '__main__':	
	cv2.namedWindow(name)
	cv2.createTrackbar("pos",name,0,255,turn)
	turn(0)
	cv2.waitKey(0)
