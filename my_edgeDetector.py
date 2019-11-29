import cv2
import numpy as np
from my_filtering import my_filtering
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

def find_zerocrossing(LoG): #0이 아니라 ,thresh를 사용하는건 실수 연산의 오차 고려.
    '''
    :param LoG: zero-crossing을 검사할 LoG 필터링 된 Image
    :param thresh: 실수 연산에서 생기는 오차가 있을 수 있기 때문에, 0이 아니라 thresh를 이용해서 수행.
                   (지우고 0으로 zero-crossing을 검사해도 괜찮습니다.)
    :return: zero-crossing 지점만 255값을 가지는 이미지.
    '''
    y, x = len(LoG), len(LoG[0])
    res = np.zeros((y,x), dtype=np.uint8)
    for i in range(1, y-1): #맨 처음과 맨 마지막은 제외.(검사에서 경계를 넘어가지 않도록)
        for j in range(1, x-1):
            # zero-crossing을 검사하는 코드를 작성해주세요.
           if LoG[i-1,j-1] * LoG[i+1,j+1] < 0:
               res[i, j] = 255
           elif LoG[i-1,j] * LoG[i+1,j] < 0:
               res[i, j] = 255
           elif LoG[i+1,j-1] * LoG[i-1,j+1] < 0:
               res[i, j] = 255
           elif LoG[i,j-1] * LoG[i,j+1] < 0:
               res[i, j] = 255

           else:
               res[i,j] = 0

    return res


def my_LoG(img, ksize=7, boundary = 0):  # default sigma =1, sigma = 0.3(n/2 -1) + 0.8
    '''
    :param img: LoG edge detection을 수행할 이미지.
    :param ksize: Kernel size. ksize x ksize의 kernel 사용.
    :param boundary: filtering 경계 처리 방법. 0 : zero-padding,(default) 1 : repetition, 2 : mirroring
    :return: LoG 방법으로 찾아낸 Edge 이미지.
    '''
    m = ksize//2
    sigma = 0.3 * (m - 1) + 0.8
    y, x = np.mgrid[-m: m + 1, -m: m+ 1]  # make kernel size of matrix
    g = - ( x * x + y * y ) / (2 * sigma ** 2)
    LoG = -(1.0+g)*np.exp(g)/(np.pi * sigma**4)  # apply gaussian formula to matrix

    LoG_img = my_filtering(img, LoG, boundary=boundary) # LoG는 만들어진 kernel
    LoG_img = find_zerocrossing(LoG_img)
    return LoG_img

def my_DoG(img, ksize, sigma = 1, gx = 0, boundary = 0): #default (3,3) sigma = 1, y축 편미분
    '''
    :param img: DoG edge detection을 수행할 이미지.
    :param ksize: Kernel size. ksize x ksize kernel 사용.
    :param sigma: Gaussian 분포에서 사용하는 표준편차.
    :param gx: 0 : y축 편미분, 1 : x축 편미분
    :param boundary: filtering 경계 처리 방법. 0 : zero-padding,(default) 1 : repetition, 2 : mirroring
    :return: 축에대한 미분 결과값 ( Gradient 값 )
    '''
    size = ksize // 2
    y, x = np.mgrid[-size:size + 1, -size:size+1]
    if gx == 1:
        DoG = (-x / (2*np.pi*(sigma**4)))*np.exp(-(x**2+y**2)/(2*sigma*sigma))

    else:
        DoG = (-y / (2*np.pi*sigma ** 4)) * np.exp(- (y ** 2+ x** 2) / (2 * sigma * sigma))  # apply gaussian formula to matrix


    DoG_img = my_filtering(img, DoG, boundary=boundary)
    return DoG_img

src = cv2.imread('/Users/moon/Desktop/컴그/2주차/lena.png', 1)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

LoG = my_LoG(gray, 15, boundary=0)
DoGX = my_DoG(gray, 15, sigma=3, gx=1, boundary=0)
DoGY = my_DoG(gray, 15, sigma=3, gx=0, boundary=0)
DoG = np.sqrt(DoGX**2 + DoGY**2)

cv2.imshow("DoG", DoG.astype(np.uint8))
cv2.imshow("LoG", LoG)

cv2.waitKey()
cv2.destroyAllWindows()

