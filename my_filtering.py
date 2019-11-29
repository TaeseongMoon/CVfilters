import cv2
import numpy as np
import time

def my_padding(img, shape, boundary = 0):
    '''
    :param img: boundary padding을 해야할 이미지
    :param shape: kernel의 shape
    :param boundary: default = 0, zero-padding : 0, repetition : 1, mirroring : 2
    :return: padding 된 이미지.
    '''
    row, col = len(img), len(img[0])
    pad_sizeY, pad_sizeX = shape[0] // 2, shape[1] // 2
    res = np.zeros((row + (2 * pad_sizeY), col + (2 * pad_sizeX)), dtype=np.float)
    pad_row, pad_col = len(res), len(res[0])
    if pad_sizeY == 0 :
        res[pad_sizeY:, pad_sizeX:-pad_sizeX] = img.copy()
    elif pad_sizeX == 0:
        res[pad_sizeY:-pad_sizeY, pad_sizeX:] = img.copy()
    else:
        res[pad_sizeY:-pad_sizeY, pad_sizeX:-pad_sizeX] = img.copy()
    if boundary == 0:
        return res
    elif boundary == 1:
        res[0:pad_sizeY, 0:pad_sizeX] = img[0, 0]  # 좌측 상단
        res[-pad_sizeY:, 0:pad_sizeX] = img[row - 1, 0]  # 좌측 하단
        res[0:pad_sizeY, -pad_sizeX:] = img[0, col - 1]  # 우측 상단
        res[-pad_sizeY:, -pad_sizeX:] = img[row - 1, col - 1]  # 우측 하단
        # axis = 1, 열반복, axis = 0, 행반복. default 0
        res[0:pad_sizeY, pad_sizeX:pad_col-pad_sizeX] = np.repeat(img[0:1, 0:], [pad_sizeY], axis=0)  # 상단
        res[pad_row-pad_sizeY:, pad_sizeX:pad_col-pad_sizeX] = np.repeat(img[row - 1:row, 0:], [pad_sizeY], axis=0)  # 하단
        res[pad_sizeY:pad_row-pad_sizeY, 0:pad_sizeX] = np.repeat(img[0:, 0:1], [pad_sizeX], axis=1)  # 좌측
        res[pad_sizeY:pad_row-pad_sizeY, pad_col-pad_sizeX:] = np.repeat(img[0:, col - 1:col], [pad_sizeX], axis=1)  # 우측
        return res
    else:
        res[0:pad_sizeY, 0:pad_sizeX] = np.flip(img[0:pad_sizeY, 0:pad_sizeX])  # 좌측 상단
        res[-pad_sizeY:, 0:pad_sizeX] = np.flip(img[-pad_sizeY:, 0:pad_sizeX])  # 좌측 하단
        res[0:pad_sizeY, -pad_sizeX:] = np.flip(img[0:pad_sizeY, -pad_sizeX:])  # 우측 상단
        res[-pad_sizeY:, -pad_sizeX:] = np.flip(img[-pad_sizeY:, -pad_sizeX:])  # 우측 하단

        res[pad_sizeY:pad_row-pad_sizeY, 0:pad_sizeX] = np.flip(img[0:, 0:pad_sizeX], 1)  # 좌측
        res[pad_sizeY:pad_row-pad_sizeY, pad_col-pad_sizeX:] = np.flip(img[0:, col-pad_sizeX:], 1)  # 우측
        res[0:pad_sizeY, pad_sizeX:pad_col-pad_sizeX] = np.flip(img[0:pad_sizeY, 0:], 0)  # 상단
        res[pad_row-pad_sizeY:, pad_sizeX:pad_col-pad_sizeX] = np.flip(img[row-pad_sizeY:, 0:], 0)  # 하단
        return res


def my_getGKernel(shape, sigma):
    '''
    :param shape: 생성하고자 하는 gaussian kernel의 shape입니다. (5,5) (1,5) 형태로 입력받습니다.
    :param sigma: Gaussian 분포에 사용될 표준편차입니다. shape가 커지면 sigma도 커지는게 좋습니다.
    :return: shape 형태의 Gaussian kernel
    '''
    #Gaussian kernel 생성 코드를 작성해주세요.
    sizeY, sizeX = shape[0]//2, shape[1]//2      #kernel size
    y, x = np.mgrid[-sizeY: sizeY+1, -sizeX: sizeX+1] #make kernel size of matrix
    gaus = np.exp(- (x ** 2 + y ** 2 )/(2*sigma*sigma)) #apply gaussian formula to matrix
    gaus = gaus / np.sum(gaus) #averaging kernel
    return gaus

def my_filtering(img, kernel, boundary = 0):
    '''
    :param img: Gaussian filtering을 적용 할 이미지
    :param kernel: 이미지에 적용 할 Gaussian Kernel
    :param boundary: 경계 처리에 대한 parameter (0 : zero-padding, default, 1: repetition, 2:mirroring)
    :return: 입력된 Kernel로 gaussian filtering이 된 이미지.
    '''
    row, col = len(img), len(img[0])
    ksizeY, ksizeX = kernel.shape[0], kernel.shape[1]

    pad_image = my_padding(img, (ksizeY, ksizeX), boundary=boundary) # 경계가 padding된 이미지 생성.
    filtered_img = np.zeros((row,col), dtype = np.float32)
    for i in range(row):
        for j in range(col):
            #filter convolution
            filtered_img[i, j] = np.sum((pad_image[i:i + ksizeY, j:j + ksizeX])*kernel)

    return filtered_img

# src = cv2.imread('/Users/moon/Desktop/컴그/2주차/sky.jpeg', 0) #read Grayscale image
# gaus2D = my_getGKernel((5,5), 13)
# gaus1D = my_getGKernel((1,5), 13)
#
# start = time.perf_counter() # 시간 측정
# img2D = my_filtering(src, gaus2D,boundary = 0) #2D filtering
# end = time.perf_counter()
# print(end-start)
#
# start = time.perf_counter()
# img1D = my_filtering(src, gaus1D, boundary = 0)       # 1D * 1D  filtering
# img1D = my_filtering(img1D, gaus1D.T, boundary = 0)
# end = time.perf_counter()
# print(end-start)
#
# cv2.imshow('img1D', img1D.astype(np.uint8))
# cv2.imshow('img2D', img2D.astype(np.uint8))
# residual = img2D - img2D #residual
# print(sum(sum(residual))) #checking residual
# cv2.waitKey()
# cv2.destroyAllWindows()