import cv2
import numpy as np

'''
필요한 정보.
matchees의 member 
 - matches[i].distance : 해당 matching point간의 거리
 - matches[i].trainIdx : img1의 descriptor index (keypoint index)
 - matches[i].queryIdx : img2의 descriptor index (keypoint index)

kp의 member
 - kp[i].pt : 해당 keypoint의 좌표 ( x, y )
'''
#ORB를 사용해 keypoint를 구하고, 해당 descriptor로 matching을 수행한다.
def get_matchpoint(img1,img2,count):
    '''
    img1과 img2사이의 Affine matrix를 찾는다.
    :param img1: Affine matrix를 찾을 변환 전 이미지 1
    :param img2: Affine matrix로 변환이 된 이미지 2
    :param count: 찾을 feature의 수.
    :return : img1, img2의 keypoint. matching된 keypoints
    '''
    orb = cv2.ORB_create(nfeatures = count) # count는 찾을 feature의 수

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None) # img1, img2에서 각각 keypoints, descriptor를 얻음

    bf = cv2.BFMatcher(cv2.NORM_HAMMING) # Brute force 방식으로 Matching point를 찾음.

    matches = bf.match(des1, des2)

    matches = sorted(matches, key = lambda x:x.distance) #point 정렬 (거리순으로)

    return kp1, kp2, matches

#bilinear interpolation
def my_bilinear(img, x, y):
    '''
    :param img: 값을 찾을 img
    :param x: interpolation 할 x좌표
    :param y: interpolation 할 y좌표
    :return: img[x,y]에서의 value (bilinear interpolation으로 구해진)
    '''
    floorX, floorY = int(x), int(y)

    t, s = x-floorX, y-floorY

    zz = (1-t) * (1-s)
    zo = t * (1-s)
    oz = (1-t) * s
    oo = t * s

    interVal = img[floorY, floorX, :] * zz + img[floorY, floorX +1, : ] * zo + \
            img[floorY+1, floorX, :] * oz + img[floorY+1, floorX+1, :] * oo

    return interVal

#Least square 방식으로 행렬 M의 요소인 [a, b, c, d, e ,f].T 를 구한다.
def my_LS(matches, kp1, kp2):
    '''
    :param matches: keypoint matching 정보
    :param kp1: keypoint 정보.
    :param kp2: keypoint 정보2.
    :return: X : 위의 정보를 바탕으로 Least square 방식으로 구해진 Affine 변환 matrix의 요소 [a, b, c, d, e, f].T
    '''

    length = len(matches)
    A = np.zeros((length * 2, 6))
    B = np.zeros((length * 2, 1))

    '''행렬 A,B를 만들고 이를 이용해 X를 구해주세요.'''
    for i in range(length):
        x1, y1 = kp1[matches[i].queryIdx].pt
        x2, y2 = kp2[matches[i].trainIdx].pt
        j = i*2
        A[j, 0], A[j, 1], A[j, 2] = x1, y1, 1
        A[j+1, 3], A[j+1, 4], A[j+1, 5] = x1, y1, 1
        B[j, 0], B[j+1, 0] = x2, y2

    AInv = np.linalg.inv(A.T @ A)
    X = AInv @ A.T @ B
    return X

#forward warping을 수행한다.
# Mx = x'
def my_forward(img, X):
    '''
    :param img: warping 대상 image
    :param X: 변환행렬값 6개가 저장된 행렬 (6, 1)
    :return: forward warping이 된 image
    '''

    # 겹치는 좌표에 대한 처리는 값을 모두 더해서 평균냄.

    y, x, c = img.shape
    result = np.zeros((y,x,c))
    count = np.zeros((y,x,c))

    '''X를 이용해 Affine matrix M을 만드세요.'''
    M = np.array([[X[0, 0], X[1, 0], X[2, 0]],
                  [X[3, 0], X[4, 0], X[5, 0]],
                  [0, 0, 1]])
    for i in range(y):
        for j in range(x):
            '''
            M과 좌표 (x,y)를 이용해 새로운 좌표인 newX, newY를 구하세요
            .
            '''
            A = np.array((j, i, 1)).reshape(-1, 1)
            B = M @ A
            newX = int(round(B[0, 0]))
            newY = int(round(B[1, 0]))

            if newY < y and newX < x and newY > 0 and newX > 0:
                result[newY, newX, :] += img[i, j, :] #값을 해당 좌표에 합산
                count[newY, newX, :] += 1           #몇 개의 값이 한 좌표에 겹쳤는지 count

    result = np.divide(result, count)
    result[np.isnan(result)] = 0                    #divide 후 not a number 처리

    return result

#backward warping을 수행
#x = M.inv * x' 로 x를 구한 뒤, 해당 좌표에서의 값을 interpolation으로 구함.
def my_backward(img, X):
    '''
    :param img: warping 대상 image
    :param X: 변환행렬값 6개가 저장된 행렬 (6, 1)
    :return: backward warping이 된 image
    '''

    '''backward warping을 수행하는 코드를 작성하세요.'''

    y, x, c = img.shape
    result = np.zeros((y, x, c))

    M = np.array([[X[0, 0], X[1, 0], X[2, 0]],
                  [X[3, 0], X[4, 0], X[5, 0]],
                  [0, 0, 1]])

    for i in range(y):
        for j in range(x):
            '''
            M과 좌표 (x,y)를 이용해 새로운 좌표인 newX, newY를 구하세요
            .
            '''
            B = np.array((j, i, 1)).reshape(-1, 1)
            A = np.linalg.inv(M) @ B
            newX = A[0, 0]
            newY = A[1, 0]

            if newY < y-1 and newX < x-1 and newY > 1 and newX > 1:
                result[i, j, :] = my_bilinear(img, newX, newY)

    return result

if __name__ == "__main__":

    cap = cv2.VideoCapture('./test.mp4')
    count = 0
    while cap.isOpened():
        ret, fr = cap.read()

        if count == 224:
            first = fr
        elif count == 444:  # 적당한 frame을 추출
            target = fr
            break
        count += 1

    # first = cv2.imread('./LenaFaceShear.png')
    # target = cv2.imread('./Lena.png') #대상이 되는 두 이미지.

    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    #찾을 feature의 수는 100개, matching을 수행.
    feature_count = 100
    kp1, kp2, matches = get_matchpoint(first_gray, target_gray, feature_count)

    #Least square 방법으로 matrix M의 값을 구한다.
    #matches중 distance가 가까운 순으로 10개만 사용해 계산한다.
    X = my_LS(matches[0:10], kp1, kp2)

    forward_result = my_forward(first, X)
    backward_result = my_backward(first, X)

    cv2.imshow('first', first)
    cv2.imshow('target', target)
    cv2.imshow('forward', forward_result.astype(np.uint8))
    cv2.imshow('backward', backward_result.astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()

