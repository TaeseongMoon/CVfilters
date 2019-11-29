import cv2
import numpy as np
import matplotlib.pyplot as plt

def my_divHist(fr):
    '''
    :param fr: 3x3 (9) 등분으로 분할하여 Histogram을 계산할 이미지.
    :return: length (216) 혹은 (216,1) array ( histogram )
    '''
    y, x = fr.shape[0], fr.shape[1]
    div = 3 # 3x3 분할
    divY, divX = y // div, x // div # 3등분 된 offset 계산.
    gethist = np.zeros((9, 24))

    # cell 단위의 histogram을 계산하기 위해 필요한 작업 및 계산을 수행하세요.

    for i in range(div):
        for j in range(div):
            gethist[i, :] = my_hist(fr[i*divY:(i+1)*divY, j*divX:(j+1)*divX])

    hist = gethist[0]
    for i in range(1,9):
        hist = np.concatenate([hist, gethist[i]])
    return hist

#color histogram 생성.
def my_hist(fr):
    '''
    :param fr: histogram을 구하고자 하는 대상 영역
    :return: fr의 color histogram
    '''
    blue = fr[:, :, 2] // 32
    green = fr[:, :, 1] // 32
    red = fr[:, :, 0] // 32
    hist = np.zeros((3, 8))
    hist[0, :] = np.add(hist[0, :], np.bincount(blue.flatten(), minlength=8))
    hist[1, :] = np.add(hist[1, :], np.bincount(green.flatten(), minlength=8))
    hist[2, :] = np.add(hist[2, :], np.bincount(red.flatten(), minlength=8))

    hist = np.concatenate((hist[0], hist[1], hist[2]))
    return hist

#주변을 탐색해, 최단 거리를 가진 src의 영역을 return
def get_minDist(src, target, start):
    '''
    :param src: target을 찾으려는 이미지
    :param target: 찾으려는 대상
    :param start : 이전 frame에서 target이 검출 된 좌표 ( 좌측 상단 ) ( y, x )
    :return: target과 최소의 거리를 가진 영역(사각형) 좌표. (좌상단x, 좌상단y, 우하단x, 우하단y)
    '''
    sy, sx = src.shape[0], src.shape[1]
    ty, tx = target.shape[0], target.shape[1]
    offset_y, offset_x = start[0], start[1]
    hist = my_hist(target)
    min = 100000000
    minIndex= np.zeros(2)
    for i in range(offset_y-20, offset_y+20):
        for j in range(offset_x-20, offset_x+20):
            tempHist = my_hist(src[i:i + ty, j:j + tx])
            sum = 0
            for k in range(tempHist.shape[0]):
                if hist[k] == 0 and tempHist[k] == 0:
                    continue
                else:
                    sum += (hist[k] - tempHist[k])**2 / (hist[k] + tempHist[k])
            if min > sum:
                min = sum
                minIndex = [j, i]

    x, y = int(minIndex[0]), int(minIndex[1])
    coord = [x, y, x+tx, y+ty]
    return coord

# Mouse Event를 setting 하는 영역
roi = None
drag_start = None
mouse_status = 0
tracking_strat = False
def onMouse(event, x, y, flags, param=None):
    global roi
    global drag_start
    global mouse_status
    global tracking_strat
    if event == cv2.EVENT_LBUTTONDOWN:
        drag_start = (x,y)
        mouse_status = 1 #Left button down
        tracking_strat = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if flags == cv2.EVENT_FLAG_LBUTTON:
            xmin = min(x, drag_start[0])
            ymin = min(y, drag_start[1])
            xmax = max(x, drag_start[0])
            ymax = max(y, drag_start[1])
            roi = (xmin, ymin, xmax, ymax)
            mouse_status = 2 # dragging
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_status = 3 #complete

# Window를 생성하고, Mouse event를 설정
cv2.namedWindow('tracking')
cv2.setMouseCallback('tracking', onMouse)

#Video capture
cap = cv2.VideoCapture('./ball.wmv')
if not cap.isOpened():
    print('Error opening video')
h, w = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
fr_roi = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if fr_roi is not None: #fr_roi가 none이 아닐 때만
        x1,y1,x2,y2 = get_minDist(frame, fr_roi, start)
        start = (y1,x1)
        cv2.rectangle(frame,(x1,y1), (x2,y2), (255, 0,0), 2)

    if mouse_status == 2: #Mouse를 dragging 중일 때
        x1,y1,x2,y2 = roi
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)

    if mouse_status == 3: #Mouse를 놓아서 영역이 정상적으로 지정되었을 때.
        mouse_status = 0
        x1, y1, x2, y2 = roi
        start = (y1,x1)
        fr_roi = frame[y1:y2, x1:x2]

    cv2.imshow('tracking', frame)
    key = cv2.waitKey(100) #지연시간 100ms
    if key == ord('c'): # c를 입력하면 종료.
        break

if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()