import cv2
import math
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def calculateVector(keypoint, Ix, Iy, It):
    x, y = keypoint.pt
    x = int(x)
    y = int(y)

    A = np.zeros((2, 2))
    B = np.zeros((2, 1))

    window_size = 3
    offset = int(3 / 2)

    for i in range(y - offset, y + offset, 1):
        for j in range(x - offset, x + offset, 1):
            A[0, 0] += Ix[i, j] * Ix[i, j]
            A[0, 1] += Ix[i, j] * Iy[i, j]
            A[1, 0] += Ix[i, j] * Iy[i, j]
            A[1, 1] += Iy[i, j] * Iy[i, j]

            B[0] += Ix[i, j] * It[i, j]
            B[1] += Iy[i, j] * It[i, j]

    Ainv = np.linalg.inv(A)
    return (np.array([x, y]), np.matmul(Ainv, B))


def calculateVectors(fast, prev_frame, cur_frame):
    keypoints = fast.detect(prev_frame, None)

    kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    prev_frame = prev_frame.astype('float32')
    cur_frame = cur_frame.astype('float32')

    Ix_prev = signal.convolve2d(prev_frame, kernel_x, boundary='symm', mode='same')
    Iy_prev = signal.convolve2d(prev_frame, kernel_y, boundary='symm', mode='same')
    Ix_cur = signal.convolve2d(cur_frame, kernel_x, boundary='symm', mode='same')
    Iy_cur = signal.convolve2d(cur_frame, kernel_y, boundary='symm', mode='same')

    Ix = (Ix_prev + Ix_cur) / 2
    Iy = (Iy_prev + Iy_cur) / 2
    It = cur_frame - prev_frame

    coordinates = np.zeros((len(keypoints), 2))
    vectors = np.zeros(coordinates.shape)

    for idx, kp in enumerate(keypoints):
        coordinate, vector = calculateVector(kp, Ix, Iy, It)
        coordinates[idx] = coordinate
        vectors[idx] = np.squeeze(vector)

    return (coordinates, vectors)

def createFlowVisualization(frame, coordinates, vectors):
    for idx, coordinate in enumerate(coordinates):
        x = int(coordinate[0])
        y = int(coordinate[1])
        u = vectors[idx, 0]
        v = vectors[idx, 1]
        magnitude = math.sqrt(u ** 2 + v ** 2) * 10
        frame[y, x] = magnitude
    frame *= (255 / np.max(frame))
    return frame.astype('uint8')

if __name__ == '__main__':
    # Capture from webcam
    cap = cv2.VideoCapture(0)
    fast = cv2.FastFeatureDetector_create()
    
    prev_frame = np.empty(0)

    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cur_frame = cv2.resize(gray_frame, (640, 360), fx = 0, fy = 0,
                               interpolation = cv2.INTER_CUBIC)

        if not prev_frame.any():
            prev_frame = cur_frame
            continue

        prev_frame = cv2.GaussianBlur(prev_frame, (7, 7), 0)
        cur_frame = cv2.GaussianBlur(cur_frame, (7, 7), 0)

        coordinates, vectors = calculateVectors(fast, prev_frame, cur_frame)

        frame = np.zeros(cur_frame.shape)
        result = createFlowVisualization(frame, coordinates, vectors)

        prev_frame = cur_frame
        
        cv2.imshow('Frame', result)

        # Press 'q' to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
     
    cap.release()
    cv2.destroyAllWindows()
