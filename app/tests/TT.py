import numpy as np
import cv2

# frame1 = (cv2.imread('basketball1.png', cv2.IMREAD_GRAYSCALE))
# frame2 = (cv2.imread('basketball2.png', cv2.IMREAD_GRAYSCALE))
frame1 = np.array([])
frame2 = np.array([])

cv2.cuda.NvidiaOpticalFlow_1_0_create(width=512, height=512)
nvof = cv2.cuda_NvidiaOpticalFlow_1_0.create(frame1.shape[1], frame1.shape[0], 5, False, False, False, 0)

flow = nvof.calc(frame1, frame2, None)

flowUpSampled = nvof.upSampler(flow[0], frame1.shape[1], frame1.shape[0], nvof.getGridSize(), None)

cv2.writeOpticalFlow('OpticalFlow.flo', flowUpSampled)

nvof.collectGarbage()