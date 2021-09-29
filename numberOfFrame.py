import cv2
cap= cv2.VideoCapture('round2.mp4')

totalframecount= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("The total number of frames in this video is ", totalframecount)
