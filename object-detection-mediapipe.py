import cv2
import numpy as np
import mediapipe as mp



cap=cv2.VideoCapture('30.mp4')

flag=1
whT=320
classesFile='data\obj.names'
classNames=[]
confThreshold=0.5
nmsThreshold=0.3

pts=[]

mpDraw=mp.solutions.drawing_utils
mpPose=mp.solutions.pose
pose=mpPose.Pose()

with open(classesFile,'rt') as f:
    classNames=f.read().rstrip('\n').split('\n')

#print(classNames)
#print(len(classNames))
modelConfiguration='cfg\yolov4-obj.cfg'
modelWeights='backup\yolov4-obj_6000.weights'

net=cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

feature_params = dict(maxCorners=1,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                           10, 0.03))

color = np.random.randint(0, 255, (100, 3))

def findObjects(outputs,img):
    hT, wT, cT=img.shape
    bbox= []
    classIds=[]
    confs=[]
    for output in outputs:
        for detection in output:
            scores=detection[5:]
            #print(scores)
            classId=np.argmax(scores)
            confidence=scores[classId]
            #print(confidence)
            if confidence > confThreshold:
                w,h=int(detection[2]*wT), int(detection[3]*hT)
                x,y=int((detection[0]*wT) - w/2), int((detection[1]*hT) - h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    #print(len(bbox))
    indices=cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
    #print(indices)
    for i in indices.flatten():
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        label=str(classNames[classIds[i]])
        confi=str(int(confs[i]*100))+'%'
        global flag, pts
        '''
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, label + " " + confi,
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        '''
        if label == 'Goalpost':
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, label + " " + confi,
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            goal = img[y:y + h, x:x + w]
            cv2.line(goal, (x+70,y-h),(x+70, y + h),(0,255,0),3)
            cv2.line(goal, (x-w, y ), (x+w, y ), (0, 255, 0), 3)
            cv2.putText(img,  "1" ,
                        (x+10, y+20 ), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
            cv2.putText(img, "2",
                        (x + w-20, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(img, "3",
                        (x + 10, y + h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(img, "4",
                        (x + w - 20, y + h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        if label=='Kicker':
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(img,label+" "+confi,
                        (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)

            kicker=img[y:y+h, x:x+w]    # ashol  jinissh
            imgRGB = cv2.cvtColor(kicker, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)
            if results.pose_landmarks:
                mpDraw.draw_landmarks(kicker, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = kicker.shape
                    print(id, lm)
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(kicker, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        if label == 'Football':
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(img, label + " " + confi,
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


            football = img[y:y + h, x:x + w]

            old_point = (x+int(w/2), y+int(h/2))
            pts.append(old_point)

            for i in range(0, len(pts)):
                # if either of the tracked points are None, ignore
                # them
                #if pts[i - 1] is None or pts[i] is None:
                 #   continue

                cv2.line(img, pts[i - 1], pts[i], (255, 0, 0), 2)


                #img = cv2.circle(img, old_point, 5,
                 #                    (0, 0, 255), -1)
                #old_point = (x, y)
        if label == 'Goalkeeper':
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, label + " " + confi,
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)





while True:
    success,img =cap.read()
    img = cv2.resize(img, (1280, 720))

    # print(results.pose_landmarks)


    blob=cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop=False)
    net.setInput(blob)

    layerNames=net.getLayerNames()
    #print(layerNames)
    #print(net.getUnconnectedOutLayers())
    outputNames=[layerNames[i-1] for i in net.getUnconnectedOutLayers()]


    #print(outputNames)
    #print(net.getUnconnectedOutLayers())

    outputs=net.forward(outputNames)
    #print(outputs[0].shape)
    #print(outputs[1].shape)
    #print(outputs[2].shape)
    #print(outputs[0][0])

    findObjects(outputs,img)




    cv2.imshow('Image',img)
    cv2.waitKey(10)


