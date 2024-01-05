import cv2
import time
import numpy as np
from datetime import datetime
import tensorflow as tf
from threading import Thread, Lock

import keyboard

global currentcounter
global analytics_flag


class CameraStream(object):
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)

        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

        self.show_text = False  

    def start(self):
        if self.started:
            #print("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()
            time.sleep(.05)

    def read(self):
        try:
            self.read_lock.acquire()
            frame = self.frame.copy()
            self.read_lock.release()
            return frame
        except:
            pass

    def stop(self):
        self.started = False
        self.thread.join(timeout=1)

    def __exit__(self, exc_type, exc_value, traceback):
        self.stream.release()

def image_inference():

    frame_size = (512,512)
    currentcounter = 0
    analytics_flag = False

    while video_capture:

        time.sleep(.01)

        image_np_original = video_capture.read()

        currentcounter+=1

        try:
            if image_np.shape == 0:
                time.sleep(1)
                break
        except:
            continue

        key = cv2.waitKey(1)

        try:  # used try so that if user pressed other than the given key error will not be shown
            if keyboard.is_pressed('s'):  
                print('You Pressed start Key! Analytics will start now')
                analytics_flag = True
        except:
            break  # if user pressed a key other than the given key the loop will break

        if analytics_flag: 
            cv2.putText(image_np,"Analytics started ",(512,20),cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0),1)
                   
            image_np_original = image_np_original.copy()
            dim = image_np_original.shape[0:2]
            image_np = cv2.resize(image_np_original, frame_size)
            
                    
            # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
            input_tensor = tf.convert_to_tensor(image_np)
            # The model expects a batch of images, so add an axis with `tf.newaxis`.
            input_tensor = input_tensor[tf.newaxis, ...]
            
            #input_tensor = np.expand_dims(image_np, 0)
            detections = detect_fn(input_tensor)
            
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
            detections['num_detections'] = num_detections
            
            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
            
            boxes=(detections['detection_boxes'])
            classes =(detections['detection_classes'])
            scores = (detections['detection_scores'])
                        
            cls = classes.tolist()
            
            dets = []
            for i, value in enumerate(cls):
                if scores[i] > .5:
                    if int(value)<3:
                        (y1, x1) = (boxes[i][0], boxes[i][1])
                        (y2, x2) = (boxes[i][2], boxes[i][3])                        
                        dets.append([x1,y1,x2,y2,scores[i],cls[i]])

            for variables in dets :
                x1, y1, x2, y2, score, cls = variables
            
                pt1 = (int(x1 * dim[0]), int(y1 * dim[1])) 
                pt2 = (int(x2 * dim[0]), int(y2 * dim[1]))

                # class 1 : horizontal , class 2 : vertical

                if cls == 1:
                    cv2.rectangle(image_np_original, pt1, pt2, (0, 255,0), 1)  # green rectangle
                if cls == 2:
                    cv2.rectangle(image_np_original, pt1, pt2, (0, 0,255), 1)  # red rectangle
                    cv2.putText(image_np_original,"Anomaly Detected",(20,50),cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),1)

        cv2.imshow('Antenna', image_np_original)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            print("Stoping the analysis since Q key is pressed") 
            cv2.destroyAllWindows()
            break
            
if __name__ == '__main__':
    
    PATH_TO_SAVED_MODEL = "saved_model"
    video_path = "video.mp4" 
    first_frame=None
    
    tf.saved_model.LoadOptions(experimental_io_device='/job:localhost',)

    # Load saved model and build the detection function
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    print("Model has been loaded ")
    
    video_capture = CameraStream(video_path).start()

    image_inference()

    video_capture.stop()
    
    cv2.destroyAllWindows()    
